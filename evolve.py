import json, os, sys, time, copy, random, threading
from concurrent.futures import ThreadPoolExecutor
from llm import LLM, SUPER, extract_json
from team import Team, Genome, show, console
from tricks import BASE_TRICKS, make_task
from trickster import invent_trick, compile_trick
from evaluate import evaluate, summary_line
from data import TRAIN_DOMAINS

os.makedirs("runs", exist_ok=True)
LOG = open("runs/live_log.jsonl", "a", encoding="utf-8")
LOCK = threading.Lock()


def emit(tid, kind, msg):
    """Show on screen AND record for the replay demo."""
    show(tid, kind, msg)
    with LOCK:
        LOG.write(json.dumps({"t": time.time(), "tid": tid, "kind": kind, "msg": msg}) + "\n")
        LOG.flush()


REFLECT_SYS = """You coach a data-cleaning agent team with two members:
SPEC writer (writes the definition-of-done checks) and BUILDER (writes the clean(raw) code).
You see its current lessons and its HIDDEN mistakes on practice jobs (real errors versus the true answer,
and whether it falsely claimed to be done).
Propose exactly ONE new lesson that fixes the MOST COMMON mistake on FUTURE, DIFFERENT datasets.
Choose target "build" when the code produced wrong values or dropped rows (most mistakes are like this).
Choose target "spec" only when the checks let clearly wrong output pass as done.
The lesson must be GENERAL and CONCRETE (a specific technique, e.g. which normalization to apply first),
never about a specific id or row. Max 2 sentences. Do not repeat an existing lesson.
Return JSON only: {"target": "spec" or "build", "lesson": "...", "why": "...", "predicted_gain_pct": number}"""


def reflect(llm, g, results):
    fails = [{"job": r.task_id, "verdict": r.verdict, "real_accuracy": round(r.acc, 3),
              "hidden_mistakes": r.errors[:8]} for r in results if r.acc < 1.0]
    if not fails:
        return None, {}
    user = json.dumps({"current_spec_lessons": g.spec_lessons, "current_build_lessons": g.build_lessons,
                       "practice_failures": fails[:6]}, indent=1, ensure_ascii=False)[:12000]
    m = extract_json(llm.chat(SUPER, REFLECT_SYS, user, max_tokens=1500, temperature=0.9)) or {}
    lesson = str(m.get("lesson", "")).strip()
    if len(lesson) < 15:
        return None, m
    child = copy.deepcopy(g)
    target = "spec" if m.get("target") == "spec" else "build"
    if target == "spec":
        child.spec_lessons = (child.spec_lessons + [lesson])[-8:]
    else:
        child.build_lessons = (child.build_lessons + [lesson])[-8:]
    child.note = f"{target}: {lesson[:110]}"
    return child, m


def strip(s):
    return {k: v for k, v in s.items() if k != "results"} if s else None


def save_state(archive, tricks, best):
    with open("runs/archive.json", "w", encoding="utf-8") as f:
        json.dump([{k: (v.__dict__ if isinstance(v, Genome) else v) for k, v in a.items()} for a in archive], f, indent=2)
    with open("runs/tricks.json", "w", encoding="utf-8") as f:
        json.dump([{"name": n, "code": c} for n, _, c in tricks], f, indent=2)
    with open("runs/best.json", "w", encoding="utf-8") as f:
        json.dump(best.__dict__, f, indent=2)


def load_resume():
    with open("runs/best.json", encoding="utf-8") as f:
        best = Genome(**json.load(f))
    tricks = []
    with open("runs/tricks.json", encoding="utf-8") as f:
        for t in json.load(f):
            fn, _ = compile_trick(t["code"])
            if fn:
                tricks.append((t["name"], fn, t["code"]))
    with open("runs/archive.json", encoding="utf-8") as f:
        archive = json.load(f)
    return best, tricks, archive


def main(generations, resume):
    llm = LLM(budget_usd=35)
    team = Team(llm, emit=emit)
    best, archive, tricks = Genome(), [], []
    if resume and os.path.exists("runs/best.json"):
        best, tricks, archive = load_resume()
        emit("RESUME", "SCORE", f"continuing from {best.gid} with {len(tricks)} invented tricks")
    start = len(archive)
    t0 = time.time()
    weak = []
    for gen in range(start + 1, start + generations + 1):
        tag = f"GEN {gen}"
        emit(tag, "TRICK", "Trickster is inventing a new trick...")
        history = f"The Cleaner still makes mistakes like these, target them: {weak[:6]}" if weak else ""
        got = invent_trick(llm, list(BASE_TRICKS) + [n for n, _, _ in tricks], history, verbose=False)
        if got:
            tricks.append(got)
            emit(tag, "TRICK", f"new trick: {got[0]}")
        else:
            emit(tag, "TRICK", "Trickster failed to make a fair trick this round")
        active = [fn for _, fn, _ in tricks[-4:]]

        domains = random.Random(gen).sample(TRAIN_DOMAINS, 4)
        practice = [make_task(d, 200 + gen * 10 + i, active) for i, d in enumerate(domains)]
        tr = evaluate(team, best, practice, workers=4)
        emit(tag, "SCORE", summary_line(f"{best.gid} practice", tr))
        weak = [e for r in tr["results"] for e in r.errors[:2]]

        child, prop = reflect(llm, best, tr["results"])
        if child is None:
            emit(tag, "REJECT", "coach had no usable lesson")
            archive.append({"gen": gen, "kept": False, "genome": best, "reason": "no lesson"})
            save_state(archive, tricks, best)
            continue
        parent_gid = best.gid
        child.gid, child.parent = f"v{gen}", parent_gid
        emit(tag, "COACH", f"{child.gid} proposes {child.note}")

        val = [make_task(d, 50 + gen, active) for d in TRAIN_DOMAINS]
        with ThreadPoolExecutor(2) as ex:
            fp, fc = ex.submit(evaluate, team, best, val, 5), ex.submit(evaluate, team, child, val, 5)
            ps, cs = fp.result(), fc.result()
        gain = cs["accuracy"] - ps["accuracy"]
        honest = cs["false_done"] <= ps["false_done"]
        try:
            predicted = float(prop.get("predicted_gain_pct", 0)) / 100
        except (TypeError, ValueError):
            predicted = 0.0
        mirage = predicted - gain
        line = (f"{child.gid} vs {parent_gid}: accuracy {ps['accuracy']:.1%} -> {cs['accuracy']:.1%} ({gain:+.1%}) | "
                f"false-done {ps['false_done']} -> {cs['false_done']} | coach predicted {predicted:+.0%}, "
                f"mirage gap {mirage:+.0%}")
        kept = gain >= 0.02 and honest
        if kept:
            emit(tag, "KEEP", line)
            best = child
        else:
            why = "no real gain" if gain < 0.02 else "more false-dones"
            emit(tag, "REJECT", f"{line} | {why}")
        archive.append({"gen": gen, "kept": kept, "genome": child, "parent": parent_gid,
                        "parent_val": strip(ps), "child_val": strip(cs), "gain": gain,
                        "predicted": predicted, "mirage": mirage, "trick": got[0] if got else None})
        save_state(archive, tricks, best)
        console.print(f"[dim]spent ${llm.spent:.2f} | {time.time() - t0:.0f}s so far[/dim]")

    emit("DONE", "SCORE", f"best version: {best.gid} | lessons: {len(best.spec_lessons) + len(best.build_lessons)}")
    console.print(f"COST ${llm.spent:.2f} | time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    nums = [a for a in sys.argv[1:] if a.isdigit()]
    main(int(nums[0]) if nums else 3, "--resume" in sys.argv)