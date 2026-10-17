import json, time
from concurrent.futures import ThreadPoolExecutor
from rich.table import Table
from llm import LLM
from team import Team, Genome, console
from tricks import make_task
from trickster import compile_trick
from evaluate import evaluate, heldout_tasks, save
from data import HELDOUT_DOMAINS

UNFAIR = ["missing first character", "duplicate prefix"]   # human review: these destroy information


def load():
    with open("runs/best.json", encoding="utf-8") as f:
        best = Genome(**json.load(f))
    fns, skipped = [], []
    with open("runs/tricks.json", encoding="utf-8") as f:
        for t in json.load(f):
            if any(bad in t["name"].lower() for bad in UNFAIR):
                skipped.append(t["name"])
                continue
            fn, _ = compile_trick(t["code"])
            if fn:
                fns.append((t["name"], fn))
    with open("runs/heldout_v0.json", encoding="utf-8") as f:
        v0_std = json.load(f)
    return best, fns, skipped, v0_std


def hard_tasks(fns):
    return [make_task(d, s, [fn for _, fn in fns]) for d in HELDOUT_DOMAINS for s in (300, 301, 302)]


def cell(s, key):
    if key == "false_done":
        return f"{s['false_done']}/{s['claimed']} ({s['false_done_rate']:.0%})"
    if key in ("accuracy", "solved"):
        return f"{s[key]:.1%}"
    return str(s[key])


if __name__ == "__main__":
    best, fns, skipped, v0_std = load()
    llm = LLM(budget_usd=45)
    team = Team(llm)
    t0 = time.time()
    hard = hard_tasks(fns)
    console.print(f"[bold]Final test: v0 vs {best.gid} | {len(fns)} invented tricks in hard mode[/bold]")
    if skipped:
        console.print(f"[yellow]Excluded as unfair (destroy information): {skipped}[/yellow]")
    with ThreadPoolExecutor(3) as ex:
        a = ex.submit(evaluate, team, best, heldout_tasks(), 4)
        b = ex.submit(evaluate, team, Genome(), hard, 4)
        c = ex.submit(evaluate, team, best, hard, 4)
        best_std, v0_hard, best_hard = a.result(), b.result(), c.result()
    save("runs/heldout_best.json", f"{best.gid} held-out (after)", best, best_std)
    save("runs/hard_v0.json", "v0 hard held-out", Genome(), v0_hard)
    save("runs/hard_best.json", f"{best.gid} hard held-out", best, best_hard)

    cols = [("Hidden tasks: v0 (before)", v0_std), (f"Hidden tasks: {best.gid} (after)", best_std),
            ("Hard mode: v0", v0_hard), (f"Hard mode: {best.gid}", best_hard)]
    rows = [("Real accuracy", "accuracy"), ("Tasks fully solved", "solved"),
            ("False 'done' claims", "false_done"), ("Escalations to human", "escalations")]
    table = Table(title="UNFOOLABLE: before vs after (tasks never seen during training)")
    table.add_column("Metric")
    for name, _ in cols:
        table.add_column(name, justify="center")
    for label, key in rows:
        table.add_row(label, *[cell(s, key) for _, s in cols])
    console.print(table)

    lessons = best.spec_lessons + best.build_lessons
    md = ["# Results", "", "| Metric | " + " | ".join(n for n, _ in cols) + " |",
          "|---|" + "---|" * len(cols)]
    md += [f"| {label} | " + " | ".join(cell(s, key) for _, s in cols) + " |" for label, key in rows]
    md += ["", f"Held-out data kinds (never used in training): {', '.join(HELDOUT_DOMAINS)}", "",
           f"## Lessons {best.gid} taught itself", *[f"- {x}" for x in lessons], "",
           "## Fair tricks used in hard mode", *[f"- {n}" for n, _ in fns], "",
           "## Tricks excluded as unfair (the Trickster tried to cheat)", *[f"- {n}" for n in skipped], "",
           f"Final test cost ${llm.spent:.2f} on Crusoe (100% NVIDIA Nemotron)."]
    with open("RESULTS.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    console.print(f"COST ${llm.spent:.2f} | time {time.time() - t0:.0f}s | wrote RESULTS.md")