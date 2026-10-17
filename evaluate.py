import json, os, time
from statistics import mean
from concurrent.futures import ThreadPoolExecutor
from llm import LLM
from team import Team, Genome, Result, show, console, DONE_BAR
from tricks import make_task
from data import TRAIN_DOMAINS, HELDOUT_DOMAINS


def train_tasks(extra=None):
    return [make_task(d, s, extra) for d in TRAIN_DOMAINS for s in (1, 2)]


def val_tasks(extra=None):
    return [make_task(d, 50, extra) for d in TRAIN_DOMAINS]


def heldout_tasks():
    """Never used for learning. Different data kinds the team has never seen."""
    return [make_task(d, s) for d in HELDOUT_DOMAINS for s in (100, 101, 102)]


def evaluate(team, g, tasks, workers=4):
    def safe_run(t):
        try:
            return team.run(g, t)
        except Exception as e:
            show(t["id"], "ESCALATE", f"crashed: {str(e)[:120]}")
            return Result(t["id"], verdict="CRASH")
    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(safe_run, tasks))
    claimed = [r for r in results if r.claimed_done]
    false_done = [r for r in claimed if r.acc < DONE_BAR]
    return {
        "accuracy": mean(r.acc for r in results),
        "solved": mean(r.acc >= DONE_BAR for r in results),
        "claimed": len(claimed),
        "false_done": len(false_done),
        "false_done_rate": len(false_done) / len(claimed) if claimed else 0.0,
        "escalations": sum(r.decision == "ESCALATE" for r in results),
        "n": len(results),
        "results": results,
    }


def summary_line(name, s):
    return (f"{name}: accuracy {s['accuracy']:.1%} | solved {s['solved']:.0%} | "
            f"false-done {s['false_done']}/{s['claimed']} claims ({s['false_done_rate']:.0%}) | "
            f"escalations {s['escalations']} | tasks {s['n']}")


def save(path, name, g, s):
    os.makedirs("runs", exist_ok=True)
    data = {k: v for k, v in s.items() if k != "results"}
    data["genome"] = g.__dict__
    data["per_task"] = [(r.task_id, r.verdict, round(r.acc, 3)) for r in s["results"]]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"name": name, **data}, f, indent=2)


if __name__ == "__main__":
    llm = LLM()
    team = Team(llm)
    t0 = time.time()
    v0 = Genome()
    s = evaluate(team, v0, heldout_tasks())
    show("BEFORE", "SCORE", summary_line("v0 on held-out", s))
    save("runs/heldout_v0.json", "v0 held-out (before)", v0, s)
    console.print(f"COST ${llm.spent:.2f} | time {time.time() - t0:.0f}s")