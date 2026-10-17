import json, time
from concurrent.futures import ThreadPoolExecutor
from rich.table import Table
from llm import LLM
from team import Team, Genome, console
from evaluate import evaluate, heldout_tasks, save
from final import load, hard_tasks, cell


def jl(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    best, fns, skipped, _ = load()
    saved = {"v0 std": jl("runs/heldout_v0.json"), "v0 hard": jl("runs/hard_v0.json"),
             "best std": jl("runs/heldout_best.json"), "best hard": jl("runs/hard_best.json")}
    llm = LLM(budget_usd=45)
    team_a = Team(llm, auditor=True)
    std, hard = heldout_tasks(), hard_tasks(fns)
    t0 = time.time()
    console.print(f"[bold]Auditor experiment: v0 and {best.gid}, with vs without independent Auditor[/bold]")
    with ThreadPoolExecutor(4) as ex:
        jobs = {"v0 std": ex.submit(evaluate, team_a, Genome(), std, 3),
                "v0 hard": ex.submit(evaluate, team_a, Genome(), hard, 3),
                "best std": ex.submit(evaluate, team_a, best, std, 3),
                "best hard": ex.submit(evaluate, team_a, best, hard, 3)}
        audited = {k: f.result() for k, f in jobs.items()}
    for k, s in audited.items():
        save(f"runs/audit_{k.replace(' ', '_')}.json", f"{k} + auditor", best if k.startswith("best") else Genome(), s)

    rows = [("Real accuracy", "accuracy"), ("Tasks fully solved", "solved"),
            ("False 'done' claims", "false_done"), ("Escalations to human", "escalations")]
    md = ["# Auditor experiment (hidden tasks, never seen in training)", ""]
    for mode, title in (("std", "Standard mess"), ("hard", "Hard mode (fair adversarial tricks)")):
        cols = [("v0", saved[f"v0 {mode}"]), ("v0 + Auditor", audited[f"v0 {mode}"]),
                (best.gid, saved[f"best {mode}"]), (f"{best.gid} + Auditor", audited[f"best {mode}"])]
        table = Table(title=title)
        table.add_column("Metric")
        for name, _ in cols:
            table.add_column(name, justify="center")
        for label, key in rows:
            table.add_row(label, *[cell(s, key) for _, s in cols])
        console.print(table)
        md += [f"## {title}", "", "| Metric | " + " | ".join(n for n, _ in cols) + " |", "|---|" + "---|" * len(cols)]
        md += [f"| {label} | " + " | ".join(cell(s, key) for _, s in cols) + " |" for label, key in rows]
        md += [""]

    def tot(group):
        return sum(s["false_done"] for s in group), sum(s["claimed"] for s in group)
    fd0, cl0 = tot(saved.values())
    fd1, cl1 = tot(audited.values())
    headline = (f"False 'done' claims across all hidden tasks: WITHOUT Auditor {fd0}/{cl0} "
                f"-> WITH Auditor {fd1}/{cl1}")
    console.print(f"[bold]{headline}[/bold]")
    md += [f"**{headline}**", "", f"Cost ${llm.spent:.2f} on Crusoe (100% NVIDIA Nemotron)."]
    with open("AUDIT_RESULTS.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    console.print(f"COST ${llm.spent:.2f} | time {time.time() - t0:.0f}s | wrote AUDIT_RESULTS.md")