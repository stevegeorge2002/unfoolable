import json, os, re, sys, unicodedata
from rich.console import Console
from rich.table import Table
import team as T
from team import Team, Genome
from llm import LLM
from tricks import make_task, _same

console = Console()
captured = {}
_orig_score = T.score


def _capture(out, truth, schema):
    captured["out"] = out
    return _orig_score(out, truth, schema)


T.score = _capture   # lets us keep the agents' cleaned file to show it


def digits(s):
    return re.sub(r"\D", "", unicodedata.normalize("NFKC", str(s)))


def messiness(row):
    return sum(1 for v in row for ch in str(v) if ord(ch) > 127 or ch in "#$(),;:'\"") + \
           sum(1 for v in row if str(v) != str(v).strip())


if __name__ == "__main__":
    domain = sys.argv[1] if len(sys.argv) > 1 else "payments"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    task = make_task(domain, seed)
    g = Genome()
    if os.path.exists("runs/best.json"):
        with open("runs/best.json", encoding="utf-8") as f:
            g = Genome(**json.load(f))
    os.makedirs("runs", exist_ok=True)
    task["messy"].to_csv("runs/show_messy.csv", index=False)
    console.print(f"[bold]MESSY FILE: {len(task['messy'])} rows of {domain} data, never seen in training. "
                  f"Agents: {g.gid} + Auditor[/bold]")
    console.print(task["messy"].head(8).to_string(index=False))
    console.print()

    r = Team(LLM(), auditor=True).run(g, task)
    out = captured.get("out")
    if out is None:
        console.print("[red]The agents produced no output this time.[/red]")
        sys.exit()
    out.to_csv("runs/show_cleaned.csv", index=False)

    schema = task["schema"]
    cols = [c[0] for c in schema]
    types = {c[0]: c[1] for c in schema}
    truth = task["truth"].set_index(cols[0])
    cleaned = {}
    for _, row in out.iterrows():
        try:
            cleaned[int(float(row[cols[0]]))] = row
        except (ValueError, TypeError, KeyError):
            pass

    messy = task["messy"]
    order = sorted(range(len(messy)), key=lambda i: -messiness(messy.iloc[i].tolist()))
    shown, seen = 0, set()
    for i in order:
        mrow = messy.iloc[i].tolist()
        key = digits(mrow[0])
        if not key or int(key) in seen or int(key) not in truth.index:
            continue
        k = int(key)
        seen.add(k)
        t = Table(title=f"Record {k}: BEFORE -> AFTER")
        t.add_column("Field", style="bold")
        t.add_column("Messy input", style="red")
        t.add_column("Cleaned by agents", style="green")
        t.add_column("Correct?")
        crow = cleaned.get(k)
        for j, c in enumerate(cols):
            got = crow[c] if crow is not None and c in crow.index else "(row missing)"
            exp = k if j == 0 else truth.loc[k, c]
            ok = got != "(row missing)" and _same(got, exp, types[c])
            t.add_row(c, repr(mrow[j]), str(got), "OK" if ok else f"WRONG (should be {exp})")
        console.print(t)
        shown += 1
        if shown >= 3:
            break

    console.print(f"\n[bold]VERDICT: {r.verdict} | real accuracy on the whole file: {r.acc:.1%}[/bold]")
    console.print("Saved runs\\show_messy.csv (before) and runs\\show_cleaned.csv (after). Open them in Excel.")