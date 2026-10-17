import os, sys, json, secrets, subprocess, tempfile, hashlib
from dataclasses import dataclass, field
import pandas as pd
from rich.console import Console
from rich.text import Text
from llm import LLM, SUPER, extract_json, extract_code
from data import schema_text
from tricks import make_task, score

console = Console()
DONE_BAR = 0.98   # real accuracy needed for a "done" claim to count as true
STYLE = {"SPEC": "cyan", "CONTINUE": "white", "REPLAN": "yellow", "STOP": "green", "ESCALATE": "bold red",
         "TRUTH": "bold", "KEEP": "bold green", "REJECT": "red", "TRICK": "magenta", "SCORE": "bold blue"}


def show(tid, kind, msg):
    style = STYLE.get(kind, "white")
    if "FALSE DONE" in msg:
        style = "bold white on red"
    console.print(Text(f"{tid:<16} {kind:<9} {msg}", style=style))


# ---------- sandbox: runs untrusted cleaning code in a separate process ----------
RUN_CLEAN = r'''
import pandas as pd, traceback
raw = pd.read_csv("messy.csv", dtype=str, keep_default_na=False)
try:
    from solution import clean
    out = clean(raw.copy())
    if not isinstance(out, pd.DataFrame):
        raise TypeError("clean() must return a pandas DataFrame")
    out.to_pickle("out.pkl")
    print("CLEAN_OK")
except Exception:
    print("CLEAN_ERROR\n" + traceback.format_exc()[-1500:])
'''

RUN_CHECKS = r'''
import pandas as pd, re, json, os
raw = pd.read_csv("messy.csv", dtype=str, keep_default_na=False)
out = pd.read_pickle("out.pkl")
res = []
for c in json.load(open("checks.json")):
    try:
        exec(c, {"out": out, "raw": raw, "pd": pd, "re": re})
        res.append([c, True, ""])
    except Exception as e:
        res.append([c, False, (type(e).__name__ + ": " + str(e))[:150]])
print(os.environ["NONCE"] + json.dumps(res))
'''


def run_in_sandbox(code, messy, checks, timeout=60):
    """Step 1 runs the builder's code. Step 2 (separate process, never imports builder code) runs the checks."""
    with tempfile.TemporaryDirectory() as d:
        messy.to_csv(os.path.join(d, "messy.csv"), index=False)
        with open(os.path.join(d, "solution.py"), "w", encoding="utf-8") as f:
            f.write(code)
        with open(os.path.join(d, "checks.json"), "w", encoding="utf-8") as f:
            json.dump(checks, f)
        env = {k: v for k, v in os.environ.items() if k != "CRUSOE_API_KEY"}

        def py(script):
            return subprocess.run([sys.executable, "-c", script], cwd=d, capture_output=True,
                                  text=True, timeout=timeout, env=env)
        try:
            p = py(RUN_CLEAN)
        except subprocess.TimeoutExpired:
            return None, [], "TIMEOUT: clean() took too long"
        if "CLEAN_OK" not in p.stdout:
            return None, [], (p.stdout + p.stderr)[-1500:]
        out = pd.read_pickle(os.path.join(d, "out.pkl"))
        nonce = "__N" + secrets.token_hex(6) + "__"
        env["NONCE"] = nonce
        try:
            q = py(RUN_CHECKS)
        except subprocess.TimeoutExpired:
            return out, [], "TIMEOUT while running checks"
        for line in q.stdout.splitlines():
            if line.startswith(nonce):
                return out, [tuple(x) for x in json.loads(line[len(nonce):])], ""
        return out, [], "CHECKS CRASHED: " + (q.stderr or "")[-800:]


# ---------- the team's evolvable settings ----------
@dataclass
class Genome:
    gid: str = "v0"
    parent: str = None
    spec_lessons: list = field(default_factory=list)
    build_lessons: list = field(default_factory=list)
    max_iters: int = 4
    note: str = "starting version"


@dataclass
class Result:
    task_id: str
    claimed_done: bool = False
    check_pass: float = 0.0
    acc: float = 0.0
    iters: int = 0
    decision: str = "STOP"
    verdict: str = ""
    events: list = field(default_factory=list)
    errors: list = field(default_factory=list)


class Watchdog:
    """Plain code, no AI: catches loops, stalls and drift."""
    def __init__(self, max_iters):
        self.seen, self.scores, self.replans, self.max_iters = set(), [], 0, max_iters

    def decide(self, code, s, i, crashed):
        h = hashlib.sha1(code.encode()).hexdigest()
        looped = h in self.seen
        self.seen.add(h)
        self.scores.append(s)
        if "def clean" not in code:
            return "REPLAN", "drift: no clean() function written"
        if s == 1.0:
            return "STOP", "all done-checks pass"
        stalled = len(self.scores) >= 3 and max(self.scores[-2:]) <= self.scores[-3]
        if looped or stalled:
            if self.replans == 0:
                self.replans += 1
                return "REPLAN", "loop: same code again" if looped else "stall: no progress in 2 tries"
            if s < 0.5:
                return "ESCALATE", "still failing after re-plan; done-checks may be wrong -> needs a human"
            return "STOP", "stuck after re-plan; reporting NOT DONE"
        if i + 1 >= self.max_iters:
            return "STOP", "out of tries"
        return "CONTINUE", f"{s:.0%} of done-checks pass" + (" (code crashed)" if crashed else "")


SPEC_SYS = """You write the DEFINITION OF DONE for a data-cleaning job.
You see the target schema and the messy input. Write 10-16 Python assert statements that the cleaned
pandas DataFrame `out` must satisfy. Available names: out, raw (messy input, all strings), pd, re.
Cover: exact column names and order, unique ids, row count, formats (use regex), allowed category values,
numeric types and rounding, no stray spaces or symbols. Each assert must fit on ONE line.
Return JSON only: {"assumptions": ["..."], "checks": ["assert ...", "..."]}"""

BUILD_SYS = """You are a meticulous data engineer. Write a Python function clean(raw) that takes the messy pandas
DataFrame `raw` (every value is a string; headers may be messy) and returns a NEW DataFrame that matches the
target schema exactly: same column names in the same order, correct types and formats.
Use only pandas, re and datetime. Return exactly ONE ```python``` block with imports and def clean(raw):"""


def _lessons(lst):
    return "\n\nLESSONS FROM PAST JOBS:\n" + "\n".join(f"- {x}" for x in lst) if lst else ""


class Team:
    def __init__(self, llm, emit=show):
        self.llm, self.emit = llm, emit

    def _job(self, task):
        cols = [c[0] for c in task["schema"]]
        return (f"TARGET SCHEMA:\n{schema_text(task['schema'])}\nREQUIRED COLUMN ORDER: {cols}\n\n"
                f"MESSY INPUT (CSV):\n{task['messy'].to_csv(index=False)}")

    def write_checks(self, g, task):
        reply = self.llm.chat(SUPER, SPEC_SYS + _lessons(g.spec_lessons), self._job(task), max_tokens=2500)
        spec = extract_json(reply) or {}
        checks = [c.strip() for c in spec.get("checks", []) if isinstance(c, str) and c.strip().startswith("assert")]
        return spec.get("assumptions", []), checks

    def run(self, g, task):
        r = Result(task["id"])

        def ev(kind, msg):
            r.events.append((kind, msg))
            self.emit(task["id"], kind, msg)

        assumptions, checks = self.write_checks(g, task)
        ev("SPEC", f"{len(checks)} done-checks | assumes: {'; '.join(map(str, assumptions))[:140]}")
        if not checks:
            r.decision = "ESCALATE"
            ev("ESCALATE", "could not write a definition of done")
            return r
        wd, code, feedback, out = Watchdog(g.max_iters), "", "", None
        temp = 0.6
        for i in range(g.max_iters):
            r.iters = i + 1
            prev = f"\n\nYOUR PREVIOUS CODE:\n```python\n{code}\n```\nPROBLEMS:\n{feedback}" if code else ""
            reply = self.llm.chat(SUPER, BUILD_SYS + _lessons(g.build_lessons),
                                  self._job(task) + "\n\nDEFINITION OF DONE (your output must pass):\n"
                                  + "\n".join(checks) + prev, max_tokens=3500, temperature=temp)
            code = extract_code(reply)
            out, res, err = run_in_sandbox(code, task["messy"], checks)
            r.check_pass = sum(ok for _, ok, _ in res) / len(checks) if res else 0.0
            fails = [f"{c}  -> {e}" for c, ok, e in res if not ok]
            feedback = (err + "\n" if err else "") + "\n".join(fails)[:2500]
            if out is not None:
                feedback += "\nFIRST ROWS OF YOUR OUTPUT:\n" + out.head(5).to_string()[:1200]
            decision, why = wd.decide(code, r.check_pass, i, bool(err))
            ev(decision, f"try {i + 1}: {why}")
            if decision == "REPLAN":
                feedback += "\nWATCHDOG: your approach is stuck. Rethink from scratch with a different strategy."
                temp = 1.0
            elif decision in ("STOP", "ESCALATE"):
                r.decision = decision
                break
        r.claimed_done = out is not None and r.check_pass == 1.0
        if out is not None:
            r.acc, r.errors = score(out, task["truth"], task["schema"])
        else:
            r.acc, r.errors = 0.0, ["no output produced"]
        r.verdict = ("TRUE DONE" if r.claimed_done and r.acc >= DONE_BAR
                     else "FALSE DONE" if r.claimed_done else "HONEST NOT DONE")
        ev("TRUTH", f"claimed {'DONE' if r.claimed_done else 'NOT DONE'} | real accuracy {r.acc:.1%} -> {r.verdict}")
        return r


if __name__ == "__main__":
    llm = LLM()
    team = Team(llm)
    result = team.run(Genome(), make_task("customers", seed=1))
    if result.errors:
        console.print("Hidden mistakes the team missed:", *result.errors[:5], sep="\n  ")
    console.print(f"COST: ${llm.spent:.3f} over {llm.calls} calls")