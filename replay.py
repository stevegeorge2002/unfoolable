import json, sys, time
from team import show

path = sys.argv[1] if len(sys.argv) > 1 else "runs/live_log.jsonl"
speed = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0     # 20 = twenty times faster than real life
only = sys.argv[3].split(",") if len(sys.argv) > 3 else None   # e.g. TRICK,COACH,KEEP,REJECT,SCORE

with open(path, encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]
if only:
    rows = [r for r in rows if r["kind"] in only or any(r["tid"].startswith(o) for o in only)]
print(f"REPLAY of a real recorded run ({len(rows)} events, {speed:g}x speed)\n")
for a, b in zip(rows, rows[1:] + rows[-1:]):
    show(a["tid"], a["kind"], a["msg"])
    time.sleep(min(max(b["t"] - a["t"], 0) / speed, 0.8))