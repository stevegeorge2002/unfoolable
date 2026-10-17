import json, time, threading
from concurrent.futures import ThreadPoolExecutor
from llm import LLM
from team import Team, Genome, show
from tricks import make_task

LOG = open("runs/demo_log.jsonl", "a", encoding="utf-8")
LOCK = threading.Lock()


def emit(tid, kind, msg):
    show(tid, kind, msg)
    with LOCK:
        LOG.write(json.dumps({"t": time.time(), "tid": tid, "kind": kind, "msg": msg}) + "\n")
        LOG.flush()


if __name__ == "__main__":
    llm = LLM()
    team = Team(llm, emit=emit, auditor=True)
    tasks = [make_task("products", 102), make_task("payments", 100), make_task("shipments", 100)]
    with ThreadPoolExecutor(3) as ex:
        results = list(ex.map(lambda t: team.run(Genome(), t), tasks))
    for r in results:
        show(r.task_id, "SCORE", f"final verdict {r.verdict} | real accuracy {r.acc:.1%}")
    print(f"COST ${llm.spent:.3f}")