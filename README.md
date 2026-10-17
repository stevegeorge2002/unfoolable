\# Unfoolable 🛡️

\*\*Most AI agents can be fooled into saying "done." Unfoolable is a self-improving agent team that trains against an adversary — and a referee that never lets it fool itself.\*\*



Built from scratch at The Recursive Agents Hackathon (Oct 17, 2026) on \*\*Crusoe Managed Inference\*\*, 100% \*\*NVIDIA Nemotron\*\*. Total cost of every experiment: \*\*under $4\*\*.



\## Headline results (tasks never seen during training)

| | v0 (start) | v6 (self-evolved) |

|---|---|---|

| Accuracy on unseen data kinds + adversarial tricks | 45.8% | \*\*62.3% (+16.5 pts)\*\* |

| Accuracy on unseen data kinds (standard mess) | 94.0% | 90.5% (within noise) |

| Coach's predicted gain vs. real gain | | \*\*overestimated 7 of 7 times, avg +48 pts\*\* |

| Self-edits proposed / rejected by the gate | | 7 / \*\*4 rejected\*\* |



Full tables: `RESULTS.md` and `RESULTS\_backup1.md`. Raw data: `runs/` and `runs\_backup1/`.



\## The problem

Agents fill vague goals with their own assumptions and confidently report success. When agents improve themselves, they also claim every change helped. Our v0 team said "done" on hidden tasks twice — \*\*both times it was wrong.\*\*



\## How it works

\*\*Cleaner team\*\* (turns a messy data file + one-line goal into clean data):

\- \*\*Spec writer\*\* (Nemotron Super): writes its own \*definition of done\* as executable checks

\- \*\*Builder\*\* (Nemotron Super): writes the cleaning code

\- \*\*Sandbox\*\*: runs code in a separate process; checks run in another process that never imports the builder's code (nonce-protected), so the builder can't fake a pass

\- \*\*Watchdog\*\* (plain code, no AI): detects loops (same code twice), stalls, drift; decides CONTINUE / REPLAN / STOP / ESCALATE to a human

\- \*\*Checkpoint\*\*: keeps the best attempt, not the last one



\*\*Trickster\*\* (Nemotron Super): invents new, realistic ways to corrupt data (fullwidth Unicode, zero-width spaces, quoted ids...), aimed at the Cleaner's current weak spots. Every trick is code, auto-checked for safety and shape.



\*\*Coach\*\* (Nemotron Super): reads the Cleaner's hidden mistakes and proposes ONE general lesson, plus a prediction of how much it will help.



\*\*Gate\*\*: old vs. new version on the same fresh validation jobs. A lesson is kept only if accuracy rises ≥2 points AND false "done" claims don't increase. We log the \*\*mirage gap\*\* = predicted gain − real gain.



\*\*Referee\*\*: ground truth is known exactly (we corrupt clean data we generated), so scoring never needs an LLM judge.



\## What we learned

1\. \*\*Agents can't grade their own homework.\*\* Even the evolved team's self-written checks passed wrong output. Done must be verified from outside the agent.

2\. \*\*Self-assessment is a mirage.\*\* The Coach predicted +28% to +65% for every lesson; real effects ranged from −28 to +12 points.

3\. \*\*Adversaries cheat too.\*\* The Trickster invented 2 tricks that destroyed information (e.g. dropping first characters). We excluded them from the final exam after review. Next step: require every trick to ship its own inverse, verified automatically.

4\. \*\*Self-evolved lessons generalize.\*\* Lessons learned on customers/invoices/orders improved robustness on products/payments/shipments, which it had never seen.



\## Honest limitations

\- Small samples (9 hidden tasks per condition, 1 seed); differences of a few points are noise.

\- The 16.5-pt gain is on adversarial "hard mode"; on standard mess v6 ≈ v0.

\- Run 1 used a harness with two bugs (JSON parsing of regex checks; discarding the best attempt). We fixed them, re-measured v0, and report the fixed numbers. Both runs are kept in the repo.



\## Run it

&#x20;   python -m venv .venv \&\& .venv\\Scripts\\activate

&#x20;   pip install openai rich pandas python-dotenv

&#x20;   # .env -> CRUSOE\_API\_KEY=...

&#x20;   python evaluate.py              # v0 baseline on hidden tasks

&#x20;   python evolve.py 3 \[--resume]   # Trickster vs Cleaner, Coach + Gate

&#x20;   python final.py                 # before/after table -> RESULTS.md

&#x20;   python replay.py runs/live\_log.jsonl 40 TRICK,COACH,KEEP,REJECT,SCORE   # replay highlights



\## Models (Crusoe Managed Inference)

`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B` for all agents (gpt-oss-120b only as automatic fallback when busy).



\## Files

`data.py` clean data generator · `tricks.py` base tricks + scorer · `team.py` Cleaner team, sandbox, watchdog · `trickster.py` · `evolve.py` · `evaluate.py` · `final.py` · `replay.py`

