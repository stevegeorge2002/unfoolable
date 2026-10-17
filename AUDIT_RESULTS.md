# Auditor experiment (hidden tasks, never seen in training)

## Standard mess

| Metric | v0 | v0 + Auditor | v6 | v6 + Auditor |
|---|---|---|---|---|
| Real accuracy | 94.0% | 96.8% | 90.5% | 84.3% |
| Tasks fully solved | 33.3% | 44.4% | 11.1% | 22.2% |
| False 'done' claims | 2/2 (100%) | 0/0 (0%) | 1/1 (100%) | 1/2 (50%) |
| Escalations to human | 1 | 0 | 0 | 1 |

## Hard mode (fair adversarial tricks)

| Metric | v0 | v0 + Auditor | v6 | v6 + Auditor |
|---|---|---|---|---|
| Real accuracy | 45.8% | 51.9% | 62.3% | 32.0% |
| Tasks fully solved | 11.1% | 11.1% | 0.0% | 0.0% |
| False 'done' claims | 2/3 (67%) | 0/0 (0%) | 3/3 (100%) | 0/0 (0%) |
| Escalations to human | 2 | 0 | 0 | 3 |

**False 'done' claims across all hidden tasks: WITHOUT Auditor 8/9 -> WITH Auditor 1/2**

Cost $0.84 on Crusoe (100% NVIDIA Nemotron).