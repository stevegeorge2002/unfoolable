# Results

| Metric | Hidden tasks: v0 (before) | Hidden tasks: v3 (after) | Hard mode: v0 | Hard mode: v3 |
|---|---|---|---|---|
| Real accuracy | 83.2% | 65.8% | 35.1% | 33.1% |
| Tasks fully solved | 22.2% | 44.4% | 0.0% | 0.0% |
| False 'done' claims | 2/2 (100%) | 1/2 (50%) | 1/1 (100%) | 1/1 (100%) |
| Escalations to human | 1 | 2 | 2 | 2 |

Held-out data kinds (never used in training): products, payments, shipments

## Lessons v3 taught itself
- Add a strict completeness check that requires every field to be non-empty and match the expected format for all rows; any row with missing or placeholder values (e.g., 'MISSING') must fail the spec.

## Tricks the Trickster invented
- fullwidth substitution - replace ASCII alphanumerics with lookalike fullwidth Unicode
- invisible zero-width space insertion - inserts undetectable Unicode ZWSP into strings

Final test cost $0.49 on Crusoe (100% NVIDIA Nemotron).