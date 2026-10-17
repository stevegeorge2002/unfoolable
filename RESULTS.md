# Results

| Metric | Hidden tasks: v0 (before) | Hidden tasks: v6 (after) | Hard mode: v0 | Hard mode: v6 |
|---|---|---|---|---|
| Real accuracy | 94.0% | 90.5% | 45.8% | 62.3% |
| Tasks fully solved | 33.3% | 11.1% | 11.1% | 0.0% |
| False 'done' claims | 2/2 (100%) | 1/1 (100%) | 2/3 (67%) | 3/3 (100%) |
| Escalations to human | 1 | 0 | 2 | 0 |

Held-out data kinds (never used in training): products, payments, shipments

## Lessons v6 taught itself
- Add a strict completeness check that requires every field to be non-empty and match the expected format for all rows; any row with missing or placeholder values (e.g., 'MISSING') must fail the spec.
- When cleaning numeric or string fields, scan for accidental digit duplication (e.g., '2092096.3' → '2096.3') by detecting repeated substrings and applying a rule to keep only the first occurrence if the duplicated part matches a plausible value pattern.
- When parsing numeric fields from strings, first remove any non-digit characters except a single decimal point or leading minus sign, then validate the resulting value against plausible ranges (e.g., monthly fees < 100k, seats < 1000) to catch digit duplication or prefix errors like '22794.63' → '2794.63'.

## Fair tricks used in hard mode
- fullwidth substitution - replace ASCII alphanumerics with lookalike fullwidth Unicode
- invisible zero-width space insertion - inserts undetectable Unicode ZWSP into strings
- id as quoted string with trailing space - ids become quoted values with extra space after 

## Tricks excluded as unfair (the Trickster tried to cheat)
- duplicate prefix addition - prepends first few characters of the value to itself
- missing first character - drops the initial character of the string value

Final test cost $0.62 on Crusoe (100% NVIDIA Nemotron).