# Atlas Predictor v5

## Why v5

The v4 sweep exposed an important weakness: several hyperparameters produced
almost identical prediction counts, and prompt/decode routing was being treated
as one statistical stream. That made the benchmark less useful for deployment.

## Changes

- Prompt and decode histories are isolated.
- Decode benchmark scores only same-phase next-token predictions.
- Transition scores use per-source conditional probabilities instead of raw
  count/max normalization.
- Candidate generation is sparse and bounded by the strongest transitions from
  active experts.
- Local request evidence is weighted above persistent cross-session history.
- Confidence controls abstention and only a small candidate-budget expansion.
- `history_window` remains API-compatible, but v5 deliberately avoids sparse
  exact set-context order-2 models because the 35B trace has very few repeated
  exact set contexts.
- Physical and logical simulators now pass routing phase into the predictor.
- Benchmark output reports decode precision/recall/F1/F2 separately from prompt.
- New tests verify the prompt/decode boundary and leakage behavior.

## Interpretation

This remains a trace-driven predictor. It does not change the router and does
not claim real GPU performance. The next engineering target is to improve the
predictor's *physical utility* (correct bytes prefetched before demand) rather
than maximizing raw recall with aggressive over-prefetching.


## v5.1 — physical prefetch integration

- Added a persistent prompt-last -> first-decode boundary predictor.
- Physical simulation now retains prompt records so the predictor can warm the
  first decode token without using future expert IDs.
- Added cache-aware prediction filtering: candidates already resident in VRAM
  are removed before the candidate budget is consumed, so the predictor spends
  its limited budget on potentially useful transfers.
- Physical simulator now reports predicted/correct/false prefetch counts and
  predictor precision.
- Decode effective TPS is measured over the decode interval; prompt tokens are
  reported separately as warm-up tokens.
