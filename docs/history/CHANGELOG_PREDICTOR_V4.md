# Atlas Predictor v4

## What changed

- Added `atlas_predictor.py` with adaptive order-1/order-2 routing prediction.
- Added compact token-set order-2 context to avoid cubic expert-triple state.
- Added persistent cross-session learning with post-session commit to prevent
  target-session future leakage.
- Added confidence-aware abstention and bounded prefetch budget.
- Added `benchmark_predictor.py` for cheap hyperparameter sweeps before the
  physical I/O simulator.
- Added `atlas_predictor` to the logical and physical simulators.
- Added predictor configuration to the main simulator CLI.
- Added predictor unit tests.

## Important interpretation

This is still a trace-driven predictor benchmark. It does not claim real GPU
performance, and it does not replace the model router. The real router remains
the source of truth; Atlas only attempts to make the required expert data ready
before the router asks for it.
