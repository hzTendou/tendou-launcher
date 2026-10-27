# Atlas Engine V7

## Changes
- Fixed `effective_tps` so prompt-token work between sessions is excluded from decode TPS.
- Added `decode_active_ms` to expose pure decode elapsed time (`sum(exposed_io + compute)` per decode token).
- Fixed prediction cold/hot accounting to snapshot cache state before satisfying current demand.
- Changed `persistent_fallback` behavior to blend local routing evidence (75%) with persistent cross-session evidence (25%) when enabled, instead of using persistent evidence only when local evidence is absent.

## Verified
- `7 tests passed`
- Baseline effective TPS: ~17.926 TPS
- Atlas, 16 candidates + persistent blend: ~18.104 TPS
- Oracle effective TPS: ~19.999 TPS

## Interpretation
The simulator can reach ~20 TPS under oracle routing once prompt accounting is corrected. Atlas remains ~1.9 TPS behind oracle. The remaining gap is primarily prediction coverage / timely useful prefetch rather than the TPS metric itself.
