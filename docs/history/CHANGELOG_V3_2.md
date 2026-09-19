# Atlas v3.2 — physical layout stage

## Added
- `gguf_utils.py`: streaming GGUF v3 header/metadata/tensor-index parser.
- `build_physical_map.py`: exact GGUF tensor offsets + expert-slice map, joined with trace logical map.
- `atlas_physical_simulator.py`: variable-size physical I/O simulator with NVMe read coalescing, RAM staging, PCIe transfer, read-ahead waste, I/O op counts, exposed latency and effective TPS.
- `test_physical_map.py`: synthetic GGUF regression test.

## Safety / non-claims
- No tensor offsets are guessed.
- Exact per-expert slicing is reported only when the tensor layout permits a safe contiguous split.
- Unsupported/inexact tensor layouts are explicitly marked; they are not silently treated as exact.
- Physical simulator is not a real CUDA runtime benchmark.

## Target hardware model
- Physical VRAM: 8 GB
- VRAM reserve: 1 GB
- Atlas VRAM budget: 7 GB
- Physical RAM: 16 GB
- RAM reserve: 2 GB
- Atlas RAM budget: 14 GB
- NVMe: 7 GB/s
- PCIe: 12 GB/s
- Host RAM copy: 40 GB/s
