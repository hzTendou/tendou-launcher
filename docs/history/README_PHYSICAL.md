# Atlas physical GGUF stage

This stage is intentionally separate from the trace-only hypothesis simulator.
It reads the **actual GGUF tensor index** and never invents tensor offsets.

## 1. Build the physical map

```powershell
python build_physical_map.py `
  --gguf models\YOUR_TARGET_MODEL.gguf `
  --logical-map atlas_map_35b.json `
  --out atlas_physical_map.json
```

The map records:

- GGUF data start/alignment
- architecture / layer / expert metadata when present
- every tensor name, shape, type and physical offset
- exact expert slices when an expert-packed tensor has the expert dimension on the outermost axis and its byte size is known
- explicit `inexact` entries when an exact expert slice cannot safely be derived

**No guessed offsets are emitted.**

## 2. Run physical I/O simulation

Target hardware is fixed to the Atlas target budget:

- 8 GB physical VRAM, 1 GB reserve => 7 GB Atlas VRAM
- 16 GB physical DDR5, 2 GB reserve => 14 GB Atlas RAM
- 7 GB/s NVMe
- 12 GB/s PCIe
- 40 GB/s effective host RAM copy bandwidth

```powershell
python atlas_physical_simulator.py `
  --trace trace_35b_a3b_q2k.jsonl `
  --physical-map atlas_physical_map.json `
  --vram-gb 8 `
  --vram-reserve-gb 1 `
  --ram-gb 16 `
  --ram-reserve-gb 2 `
  --nvme-gbps 7 `
  --pcie-gbps 12 `
  --ram-bandwidth-gbps 40 `
  --target-tps 20
```

The simulator compares `none`, `causal` and `oracle` under the physical tensor layout.
It reports NVMe read bytes, I/O operation count, read-ahead waste, PCIe traffic,
exposed latency and effective tokens/s.

## Important limitation

This is still a **physical-I/O simulator**, not the production runtime. It does not
execute CUDA kernels or claim that a specific TPS will be achieved. The purpose is
to answer a narrower question: does the actual GGUF storage layout make Atlas
prefetch physically cheaper or more expensive than the logical trace-only model?
