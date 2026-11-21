# Tendou Launcher `v0.16.1-stable` Release Notes

**Release Date:** 2026-11-19
**Target Platform:** Kubuntu 26.04 (Linux 7.0+, GCC 14, CUDA 13.2 `sm_120a`, AVX-512 VNNI/VBMI)

## Highlights (`v0.16.1-stable` Decode TPS Acceleration — Zero Neural MTP Required)

1. **Fused 4-Row AVX2/AVX-512 `ggml_vec_dot_iq4_xs_q8_K_x4` CPU MoE Kernel (`arch/x86/quants.c`, `ggml-cpu.c`):**
   - Added 4-row fused `IQ4_XS` vector-matrix kernel (`ggml_vec_dot_iq4_xs_q8_K_x4`) with 8-row unrolled decode & speculative verification loops, reducing activation load traffic by 4x and breaking loop-carried accumulator dependencies across 4 rows.
   - Upgraded `IQ3_XXS` single-token decode and speculative verification (`n_slots == 1`) to full multi-cacheline row prefetch and 4-row `_x4` execution.
   - Hoisted `atlas_parse_moe_kind` outside all inner expert/row loops (`atlas_resolve_expert_ptr_fast`).
2. **Synchronized Dynamic-K Turbo Schedule (`ggml_cpu_get_keep_k` & `s_decode_step`):**
   - Enabled post-warmup adaptive reasoning-K (`s_decode_step <= 2 ? 4 : 2` on layers 15–30) and calibrated `Level 2 Turbo` (`--boost --atlas-boost-level 2`) across entry (`0..1`), core reasoning (`15..30`), late semantic/output (`37..39`, `44..47`), and outer MoE layers.
3. **Zero-Copy MMAP & Zero-Split 8GB VRAM Auto-Gating (`atlas-engine.cpp`):**
   - In `"auto"` mode on `<= 12 GiB` GPUs or when `--boost` Dynamic-K is active, avoids injecting thin-coverage `gpu_moe` (`GGML_OP_CUSTOM`) backend splits and synchronous decode-time H2D copies, reducing peak VRAM by `1.3–3.4 GiB` (`~4.1 GiB` on `Qwen3.6-35B`, `~6.75 GiB` on `Qwen3.8-Flash-Next`).
   - In `"auto"` MMAP mode on constrained host RAM, keeps 100% of host RAM in the zero-copy OS `mmap` page cache + `TuttiAsyncPipeline` background prefetcher instead of allocating a duplicate multi-GiB heap arena or stalling decode on synchronous `std::memcpy`.
4. **Hybrid GPU-Attention / CPU-MoE Threadpool Spin Continuity:**
   - Restored default active threadpool polling (`poll = 50`, `KMP_BLOCKTIME = 200`) so worker threads remain hot across ~80 µs GPU FlashAttention transitions instead of entering kernel futex sleep 48 times per token.
5. **Fast Tool-Call Streaming & Default `--boost` With `--mtp off` (`server.py`):**
   - Enabled `boost=True` by default while keeping Neural MTP strictly OFF (`mtp="off"`).
   - Eliminated unrequested `<think>` buffering on tool-calling requests when `enable_thinking` / `thinking` / `reasoning_effort` are not specified.
6. **Preserved All Phase 17 & Phase 18 Innovations:**
   - FP8 E4M3 KV cache (`-ctk f8_e4m3 -ctv f8_e4m3`) with `VEC` FlashAttention (`7.05 µs` at 16K context), FFN chunking (`--atlas-ffn-chunk-size`), `sched_reserve` graph reuse, 3-tensor `io_uring` DMA, `_PriorityEngineGate` Swarm scheduler, and 16-slot Hierarchical `PrefixCache` KV Tree Forking.
