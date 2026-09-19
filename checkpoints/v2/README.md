# Tendou Launcher v2 checkpoint

This checkpoint preserves the bounded-memory runtime immediately before the
unlimited idle RAM/VRAM throughput experiments.

- llama.cpp commit: `a58dd6c6e`
- llama.cpp tag: `v2`
- branch at capture: `experimental-gpu-first`
- version: `2.0.0-v2-runtime`
- binary directory: `checkpoints/v2/bin/` (local and gitignored)

## Binary SHA-256

| File | Bytes | SHA-256 |
|---|---:|---|
| `ggml-base.dll` | 690688 | `1512AC6667F85F416DAE0E5FB7BAB806512BE5D3FDAC1869BBA9E66D50334716` |
| `ggml-cpu.dll` | 1101824 | `20EC0EC18D19D4C1E7955F614C5BE9B7BA5F8753ECE6190BF34B502392866524` |
| `ggml-cuda.dll` | 34533376 | `C3F5FE5AFDAA6731BB32679CE18DA5BD90CF8A52F50507A6A177CF51CF71E5B0` |
| `ggml.dll` | 68608 | `773FDA2C92AED6A1798ED5F27BBC6C4E0827479E1FAF70CD4A4893CCDC895CD7` |
| `llama-atlas-engine.exe` | 395776 | `7C0276270E3091C2E55FB78C18C822CE51382F91F8C9A3ED73553073F20F3444` |
| `llama-common.dll` | 8279040 | `3B7D367F734812D192E9C6D6CD9C63418B6A08C5352A45D68F72CA300AB20D7C` |
| `LLAMA-SERVER-IMPL.DLL` | 13113344 | `7648C6747A1D4481B94EE180E943716910CAB3CF402D5D52F106F78935AD7971` |
| `llama.dll` | 2592256 | `4860A906C690AC1EAD9E74962512E331828FCFDBC4F8DA6972C50DFB5FD2748A` |

## Rollback

1. Switch the nested `llama.cpp` checkout to tag `v2` or commit `a58dd6c6e`.
2. Copy the files from `checkpoints/v2/bin/` back to
   `llama.cpp/build/bin/Release/`.
3. Verify the SHA-256 values above before running the restored binary.
