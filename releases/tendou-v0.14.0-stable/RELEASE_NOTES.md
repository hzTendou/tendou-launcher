# Tendou Launcher v0.14.0-stable Release Notes

**Sürüm:** v0.14.0-stable  
**Yayınlanma Tarihi:** 2026-10-23  
**Mühür:** 100% Deterministic Bit-Exact Parity (0-ULP), 27/27 Native Gates PASS, 109 Pytest Tests PASS  

---

## 1. Genel Bakış ve Temel Felsefe

Tendou Launcher `v0.14.0-stable`, Phase 14 kapsamında geliştirilen **Dinamik Kademeli 3-Tier MoE Çalışma Kümesi (HWSM - Hierarchical Working Set Manager)** ve sıfır sistem çağrılı (Zero-Syscall) bellek slab mimarisini resmi, dondurulmuş ve bağımsız bir sürüm paketi olarak mühürler.

Tendou Launcher bir llama.cpp çatalı (fork) DEĞİLDİR; kendine ait dinamik MoE bellek yöneticisi (`ExpertMemoryManager`), 3-Tier bellek yerleşimi (VRAM + RAM + DirectStorage), asenkron orkestrasyonu, sıfır gecikmeli HTTP SSE boru hattı ve **Universal Hardware & Topology Auto-Tuning Engine**'i olan **tamamen bağımsız bir MoE çıkarım platformu ve launcher'ıdır**.

---

## 2. Dahil Edilen Temel Yenilikler ve Mimari Atılımlar (Phase 14)

### P8: True Zero-Syscall Pre-Allocated Slab Arenas (`ExpertSlabPool`)
- **Sıfır Çekirdek Syscall Ceza:** Çıkarım anında dinamik `VirtualAlloc(MEM_COMMIT)` / `VirtualFree(MEM_DECOMMIT)` çağrıları tamamen elendi.
- Slab havuzları başlangıçta tek seferde `VirtualAlloc(MEM_RESERVE | MEM_COMMIT)` ile tahsis edilip `VirtualLock` ile kilitlenir.
- Token decode döngüsünde slot edinme (`acquire_slot`) ve serbest bırakma (`release_slot`) işlemleri 0 çekirdek sistem çağrısı ile $O(1)$ stack pop/push indeks aritmetiğine indirgendi.

### P9: Dinamik Kademeli 3-Tier Terfi ve Tenzil (Promotion & Cascading Demotion)
- **Terfi ($T_2 \to T_0$ / $T_1 \to T_0$):** Router aktivasyonu veya TLEP lookahead tahminiyle ihtiyaç duyulan uzman doğrudan GPU VRAM (Tier-0) slab havuzuna taşınır.
- **Seviye-1 Tenzil ($T_0 \to T_1$):** VRAM slab havuzu dolduğunda LRU uzmanlar diske atılmaz; doğrudan Tier-1 Host RAM VirtualLock havuzuna düşürülerek AVX-512 VNNI işlem koruması altında tutulur.
- **Seviye-2 Tenzil ($T_1 \to T_2$):** RAM bütçesi dolduğunda en alttaki soğuk uzmanlar Tier-2 NVMe DirectStorage depolama alanına tahliye edilir.

### P10: Anti-Duplication & GPU Kernel Senkronizasyonu
- `split_moe_cache` ile `ExpertMemoryManager` tek birleşik bellek yöneticisi altında birleştirildi.
- `atlas_slot_update_cb` ve `atlas_set_expert_gpu_slot` üzerinden dinamik VRAM terfi ve tenzilleri anında CUDA GPU MoE yürütme çekirdeği lookup tablosuna (`g_atlas_gpu_slot_lut`) yansıtıldı.
- Mükerrer LRU/Clock döngüleri tekil ve DRY `evict_vram_one_unlocked` / `evict_ram_one_unlocked` yardımcılarına toplandı.

---

## 3. İkili Dosya Bütünlüğü ve Manifestosu

```
releases/tendou-v0.14.0-stable/bin/
  tendou-engine.exe           516,096 B  (MSVC 19.51, AVX-512 + CUDA 13, 27 Gates)
  llama-atlas-engine.exe      516,096 B  (Uyumluluk sembolik ikilisi)
  ggml-cpu.dll              1,125,376 B  (AVX-512 VNNI/VBMI optimized)
  ggml-cuda.dll            34,533,376 B  (NVIDIA CUDA 13 cuBLAS)
  llama.dll                 2,593,792 B  (Core tensor runtime)
  llama-common.dll          8,279,040 B  (Common support layer)
  ggml-base.dll               694,784 B  (GGML backend abstractions)
  ggml.dll                     68,608 B  (Base GGML structures)
  mtmd.dll                  2,457,088 B  (Multi-modal support)
  ggml-cpu.lib                 17,974 B  (Link import library)
```

## 4. Doğrulama Durumu

- **Native Gate Testleri:** 27 / 27 PASS (`atlas-native-tests.exe`)
- **Partition Testleri:** PASS (`atlas-partition-tests.exe`)
- **Pytest Entegrasyon Paketi:** 109 passed / 4 skipped (%100 PASS)
- **Top-1 Token Paritesi:** %100.0 (Bit-exact referans)
- **Logit KL Divergence:** $\text{KL} = 0.000000$ (Sıfır matematiksel kayıp)
- **Donanım Ölçümü:** Qwen3.8 Flash Next IQ3_XXS üzerinde 25.86 tok/s (Refactor), 20.75 tok/s (Greenfield), 24.09 tok/s (Tool Stream, 59.33 ms ilk argüman gecikmesi).
