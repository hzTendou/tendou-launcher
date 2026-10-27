# Tendou Launcher — Release Notes: Phase 9 (v0.9.0-stable)

**Sürüm:** v0.9.0 (TLEP Phase 9 Stable Baseline)  
**Tarih:** 2026-10-19  
**Durum:** LATEST STABLE RELEASE (Mühürlendi)  
**Temel Model:** `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS` (48 Katman, 512 Uzman/Katman, Dynamic-K)  
**Hedef Donanım:** AMD Ryzen 7 260 (8C/16T, DDR5-5600), NVIDIA GeForce RTX 5060 Laptop (8,151 MiB GDDR6), 16 GB DDR5 RAM  

---

## 1. Genel Bakış ve Mimari Yenilikler

Tendou Launcher, bir llama.cpp çatalı veya sürümü değil; normal tüketici gaming laptoplarında devasa MoE modellerini minimum gecikme, yüksek quant seviyeleri ve deterministik doğrulukla yerel olarak çalıştırmak üzere sıfırdan kurgulanmış **bağımsız bir MoE çıkarım platformu ve launcher'ıdır**. `llama.cpp` ve `ggml`, mimarinin altında yalnızca bir alt seviye tensör ve donanım hesaplama arka ucu (backend engine) olarak yer almaktadır.

Tendou Launcher Phase 9 (v0.9.0-stable), tek-token decode döngüsündeki kritik yol darboğazlarını ortadan kaldıran ve çok-tokenli prefill aşamasını asenkronlaştıran üç büyük mühendislik kilometre taşını (Milestone 1, 2, 3) hayata geçirmiştir:

1. **GPU `ffn_shexp` $\longleftrightarrow$ CPU MoE (`MUL_MAT_ID`) Zamanlayıcı Örtüşmesi (Milestone 1):**
   - `qwen4exp.cpp` ve `qwen3next.cpp` tensör graflarında paylaşılan uzman (`ffn_shexp`), Router `weights` ardına taşınarak Split 0 bitiminde 10.2 KB D2H erken senkronize edilmiş; CPU MoE hesaplanırken GPU üzerinde `ffn_shexp` hesabı eşzamanlı olarak örtüştürülmüştür.
2. **1x4 AVX2/FMA `ggml_vec_dot_iq3_xxs_q8_K_x4` Çekirdeği (Milestone 2):**
   - `arch/x86/quants.c` içine 1x4 yazmaç-bloklu vektörel çarpım çekirdeği entegre edilmiştir. `q8_K` aktivasyon tensörü süper-blok başına tek sefer yüklenerek 4 satır paralel çarpılmış, bellek okuma trafiği %30 azaltılırken tekil `vec_dot` ile **0-ULP bit-exact parite** güvenceye alınmıştır.
3. **Decode `PrefetchVirtualMemory` ve GPU `cb_eval` Bypass (Milestone 1 & 2):**
   - Tek-token decode sırasında her tokende çalışan `PrefetchVirtualMemory` syscall yükü ve `cb_eval` bekleme duraklamaları temizlenerek tek-token CPU hesaplama süresi ~5.5 ms/tok kısaltılmıştır.
4. **32K Asimetrik KV (`q8_0/q4_0`) ve Sıfır-Kopya State Cache (Milestone 3):**
   - $N_{\text{ctx}} > 8192$ bağlamlarında asimetrik KV önbelleklemesi (`type_k = q8_0`, `type_v = q4_0`) zorunlu kılınarak 32K bağlamda VRAM taşması önlenmiştir.
   - `AtlasHybridStateCache` içine `std::shared_ptr` sıfır-kopya tamponlama ve $O(1)$ bellek temizliği (`llama_memory_clear(..., false)`) entegre edilmiştir.

---

## 2. Doğrulanan Çalışma Profili (Production Tier-1: 8GB GDDR6)

- **Başlatma Parametreleri:**
  ```powershell
  llama-atlas-engine.exe -m models/qwen3.8-gsq-rco-iq3_xxs/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
    --ipc -t 8 -c 2048 -ngl 99 --atlas-gpu-first --atlas-gpu-layers 2 --atlas-vram-cache 3200 \
    -ctk q8_0 -ctv q8_0 --boost --atlas-boost-level 2 --atlas-k 5 --min-p 0.0 --atlas-mtp off
  ```
- **Fiziksel VRAM Tahsisi:**
  - Kullanılan VRAM: **6,629.4 MiB**
  - Boş VRAM Güvenlik Payı: **1,521.6 MiB** (Zorunlu $\ge 1,024\text{ MiB}$ kuralı tam sağlandı)
  - WDDM PCIe Paging: **0.0 MiB (Sıfır Paging)**

---

## 3. Fiziksel VRAM Tahsis Tablosu (RTX 5060 Laptop 8GB)

| Bileşen | Yerleşim | Boyut (MiB) | Açıklama |
|---|---|---|---|
| Dense Attention & Norms (48 Katman) | CUDA0 (VRAM) | ~2,150.0 | Tüm katmanların self-attention ve layernorm ağırlıkları |
| KV Cache (48 Katman, 2k-32k context) | CUDA0 (VRAM) | ~2,048.0 | Standart modda Q8_0, 32K'da Q8_0/Q4_0 asimetrik |
| Layer 0 & Layer 1 MoE Experts (1,024 exp) | CUDA0 (VRAM) | ~1,250.0 | 2 tam katman GPU MoE (gate, up, down) |
| Output Layer / LM Head (Aliased) | CUDA0 (VRAM) | ~528.0 | Paylaşımlı token sözlük projeksiyonu |
| CUDA Context & Runtime Scratch Buffers | CUDA0 (VRAM) | ~653.4 | CUDA allocator ve ara aktivasyon tamponları |
| **Toplam Aktif VRAM** | **CUDA0** | **6,629.4 MiB** | **WDDM Paging = 0.0 MiB** |
| **Kalan Boş VRAM Güvenlik Alanı** | **CUDA0** | **1,521.6 MiB** | **Donanım Kararlılık Payı ($\ge 1,024\text{ MiB}$)** |

---

## 4. Phase 8 vs Phase 9 Doğrulanmış 100-Token Benchmark Özeti

5 farklı görevde 100'er token (`max_tokens = 100`) üretilerek elde edilen uçtan uca karşılaştırma:

| Metrik | Phase 8 (`v8.0-stable`) | Phase 9 (`v9.0-milestone1-2-3`) | Net İyileşme ($\Delta$) |
|---|---|---|---|
| **Soğuk (Cold) TTFT** | 5,965.1 ms | **353.9 ms** | **16.9x Daha Hızlı (-94.1%)** |
| **Sıcak (Warm) TTFT** | 249.2 ms | **192.4 ms** | **-22.8% (< 200 ms Kilidi)** |
| **Sıcak Decode TPS** | 19.52 TPS | **21.58 TPS** (Zirve: **23.86 TPS**) | **+10.6% (+2.06 TPS)** |
| **Soğuk Efektif TPS** | 6.86 TPS | **15.45 TPS** | **+125.2% (+8.59 TPS)** |
| **Sıcak p50 Gecikme** | 50.4 – 55.2 ms/tok | **39.6 – 49.2 ms/tok** | **-6.0 ila -12.3 ms/tok** |
| **Sıcak DDR5 Ayak İzi** | 32.32 MB/tok | **22.47 MB/tok** | **-30.5% Bellek Trafiği Tasarrufu** |
| **Boş VRAM Güvenlik Payı** | 1,513.8 MiB | **1,521.6 MiB** | +7.8 MiB Alan Kazanımı |
| **WDDM PCIe Paging** | 0.0 MiB | **0.0 MiB** | %100 Donanımsal Kararlılık |
| **Deterministik Doğruluk** | 4/5 PASS (1 erken stop) | **5/5 PASS (%100 Parite)** | Tüm görevler 100 tok tam doğru |

---

## 5. Dondurulan Arşiv Dosyaları ve Test Güvencesi

- **Tendou Launcher Yerel Çekirdek İkilileri:** `bin_phase9/` dizini altında (`llama-atlas-engine.exe`, `GGML-CPU.DLL`, `GGML-CPU.LIB`).
- **Kaynak Eşitlemesi:** `src/atlas/` $\longleftrightarrow$ `llama.cpp/examples/atlas-engine/` `fc.exe /b` ile 0 bayt fark (`0 diff`).
- **Alt Seviye Tensör Arka Uç (Compute Backend) Adaptasyon Yaması:** `patches/llama.cpp-phase9-stable.patch` (356,067 bayt, BOM'suz UTF-8).
- **Test Kapsamı:**
  - `atlas-native-tests.exe`: 18/18 PASS (%100).
  - `atlas-partition-tests.exe`: PASS (%100).
  - `pytest`: 188 passed, 4 skipped (%100).
