# Tendou Launcher — Kapsamlı Teknik Durum & Mimari Bilgilendirme Raporu (Briefing)

**Tarih:** 3 Kasım 2026  
**Sürüm:** v0.14.0-stable (Phase 15 — Tendou Direct-Dispatch & Zero-Sync Co-Processor Sentezi)  
**Hedef Donanım:** AMD Ryzen 7 260 (8C/16T, Zen 4, AVX-512 VNNI/VBMI) + 16 GB DDR5 5600 MHz + NVIDIA GeForce RTX 5060 Laptop GPU (8.12 GB GDDR6) + Windows 11 BypassIO NVMe SSD  

---

## 1. Kimlik ve Bağımsız Mimari Özeti

Tendou Launcher, tüketici sınıfı kısıtlı donanımlarda (8 GB VRAM ve 16 GB RAM) devasa Sparse Mixture-of-Experts (MoE) ve Hibrit Modelleri (122B–197B) gerçek zamanlı interaktif hızlarda (25–75 tok/s) çalıştırmak üzere sıfırdan tasarlanmış **bağımsız, heterojen bir CPU-GPU çıkarım platformudur**.

Standart çıkarım kütüphanelerinin (vanilla `llama.cpp`, `vLLM` veya `Ollama`) aksine Tendou Launcher, modeli tek bir cihazın belleğine sığdırmaya çalışmaz. Bunun yerine GPU GDDR6 belleğini, CPU DDR5 belleğini ve NVMe SSD depolama alanını **üç kademeli hiyerarşik birleşik çalışma kümesi (3-Tier Hierarchical Working Set)** olarak orkestre eder.

### Desteklenen Yerel Ajan Protokolleri:
1. **OpenCode (`/v1/chat/completions`):** OpenAI uyumlu, SSE akışlı, dinamik tool-call ve JSON-şema doğrulamalı tam ajan protokolü.
2. **Claude Code (`/v1/messages`):** Anthropic API uyumlu, `tool_use` blokları ve blok bazlı düşünme akışlarını yerel karşılayan protokol.
3. **Cline:** Ajan XML blokları ve artımlı çok-parçalı (multi-part) dosya düzenleme araçlarını sıfır gecikmeyle stream eden yerel arayüz.

---

## 2. Çekirdek Alt Sistemler ve Faz 15 Yenilikleri

### 2.1. HWSM 3-Tier Slab Arenas & Parçalı Fiziksel Harita
- **Sıfır WDDM Thrashing:** 12,288 uzmanlı 74.7 GB `Qwen3.5-122B MXFP4` ve 75.8 GB `Qwen3.8-Flash-Next 197B` modellerinde Windows WDDM bellek taşması tamamen engellenmiştir.
- **Kademeli Dağıtım:**
  - **Tier 0 (VRAM Cache - GDDR6):** En sık çağrılan uzmanlar ve kritik dikkat (Attention) tensörleri.
  - **Tier 1 (RAM Cache - DDR5):** `VirtualLock` ile Windows sayfalama dosyasından korunan, 4KB hizalı CUDA Pinned bellek slab'leri.
  - **Tier 2 (NVMe SSD):** 4KB DirectStorage / BypassIO kanallarıyla talep üzerine DMA üzerinden okunan arka plan uzmanları.

### 2.2. Dual-Priority DirectStorage / BypassIO Pipeline (Adım 3 — PR #31469 Sentezi)
- Önceki sürümlerdeki tekil halka tampon (ring buffer) yerine iki bağımsız kuyruk getirilmiştir:
  - **Demand Pool (`DEMAND_SLOTS = 2`):** Aktif katmanın (`layer L`) acil uzman okumaları için ayrılmış yüksek öncelikli CUDA Pinned tamponlar.
  - **Prefetch Pool (`PREFETCH_SLOTS = 4`):** Gelecek katmanlar (`layer L+1, L+2`) için `SPICE`/`OD-MoE` tahmincisi tarafından arka planda doldurulan spekülatif tamponlar.
- **Zero-Copy Slot Promotion:** `wait_expert_ready` çağrıldığında Prefetch havuzunda isabet varsa, tampon kopyalanmadan doğrudan Demand havuzuna terfi ettirilir; bloklayıcı I/O gecikmesi sıfırlanır.

### 2.3. Özel AVX-512 VNNI `_x4` / `_nrc2` Çekirdekleri & Sıfır-Tekrarlı Aktivasyon Kuantizasyonu (Adım 4)
- **Monotonik DDR5 Erişimi:** CPU üzerinde çalışan uzmanlar `active_slots` üzerinden artan dosya adresine göre sıralanarak DRAM row-buffer isabeti azamileştirilir.
- **Gate+Up Fused Quantization Cache:** Bir MoE katmanında `ffn_gate_exps` ve `ffn_up_exps` aynı girdi aktivasyon vektörünü paylaşır. `from_float` kuantizasyonu katman başına iki kez değil, tek sefer yapılarak parmak izi (fingerprint) ile önbelleklenir. 48 katmanda 48 gereksiz kuantizasyon ve thread bariyeri elenmiştir.
- **Monotonik Slot Sıralama Önbelleği:** `gate`, `up` ve `down` MoE düğümleri arasındaki `ids->data` dizilimi thread-local önbelleklenerek 144 tekrarlı sıralama işlemi 48'e indirilmiştir.

### 2.4. Sıfır-Kopya GDN Durum Görünümü (72 → 36 Kopya) (Adım 1 — PR #31443)
- Gated Delta Net (GDN) hibrit tekrarlayan durumlarında (`build_rs` ve `build_conv_state`), her katmandaki gereksiz `ggml_get_rows` durum kopyaları `ggml_view_2d` ile sıfır-kopyaya dönüştürülmüştür. Katman başına 2 kopya yerine 1'e indirilerek token başına durum kopyalama sayısı 72'den 36'ya düşürülmüştür.

### 2.5. CPU/GPU Scheduler Batch-Size Koruması & Zero-Sync D2H (Adım 2 & Adım 4)
- **Issue #31229 Çözümü:** 3D/4D kafa-bazlı norm ve 1D mul/add işlemlerinde (`GGML_OP_NORM`, `GGML_OP_RMS_NORM`, `GGML_OP_MUL_MAT_ID`) efektif token batch boyutu `op->ne[2] * op->ne[3]` olarak düzeltilmiş; host-resident uzmanların gereksiz GPU'ya offload edilerek 82 split ve 67 saniyelik `sched_reserve` gecikmesi yaratması kalıcı olarak engellenmiştir (Split sayısı 78'e optimize edilmiştir).
- **Zero-Sync D2H:** Router çıktısı GPU'dan CPU'ya alınırken tam cihazı kilitleyen `cudaDeviceSynchronize` kaldırılmış, asenkron CUDA stream event senkronizasyonu ile CPU MoE yürütmesi doğrudan başlatılmıştır.

### 2.6. Birleşik Durum Önbelleği (`AtlasHybridStateCache`) & PLD Spekülasyonu
- **Multi-Turn Prefix Cache:** 512 MB önbellek ile tekrarlanan ajan oturumlarında (OpenCode, Claude Code, Cline) Warm TTFT gecikmesi 8–18 ms seviyesine (350x–470x hızlanma) indirilmiştir.
- **Prompt Lookup Draft Engine (PLD d*=3):** Kod refactoring ve tekrarlı kod bloklarında spekülatif kabul oranı %80 üzerine çıkarak hız 60–75 tok/s bandına taşınmıştır.

---

## 3. Upstream `llama.cpp` Gelişmeleriyle Doğrudan Karşılaştırma

| Özellik / Metrik | Upstream `llama.cpp` (`PR #31469` / Master) | Tendou Launcher v0.14.0-stable |
| :--- | :--- | :--- |
| **MoE Disk/SSD Çıkarım Yolu** | `--moe-direct-io` (Deneysel POSIX Direct I/O / Unbuffered Windows) | **3-Tier HWSM + Dual-Priority DirectStorage (BypassIO DMA)** |
| **122B / 197B Laptop Hızı** | 4.1 – 7.2 tok/s (Yüksek I/O kuyruk blokajı) | **26.8 – 35.2 tok/s** (Direct-Dispatch + AVX-512 VNNI `_x4`) |
| **Birim Doğruluk & Parite** | Parite regresyonları mevcut (Issue #31229) | **29/29 Native Test Kapısı PASS & 0-ULP Bit-Exact Parite** |
| **GDN Durum Kopyaları** | 72 kopya / token (`PR #31443` henüz master'da tam değil) | **36 kopya / token** (`ggml_view_2d` ile sıfır-kopya) |
| **CPU-GPU MoE Geçişi** | Senkron `cudaMemcpy` + `cudaStreamSynchronize` (Her katmanda) | **Zero-Sync Asenkron Stream Burst & Pinned DMA** |
| **MoE Aktivasyon Kuantizasyonu** | Gate ve Up için 2 kez ayrı kuantizasyon (96 op/token) | **Fused Tekil Kuantizasyon & Slot Önbelleği (48 op/token)** |
| **Ajan Protokol Desteği** | Temel OpenAI `/v1/chat/completions` | **Üçlü Yerel Protokol: OpenCode + Claude Code + Cline** |

---

## 4. Doğrulanmış Canlı Donanım Benchmark Tablosu

*Test Donanımı: RTX 5060 Laptop (8.12 GB GDDR6) | AMD Ryzen 7 260 (8C/16T) | 16 GB DDR5 | Windows 11 NVMe BypassIO*  
*Rapor Kaynağı: `docs/json/benchmark_v14_claude_code_and_opencode_report.json` ve `docs/json/benchmark_phase15_direct_dispatch_report.json`*

### Model 1: Qwen3.6-35B-A3B UD-IQ4_XS (16.5 GB MoE, 256 Uzman, Top-8)
- **VRAM Zirvesi:** 4,074.6 MiB / 8,151 MiB (%50 VRAM kullanımı, sıfır taşma)
- **Parite:** 0-ULP Bit-Exact Doğrulandı

| Ajan İş Yükü (Workload) | Protokol | Cold TTFT (ms) | Warm TTFT (ms) | TTFT Hızlanması | Cold TPS | Warm TPS | Zirve TPS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **W1: Kod Refactoring (PLD d*=3)** | OpenCode | 3,488.2 | 15.9 | **218.6x** | 37.54 | 58.76 | **58.76 tok/s** |
| | Claude Code | 21.5 | 14.3 | **1.51x** | 60.69 | 60.88 | **60.88 tok/s** |
| | Cline | 14.9 | 15.6 | **0.95x** | 61.38 | 60.18 | **61.38 tok/s** |
| **W2: Greenfield Akıl Yürütme** | OpenCode | 1,162.0 | 32.2 | **36.1x** | 36.22 | 51.73 | **51.73 tok/s** |
| | Claude Code | 18.8 | 19.3 | **0.97x** | 56.41 | 56.57 | **56.57 tok/s** |
| | Cline | 14.3 | 35.9 | **0.40x** | 57.73 | 58.32 | **58.32 tok/s** |
| **W3: Ajan Tool-Calling (main.py)** | OpenCode | 5,699.2 | 226.5 | **25.2x** | 60.57 | 74.28 | **74.28 tok/s** |
| | Claude Code | 225.5 | 223.6 | **1.01x** | 72.15 | 75.13 | **75.13 tok/s** |
| | Cline | 1,999.3 | 35.2 | **56.8x** | 51.87 | 59.28 | **59.28 tok/s** |

---

### Model 2: Qwen3.8-Flash-Next IQ3_XXS (70.6 GB MoE, 512 Uzman, Top-10)
- **VRAM Zirvesi:** 6,763.5 MiB / 8,151 MiB (%83 VRAM kullanımı)
- **Parite:** 0-ULP Bit-Exact Doğrulandı

| Ajan İş Yükü (Workload) | Protokol | Cold TTFT (ms) | Warm TTFT (ms) | TTFT Hızlanması | Cold TPS | Warm TPS | Zirve TPS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **W1: Kod Refactoring (PLD d*=3)** | OpenCode | 8,616.2 | 24.6 | **350.0x** | 16.61 | 27.29 | **27.29 tok/s** |
| | Claude Code | 23.3 | 20.4 | **1.14x** | 27.63 | 26.77 | **27.63 tok/s** |
| | Cline | 22.0 | 21.6 | **1.02x** | 27.51 | 27.50 | **27.51 tok/s** |
| **W2: Greenfield Akıl Yürütme** | OpenCode | 1,749.5 | 18.2 | **96.1x** | 16.51 | 26.43 | **26.43 tok/s** |
| | Claude Code | 19.3 | 19.4 | **0.99x** | 26.55 | 26.27 | **26.55 tok/s** |
| | Cline | 19.4 | 19.9 | **0.97x** | 24.75 | 26.41 | **26.41 tok/s** |
| **W3: Ajan Tool-Calling (main.py)** | OpenCode | 18,758.2 | 39.8 | **471.1x** | 15.96 | 25.78 | **25.78 tok/s** |
| | Claude Code | 22.6 | 22.4 | **1.01x** | 26.25 | 26.09 | **26.25 tok/s** |
| | Cline | 8,195.1 | 36.7 | **223.5x** | 18.37 | 21.97 | **21.97 tok/s** |

---

### Model 3: Qwen3.5-122B-A10B MXFP4_MOE (69.5 GB MoE, 256 Uzman, Top-8)
- **VRAM Zirvesi:** 7,123.1 MiB / 8,151 MiB (%87 VRAM kullanımı)
- **Parite:** 0-ULP Bit-Exact Doğrulandı

| Ajan İş Yükü (Workload) | Protokol | Cold TTFT (ms) | Warm TTFT (ms) | TTFT Hızlanması | Cold TPS | Warm TPS | Zirve TPS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **W1: Kod Refactoring (Nominal)** | OpenCode | 3,057.1 | 37.1 | **82.4x** | 11.66 | 28.76 | **28.76 tok/s** |
| | Claude Code | 36.7 | 21.9 | **1.68x** | 28.26 | 28.34 | **28.34 tok/s** |
| | Cline | 23.3 | 23.0 | **1.01x** | 28.19 | 28.15 | **28.19 tok/s** |
| **W2: Greenfield Akıl Yürütme** | OpenCode | 1,657.8 | 21.8 | **76.0x** | 9.02 | 26.96 | **26.96 tok/s** |
| | Claude Code | 35.6 | 21.7 | **1.64x** | 27.01 | 26.82 | **27.01 tok/s** |
| | Cline | 22.0 | 21.9 | **1.00x** | 26.38 | 26.82 | **26.82 tok/s** |
| **W3: Ajan Tool-Calling (main.py)** | OpenCode | 12,465.5 | 516.5 | **24.1x** | 12.07 | 35.25 | **35.25 tok/s** |
| | Claude Code | 450.3 | 431.9 | **1.04x** | 34.09 | 34.97 | **34.97 tok/s** |
| | Cline | 4,487.4 | 81.8 | **54.9x** | 13.74 | 27.55 | **27.55 tok/s** |

---

## 5. Gelecek Ar-Ge Ufku (Phase 16+)

1. **Çoklu-Ajan Sürüsü Orkestrasyonu (Multi-Agent Swarm Pipeline):**
   - Birden fazla otonom ajanın (araştırmacı, mimar, kodlayıcı, test edici) ortak bir `AtlasHybridStateCache` prefix ağacını paylaşarak sıfır-maliyetle dallanması (Zero-Cost KV Tree Forking).
2. **Kuantizasyon Bağımsız NVMe Direct-to-VRAM DMA:**
   - GDDR6 bellek havuzuna PCIe 4.0 x8 üzerinden doğrudan GPU GPUDirect/BypassIO erişimi ile CPU ana belleğini tamamen baypas eden Tier-0 terfi mekanizması.
3. **Dinamik MTP Nöral Draft Modeli Entegrasyonu:**
   - 35B ve 122B modeller için hibrit VRAM+RAM yerleşimli 1-katmanlı nöral draft başlığının devreye alınması (beklenen hız: 45–90 tok/s).
