# Tendou Launcher — Release Notes: Phase 6 (v0.6.0)

**Sürüm:** v0.6.0 (TLEP Phase 6 Stable)  
**Tarih:** 2026-10-15  
**Temel Model:** Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS (48 Layer, 512 Experts/Layer, Dynamic-K)  
**Hedef Donanım:** AMD Ryzen 7 260 (8C/16T, DDR5-5600), NVIDIA GeForce RTX 5060 Laptop (8,151 MiB VRAM), 16 GB DDR5 RAM  

---

## 1. Genel Bakış ve Mimari Yenilikler

Phase 6, Dual-Engine mimarisi üzerinde CPU ve GPU eşzamanlı hesaplamasını birleştirmiş, Zen 4 AVX-512 / AVX2 VNNI 8-satır vektörel unrolling ile IQ3_XXS formatında 12.3–14.1 TPS bandına ulaşmıştır. Ayrıca L15–L30 katmanlarında akıl yürütme budama derinliği `ATLAS_REASONING_K = 4` seviyesine çekilerek `short-math-01 = 323` (%100 Exact Match) deterministik doğruluğu geri kazanılmıştır.

---

## 2. Doğrulanan Çalışma Profilleri (Operating Tiers)

### Profil 1: Tier-1 — 8GB Native Co-Processing (Varsayılan Üretim Profili)
- **Konfigürasyon:** `--mtp-mode off --atlas-boost-level 2` (`gpu_expert_layers = 2`)
- **Açıklama:** Layer 0 ve Layer 1'in 512'şer MoE uzmanının tamamı (toplam 1,024 uzman ~ 1,250 MiB) + 48 Dense Attention + Norms + KV Cache GPU VRAM üzerinde tutulur. Kalan 46 katmanın MoE uzmanları Zen 4 CPU threadpool üzerinde AVX-512 unrolled kernel ile hesaplanır.
- **VRAM Tüketimi:**
  - Toplam VRAM: 8,151.0 MiB
  - Kullanılan VRAM: **6,680.9 MiB**
  - Boş VRAM Güvenlik Payı: **1,470.1 MiB** (Zorunlu > 1,024 MiB tavanı güvence altında; WDDM paging = 0 MiB)
- **Doğruluk & Determinizm:** %100 Exact Token ID Paritesi (`short-math-01`: 323, `short-code-01`: 42, `repetitive-code-01`: print).

### Profil 2: Tier-2 — 12GB+ VRAM-Resident MTP (Genişletilmiş VRAM Profili)
- **Konfigürasyon:** `--mtp-mode on` (Opsiyonel / Geniş VRAM donanımları veya harici GPU için)
- **Açıklama:** Speculative Multi-Token Prediction (MTP) taslak motoru tamamen GPU VRAM üzerinde barındırılır. 8GB VRAM sınırında PCIe çekişmesini önlemek için varsayılan olarak kapalıdır (`--mtp-mode off`); 12GB+ VRAM sistemlerinde kabul oranını %84.3'e çıkararak ilave decode çarpanı sağlar.

---

## 3. Fiziksel VRAM Tahsis Tablosu (Tier-1: RTX 5060 8GB)

| Bileşen | Yerleşim | Boyut (MiB) | Açıklama |
|---|---|---|---|
| Dense Attention & Norms (48 Katman) | CUDA0 (VRAM) | ~2,150.0 | Tüm katmanların self-attention ve layernorm ağırlıkları |
| KV Cache (48 Katman, 4k-32k context) | CUDA0 (VRAM) | ~2,048.0 | FP16/Q8 KV önbellek havuzu |
| Layer 0 & Layer 1 MoE Experts (1,024 exp) | CUDA0 (VRAM) | ~1,250.0 | 2 tam katman MoE (gate, up, down) |
| Output Layer / LM Head (Aliased) | CUDA0 (VRAM) | ~528.0 | Paylaşımlı token sözlük projeksiyonu |
| CUDA Context & Runtime Scratch Buffers | CUDA0 (VRAM) | ~704.9 | CUDA allocator ve ara aktivasyon tamponları |
| **Toplam Aktif VRAM** | **CUDA0** | **6,680.9 MiB** | **WDDM Paging = 0 MiB** |
| **Kalan Boş VRAM Güvenlik Alanı** | **CUDA0** | **1,470.1 MiB** | **Donanım Kararlılık Payı** |

---

## 4. Phase 6 Doğrulanmış Benchmark Sonuçları

| Görev | Beklenen Çıktı | Phase 6 Decode TPS | DDR5 Ayak İzi (MB/tok) | Durum |
|---|---|---|---|---|
| `short-math-01` | `323` | 8.68 – 9.36 TPS | 67.1 – 92.4 MB/tok | %100 PASS |
| `short-code-01` | `42` | 10.54 – 11.36 TPS | ~65.0 MB/tok | %100 PASS |
| `short-math-measure` | `323` | 12.24 – 12.30 TPS | 155.38 MB/tok | %100 PASS |
| `math-explain-01` | `17 ile 19` / `323` | 12.80 – 13.32 TPS | 110.2 MB/tok | %100 PASS |
| `repetitive-code-01` | `print` | 13.50 – 14.08 TPS | 62.19 MB/tok | %100 PASS |

---

## 5. Dondurulan Arşiv Dosyaları
- `patches/llama.cpp-phase6-stable.patch`: Phase 6 kararlı C++ çekirdek yaması.
- Test Suite: `atlas-native-tests.exe` (14/14 PASS), `atlas-partition-tests.exe` (PASS), `pytest` (188 passed, 4 skipped).
