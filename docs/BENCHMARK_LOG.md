# tendou launcher — Sürekli Benchmark Rapor Günlüğü (Continuous Benchmark Log)

Bu dosya her yeni benchmark çalıştırıldığında append-only (sürekli son satırdan eklemeli)
olarak güncellenir. Her test koşusunda metrikler (TTFT, prefill compute, decode TPS,
efektif TPS, prompt cache kullanımı) ve yapay zeka tarafından üretilen tam metin çıktısı
(AI generated output) eksiksiz olarak kaydedilir.

---
## Benchmark Koşusu: `bench-20260914-180128-037c` (2026-09-14T18:01:28.273062)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-09-14T18:01:28.273062`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S`
- **Donanım**: CPU: `AMD Ryzen 7 260 (8 Physical Zen 4 Cores)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU (8151 MiB VRAM)` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `398.20 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 9125.0 | 5.35 | 0.31 | COLD (0) | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 5627.2 | 9.83 | 0.34 | COLD (0) | PASS |
| `32k-turn1-cold` | 32768 | `q8_0` | 186 | 17 | 0 | 37785.8 | 8.36 | 0.40 | COLD (0) | PASS |
| `32k-turn2-warm` | 32768 | `q8_0` | 186 | 49 | 0 | 39219.4 | 10.34 | 1.09 | COLD (0) | PASS |
| `64k-turn1-analysis` | 65536 | `q4_0` | 189 | 45 | 0 | 47051.7 | 6.98 | 0.83 | COLD (0) | PASS |
| `64k-turn2-cache-hit` | 65536 | `q4_0` | 176 | 23 | 0 | 44760.2 | 6.99 | 0.46 | COLD (0) | PASS |
| `128k-turn1-eval` | 131072 | `q4_0` | 169 | 15 | 0 | 43853.9 | 5.67 | 0.30 | COLD (0) | PASS |
| `128k-turn2-cache-hit` | 131072 | `q4_0` | 174 | 41 | 0 | 44120.2 | 5.54 | 0.78 | COLD (0) | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`9124.98 ms`, Decode=`5.35 TPS`, Efektif=`0.31 TPS`, Toplam=`9607.56 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`8958.20 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`162.14 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|>
<|im_start|>assistant
<think>

</...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`5627.24 ms`, Decode=`9.83 TPS`, Efektif=`0.34 TPS`, Toplam=`5820.04 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`5581.09 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`42.10 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|>
<|im_start|>assistant
<...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `32k-turn1-cold` | Context: 32768 | Turn: 1
- **Kategori / İş Yükü**: `architecture` / `multi_turn_cache`
- **Context & KV Belleği**: `32768` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=186, Üretilen=17, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`37785.80 ms`, Decode=`8.36 TPS`, Efektif=`0.40 TPS`, Toplam=`39671.36 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`37532.18 ms`, gpu_compute=`93.00 ms`, prefill_io=`0.00 ms`, checkpoint=`242.27 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system
You are an expert AI software architect for high-performance low-latency MoE inference engines.<|im_end|>
<|im_start|>user
Context Document:
Tendou launcher is a...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
NVIDIA RTX 5060 Laptop, 8 GB GDDR6<|im_end|>
```

#### [4] Görev: `32k-turn2-warm` | Context: 32768 | Turn: 2
- **Kategori / İş Yükü**: `architecture` / `multi_turn_cache`
- **Context & KV Belleği**: `32768` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=186, Üretilen=49, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`39219.44 ms`, Decode=`10.34 TPS`, Efektif=`1.09 TPS`, Toplam=`43865.05 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`38834.62 ms`, gpu_compute=`93.00 ms`, prefill_io=`0.00 ms`, checkpoint=`350.46 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system
You are an expert AI software architect for high-performance low-latency MoE inference engines.<|im_end|>
<|im_start|>user
Context Document:
Tendou launcher is a...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
The module responsible for asynchronous H2D transfer is **P3 Asynchronous H2D Transfer Pipeline**, which employs **pinned staging buffers** to enable zero-copy, non-blocking host-to-device memory transfers synchronized via **CUDA event fences**.<|im_end|>
```

#### [5] Görev: `64k-turn1-analysis` | Context: 65536 | Turn: 1
- **Kategori / İş Yükü**: `memory_scaling` / `large_context_cache`
- **Context & KV Belleği**: `65536` tokens (KV: `q4_0`)
- **Token Sayaçları**: Prompt=189, Üretilen=45, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`47051.71 ms`, Decode=`6.98 TPS`, Efektif=`0.83 TPS`, Toplam=`53322.73 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`46966.26 ms`, gpu_compute=`94.50 ms`, prefill_io=`0.00 ms`, checkpoint=`75.16 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system
You are an accurate technical documentation analyzer specializing in large language model serving.<|im_end|>
<|im_start|>user
Context Document:
Section 64K-MoE: ...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
According to the text, the memory sizes are as follows:

*   **fp16 KV cache:** 3.4 GB
*   **Q4_0 KV cache:** approximately 850 MB<|im_end|>
```

#### [6] Görev: `64k-turn2-cache-hit` | Context: 65536 | Turn: 2
- **Kategori / İş Yükü**: `memory_scaling` / `large_context_cache`
- **Context & KV Belleği**: `65536` tokens (KV: `q4_0`)
- **Token Sayaçları**: Prompt=176, Üretilen=23, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`44760.25 ms`, Decode=`6.99 TPS`, Efektif=`0.46 TPS`, Toplam=`47895.87 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`44281.05 ms`, gpu_compute=`88.00 ms`, prefill_io=`0.00 ms`, checkpoint=`453.42 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system
You are an accurate technical documentation analyzer specializing in large language model serving.<|im_end|>
<|im_start|>user
Context Document:
Section 64K-MoE: ...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
According to the section, prefix prompt caching guarantees that **repetitive context prefix evaluation is bypassed completely**.<|im_end|>
```

#### [7] Görev: `128k-turn1-eval` | Context: 131072 | Turn: 1
- **Kategori / İş Yükü**: `extreme_context` / `ultra_context_cache`
- **Context & KV Belleği**: `131072` tokens (KV: `q4_0`)
- **Token Sayaçları**: Prompt=169, Üretilen=15, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`43853.89 ms`, Decode=`5.67 TPS`, Efektif=`0.30 TPS`, Toplam=`46311.23 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`43734.38 ms`, gpu_compute=`84.50 ms`, prefill_io=`0.00 ms`, checkpoint=`48.56 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system
You are a senior systems engineer specializing in extreme context MoE runtime benchmarks.<|im_end|>
<|im_start|>user
Context Document:
Protocol 128K-Serving: The...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
The prompt checkpoint budget is locked at **1024 MB**.<|im_end|>
```

#### [8] Görev: `128k-turn2-cache-hit` | Context: 131072 | Turn: 2
- **Kategori / İş Yükü**: `extreme_context` / `ultra_context_cache`
- **Context & KV Belleği**: `131072` tokens (KV: `q4_0`)
- **Token Sayaçları**: Prompt=174, Üretilen=41, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`44120.18 ms`, Decode=`5.54 TPS`, Efektif=`0.78 TPS`, Toplam=`51289.85 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`43992.29 ms`, gpu_compute=`87.00 ms`, prefill_io=`0.00 ms`, checkpoint=`51.58 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system
You are a senior systems engineer specializing in extreme context MoE runtime benchmarks.<|im_end|>
<|im_start|>user
Context Document:
Protocol 128K-Serving: The...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
Based on the provided context document:

*   **Decode TPS target:** 10+ TPS
*   **Prefill reuse speedup:** up to 100x TTFT reduction<|im_end|>
```


---
## Benchmark Koşusu: `bench-20260914-180902-cd44` (2026-09-14T18:09:02.307292)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-09-14T18:09:02.307292`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S`
- **Donanım**: CPU: `AMD Ryzen 7 260 (8 Physical Zen 4 Cores)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU (8151 MiB VRAM)` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `153.21 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 9071.4 | 5.42 | 0.31 | COLD (0) | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 4006.8 | 9.76 | 0.48 | COLD (0) | PASS |
| `32k-turn1-cold` | 32768 | `q8_0` | 186 | 17 | 0 | 38779.3 | 8.70 | 0.39 | COLD (0) | PASS |
| `32k-turn2-warm` | 32768 | `q8_0` | 235 | 31 | 0 | 51202.0 | 9.75 | 0.55 | COLD (0) | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`9071.35 ms`, Decode=`5.42 TPS`, Efektif=`0.31 TPS`, Toplam=`9604.70 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`8890.68 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`176.62 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|>
<|im_start|>assistant
<think>

</...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`4006.82 ms`, Decode=`9.76 TPS`, Efektif=`0.48 TPS`, Toplam=`4196.59 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`3956.58 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`46.60 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|>
<|im_start|>assistant
<...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `32k-turn1-cold` | Context: 32768 | Turn: 1
- **Kategori / İş Yükü**: `architecture` / `multi_turn_cache`
- **Context & KV Belleği**: `32768` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=186, Üretilen=17, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`38779.33 ms`, Decode=`8.70 TPS`, Efektif=`0.39 TPS`, Toplam=`40593.16 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`38522.94 ms`, gpu_compute=`93.00 ms`, prefill_io=`0.00 ms`, checkpoint=`242.10 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system
You are an expert AI software architect for high-performance low-latency MoE inference engines.<|im_end|>
<|im_start|>user
Context Document:
Tendou launcher is a...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
NVIDIA RTX 5060 Laptop, 8 GB GDDR6<|im_end|>
```

#### [4] Görev: `32k-turn2-warm` | Context: 32768 | Turn: 2
- **Kategori / İş Yükü**: `architecture` / `multi_turn_cache`
- **Context & KV Belleği**: `32768` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=235, Üretilen=31, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`51202.03 ms`, Decode=`9.75 TPS`, Efektif=`0.55 TPS`, Toplam=`54273.43 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`50958.81 ms`, gpu_compute=`117.50 ms`, prefill_io=`0.00 ms`, checkpoint=`222.81 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system
You are an expert AI software architect for high-performance low-latency MoE inference engines.<|im_end|>
<|im_start|>user
Context Document:
Tendou launcher is a...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
Module P3 (Asynchronous H2D Transfer Pipeline) uses pinned staging buffers and CUDA event fences to overlap host-to-device transfers with kernel execution.<|im_end|>
```


---
## Benchmark Koşusu: `bench-20260914-181715-ac66` (2026-09-14T18:17:15.638081)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-09-14T18:17:15.638081`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S`
- **Donanım**: CPU: `AMD Ryzen 7 260 (8 Physical Zen 4 Cores)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU (8151 MiB VRAM)` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `76.44 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `32k-turn1-cold` | 32768 | `q8_0` | 186 | 17 | 0 | 38200.3 | 8.83 | 0.40 | COLD (0) | PASS |
| `32k-turn2-warm` | 32768 | `q8_0` | 237 | 31 | 186 | 12044.4 | 9.38 | 1.97 | HIT (186) | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `32k-turn1-cold` | Context: 32768 | Turn: 1
- **Kategori / İş Yükü**: `architecture` / `multi_turn_cache`
- **Context & KV Belleği**: `32768` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=186, Üretilen=17, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`38200.30 ms`, Decode=`8.83 TPS`, Efektif=`0.40 TPS`, Toplam=`39991.76 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`37979.27 ms`, gpu_compute=`93.00 ms`, prefill_io=`0.00 ms`, checkpoint=`212.73 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system
You are an expert AI software architect for high-performance low-latency MoE inference engines.<|im_end|>
<|im_start|>user
Context Document:
Tendou launcher is a...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
NVIDIA RTX 5060 Laptop, 8 GB GDDR6<|im_end|>
```

#### [2] Görev: `32k-turn2-warm` | Context: 32768 | Turn: 2
- **Kategori / İş Yükü**: `architecture` / `multi_turn_cache`
- **Context & KV Belleği**: `32768` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=237, Üretilen=31, Cache Reused=186 (CACHE HIT (186 tokens reused))
- **Performans**: TTFT=`12044.36 ms`, Decode=`9.38 TPS`, Efektif=`1.97 TPS`, Toplam=`15241.61 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`11851.68 ms`, gpu_compute=`25.50 ms`, prefill_io=`0.00 ms`, checkpoint=`158.32 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system
You are an expert AI software architect for high-performance low-latency MoE inference engines.<|im_end|>
<|im_start|>user
Context Document:
Tendou launcher is a...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
Module P3 (Asynchronous H2D Transfer Pipeline) uses pinned staging buffers and CUDA event fences to overlap host-to-device transfers with kernel execution.<|im_end|>
```


---
## Benchmark Koşusu: `bench-20260914-182622-b53f` (2026-09-14T18:26:22.172133)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-09-14T18:26:22.172133`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `97.52 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `64k-turn1-analysis` | 65536 | `q4_0` | 189 | 45 | 0 | 48309.3 | 6.48 | 0.80 | COLD (0) [115.2MB] | PASS |
| `64k-turn2-cache-hit` | 65536 | `q4_0` | 263 | 23 | 189 | 17074.2 | 5.89 | 1.06 | HIT (189) [230.8MB] | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `64k-turn1-analysis` | Context: 65536 | Turn: 1
- **Kategori / İş Yükü**: `memory_scaling` / `large_context_cache`
- **Context & KV Belleği**: `65536` tokens (KV: `q4_0`)
- **Token Sayaçları**: Prompt=189, Üretilen=45, Cache Reused=0 (COLD PREFILL, prompt_cache=115.24 MB)
- **Performans**: TTFT=`48309.29 ms`, Decode=`6.48 TPS`, Efektif=`0.80 TPS`, Toplam=`55060.04 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`48250.59 ms`, gpu_compute=`94.50 ms`, prefill_io=`0.00 ms`, checkpoint=`48.35 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are an accurate technical documentation analyzer specializing in large language model serving.<|im_end|> <|im_start|>user Context Document: Section 64K-MoE: ...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
According to the text, the memory sizes are as follows:

*   **fp16 KV cache:** 3.4 GB
*   **Q4_0 KV cache:** approximately 850 MB<|im_end|>
```

#### [2] Görev: `64k-turn2-cache-hit` | Context: 65536 | Turn: 2
- **Kategori / İş Yükü**: `memory_scaling` / `large_context_cache`
- **Context & KV Belleği**: `65536` tokens (KV: `q4_0`)
- **Token Sayaçları**: Prompt=263, Üretilen=23, Cache Reused=189 (CACHE HIT (189 tokens reused), prompt_cache=230.82 MB)
- **Performans**: TTFT=`17074.17 ms`, Decode=`5.89 TPS`, Efektif=`1.06 TPS`, Toplam=`20796.68 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`16925.72 ms`, gpu_compute=`37.00 ms`, prefill_io=`0.00 ms`, checkpoint=`106.44 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are an accurate technical documentation analyzer specializing in large language model serving.<|im_end|> <|im_start|>user Context Document: Section 64K-MoE: ...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
According to the text, prefix prompt caching guarantees that **repetitive context prefix evaluation is bypassed completely**.<|im_end|>
```


---
## Benchmark Koşusu: `bench-20260914-182805-d790` (2026-09-14T18:28:05.799674)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-09-14T18:28:05.799674`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `95.43 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `128k-turn1-eval` | 131072 | `q4_0` | 169 | 15 | 0 | 44723.9 | 5.80 | 0.30 | COLD (0) [115.1MB] | PASS |
| `128k-turn2-cache-hit` | 131072 | `q4_0` | 220 | 35 | 169 | 12661.0 | 6.15 | 1.87 | HIT (169) [115.5MB] | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `128k-turn1-eval` | Context: 131072 | Turn: 1
- **Kategori / İş Yükü**: `extreme_context` / `ultra_context_cache`
- **Context & KV Belleği**: `131072` tokens (KV: `q4_0`)
- **Token Sayaçları**: Prompt=169, Üretilen=15, Cache Reused=0 (COLD PREFILL, prompt_cache=115.06 MB)
- **Performans**: TTFT=`44723.90 ms`, Decode=`5.80 TPS`, Efektif=`0.30 TPS`, Toplam=`47125.89 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`44602.54 ms`, gpu_compute=`84.50 ms`, prefill_io=`0.00 ms`, checkpoint=`53.38 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a senior systems engineer specializing in extreme context MoE runtime benchmarks.<|im_end|> <|im_start|>user Context Document: Protocol 128K-Serving: The...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
The prompt checkpoint budget is locked at **1024 MB**.<|im_end|>
```

#### [2] Görev: `128k-turn2-cache-hit` | Context: 131072 | Turn: 2
- **Kategori / İş Yükü**: `extreme_context` / `ultra_context_cache`
- **Context & KV Belleği**: `131072` tokens (KV: `q4_0`)
- **Token Sayaçları**: Prompt=220, Üretilen=35, Cache Reused=169 (CACHE HIT (169 tokens reused), prompt_cache=115.52 MB)
- **Performans**: TTFT=`12660.99 ms`, Decode=`6.15 TPS`, Efektif=`1.87 TPS`, Toplam=`18195.70 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`12516.04 ms`, gpu_compute=`25.50 ms`, prefill_io=`0.00 ms`, checkpoint=`45.94 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a senior systems engineer specializing in extreme context MoE runtime benchmarks.<|im_end|> <|im_start|>user Context Document: Protocol 128K-Serving: The...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
Protocol 128K-Serving targets **10+ TPS** for decode, and prefill reuse provides up to **100x TTFT reduction**.<|im_end|>
```


---
## Benchmark Koşusu: `bench-20260915-023341-37b8` (2026-09-15T02:33:41.760687)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-09-15T02:33:41.760687`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `98.04 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `128k-turn1-eval` | 131072 | `q4_0` | 169 | 15 | 0 | 46578.5 | 6.01 | 0.29 | COLD (0) [115.1MB] | PASS |
| `128k-turn2-cache-hit` | 131072 | `q4_0` | 220 | 35 | 169 | 12633.1 | 6.39 | 1.89 | HIT (169) [115.5MB] | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `128k-turn1-eval` | Context: 131072 | Turn: 1
- **Kategori / İş Yükü**: `extreme_context` / `ultra_context_cache`
- **Context & KV Belleği**: `131072` tokens (KV: `q4_0`)
- **Token Sayaçları**: Prompt=169, Üretilen=15, Cache Reused=0 (COLD PREFILL, prompt_cache=115.06 MB)
- **Performans**: TTFT=`46578.51 ms`, Decode=`6.01 TPS`, Efektif=`0.29 TPS`, Toplam=`48900.75 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`46441.45 ms`, gpu_compute=`84.50 ms`, prefill_io=`0.00 ms`, checkpoint=`52.44 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a senior systems engineer specializing in extreme context MoE runtime benchmarks.<|im_end|> <|im_start|>user Context Document: Protocol 128K-Serving: The...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
The prompt checkpoint budget is locked at **1024 MB**.<|im_end|>
```

#### [2] Görev: `128k-turn2-cache-hit` | Context: 131072 | Turn: 2
- **Kategori / İş Yükü**: `extreme_context` / `ultra_context_cache`
- **Context & KV Belleği**: `131072` tokens (KV: `q4_0`)
- **Token Sayaçları**: Prompt=220, Üretilen=35, Cache Reused=169 (CACHE HIT (169 tokens reused), prompt_cache=115.52 MB)
- **Performans**: TTFT=`12633.08 ms`, Decode=`6.39 TPS`, Efektif=`1.89 TPS`, Toplam=`17967.74 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`12463.33 ms`, gpu_compute=`25.50 ms`, prefill_io=`0.00 ms`, checkpoint=`44.93 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a senior systems engineer specializing in extreme context MoE runtime benchmarks.<|im_end|> <|im_start|>user Context Document: Protocol 128K-Serving: The...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
Protocol 128K-Serving targets **10+ TPS** for decode, and prefill reuse provides up to **100x TTFT reduction**.<|im_end|>
```


---
## Benchmark Koşusu: `bench-20260915-024000-2e6a` (2026-09-15T02:40:00.260157)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-09-15T02:40:00.260157`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `106.96 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `128k-turn1-eval` | 131072 | `q4_0` | 169 | 15 | 0 | 29177.5 | 6.04 | 0.44 | COLD (0) [115.1MB] | PASS |
| `128k-turn2-cache-hit` | 131072 | `q4_0` | 220 | 35 | 169 | 8111.6 | 6.42 | 2.53 | HIT (169) [115.5MB] | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `128k-turn1-eval` | Context: 131072 | Turn: 1
- **Kategori / İş Yükü**: `extreme_context` / `ultra_context_cache`
- **Context & KV Belleği**: `131072` tokens (KV: `q4_0`)
- **Token Sayaçları**: Prompt=169, Üretilen=15, Cache Reused=0 (COLD PREFILL, prompt_cache=115.06 MB)
- **Performans**: TTFT=`29177.48 ms`, Decode=`6.04 TPS`, Efektif=`0.44 TPS`, Toplam=`31485.44 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`28995.27 ms`, gpu_compute=`84.50 ms`, prefill_io=`0.00 ms`, checkpoint=`103.91 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a senior systems engineer specializing in extreme context MoE runtime benchmarks.<|im_end|> <|im_start|>user Context Document: Protocol 128K-Serving: The...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
The prompt checkpoint budget is locked at **1024 MB**.<|im_end|>
```

#### [2] Görev: `128k-turn2-cache-hit` | Context: 131072 | Turn: 2
- **Kategori / İş Yükü**: `extreme_context` / `ultra_context_cache`
- **Context & KV Belleği**: `131072` tokens (KV: `q4_0`)
- **Token Sayaçları**: Prompt=220, Üretilen=35, Cache Reused=169 (CACHE HIT (169 tokens reused), prompt_cache=115.52 MB)
- **Performans**: TTFT=`8111.60 ms`, Decode=`6.42 TPS`, Efektif=`2.53 TPS`, Toplam=`13426.91 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`7896.38 ms`, gpu_compute=`25.50 ms`, prefill_io=`0.00 ms`, checkpoint=`99.50 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a senior systems engineer specializing in extreme context MoE runtime benchmarks.<|im_end|> <|im_start|>user Context Document: Protocol 128K-Serving: The...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
Protocol 128K-Serving targets **10+ TPS** for decode, and prefill reuse provides up to **100x TTFT reduction**.<|im_end|>
```


---
## Benchmark Koşusu: `bench-20260915-025512-f2ef` (2026-09-15T02:55:12.363597)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-09-15T02:55:12.363597`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `65.58 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `128k-turn1-eval` | 131072 | `q4_0` | 169 | 15 | 0 | 24371.2 | 5.70 | 0.52 | COLD (0) [229.3MB] | PASS |
| `128k-turn2-cache-hit` | 131072 | `q4_0` | 220 | 37 | 184 | 4697.8 | 6.28 | 3.38 | HIT (184) [230.4MB] | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `128k-turn1-eval` | Context: 131072 | Turn: 1
- **Kategori / İş Yükü**: `extreme_context` / `ultra_context_cache`
- **Context & KV Belleği**: `131072` tokens (KV: `q4_0`)
- **Token Sayaçları**: Prompt=169, Üretilen=15, Cache Reused=0 (COLD PREFILL, prompt_cache=229.31 MB)
- **Performans**: TTFT=`24371.20 ms`, Decode=`5.70 TPS`, Efektif=`0.52 TPS`, Toplam=`26992.61 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`24160.83 ms`, gpu_compute=`84.50 ms`, prefill_io=`0.00 ms`, checkpoint=`176.68 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a senior systems engineer specializing in extreme context MoE runtime benchmarks.<|im_end|> <|im_start|>user Context Document: Protocol 128K-Serving: The...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
The prompt checkpoint budget is locked at **1024 MB**.<|im_end|>
```

#### [2] Görev: `128k-turn2-cache-hit` | Context: 131072 | Turn: 2
- **Kategori / İş Yükü**: `extreme_context` / `ultra_context_cache`
- **Context & KV Belleği**: `131072` tokens (KV: `q4_0`)
- **Token Sayaçları**: Prompt=220, Üretilen=37, Cache Reused=184 (CACHE HIT (184 tokens reused), prompt_cache=230.44 MB)
- **Performans**: TTFT=`4697.82 ms`, Decode=`6.28 TPS`, Efektif=`3.38 TPS`, Toplam=`10650.78 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`4521.16 ms`, gpu_compute=`18.00 ms`, prefill_io=`0.00 ms`, checkpoint=`116.53 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a senior systems engineer specializing in extreme context MoE runtime benchmarks.<|im_end|> <|im_start|>user Context Document: Protocol 128K-Serving: The...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
Protocol 128K-Serving targets a decode throughput of **10+ TPS**, and prefill reuse provides up to a **100x TTFT reduction**.<|im_end|>
```


---
## Benchmark Koşusu: `bench-20260915-025641-45e2` (2026-09-15T02:56:41.901307)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-09-15T02:56:41.901307`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `55.88 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `32k-turn1-cold` | 32768 | `q8_0` | 186 | 17 | 0 | 22697.7 | 5.85 | 0.62 | COLD (0) [232.8MB] | PASS |
| `32k-turn2-warm` | 32768 | `q8_0` | 237 | 20 | 203 | 4537.1 | 4.97 | 2.22 | HIT (203) [234.6MB] | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `32k-turn1-cold` | Context: 32768 | Turn: 1
- **Kategori / İş Yükü**: `architecture` / `multi_turn_cache`
- **Context & KV Belleği**: `32768` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=186, Üretilen=17, Cache Reused=0 (COLD PREFILL, prompt_cache=232.77 MB)
- **Performans**: TTFT=`22697.65 ms`, Decode=`5.85 TPS`, Efektif=`0.62 TPS`, Toplam=`25821.29 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`22588.95 ms`, gpu_compute=`93.00 ms`, prefill_io=`0.00 ms`, checkpoint=`151.89 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are an expert AI software architect for high-performance low-latency MoE inference engines.<|im_end|> <|im_start|>user Context Document: Tendou launcher is a...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
NVIDIA RTX 5060 Laptop, 8 GB GDDR6<|im_end|>
```

#### [2] Görev: `32k-turn2-warm` | Context: 32768 | Turn: 2
- **Kategori / İş Yükü**: `architecture` / `multi_turn_cache`
- **Context & KV Belleği**: `32768` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=237, Üretilen=20, Cache Reused=203 (CACHE HIT (203 tokens reused), prompt_cache=234.57 MB)
- **Performans**: TTFT=`4537.06 ms`, Decode=`4.97 TPS`, Efektif=`2.22 TPS`, Toplam=`8563.66 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`4395.90 ms`, gpu_compute=`17.00 ms`, prefill_io=`0.00 ms`, checkpoint=`123.06 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are an expert AI software architect for high-performance low-latency MoE inference engines.<|im_end|> <|im_start|>user Context Document: Tendou launcher is a...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
Module P3 uses pinned staging buffers and CUDA event fences to enable asynchronous H2D transfers.<|im_end|>
```


---
## Benchmark Koşusu: `bench-20260915-030312-d9f2` (2026-09-15T03:03:12.936493)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-09-15T03:03:12.936493`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `66.17 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `128k-turn1-eval` | 131072 | `q4_0` | 169 | 15 | 0 | 23606.0 | 5.73 | 0.53 | COLD (0) [229.3MB] | PASS |
| `128k-turn2-cache-hit` | 131072 | `q4_0` | 220 | 37 | 184 | 4897.3 | 6.14 | 3.28 | HIT (184) [230.4MB] | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `128k-turn1-eval` | Context: 131072 | Turn: 1
- **Kategori / İş Yükü**: `extreme_context` / `ultra_context_cache`
- **Context & KV Belleği**: `131072` tokens (KV: `q4_0`)
- **Token Sayaçları**: Prompt=169, Üretilen=15, Cache Reused=0 (COLD PREFILL, prompt_cache=229.31 MB)
- **Performans**: TTFT=`23605.99 ms`, Decode=`5.73 TPS`, Efektif=`0.53 TPS`, Toplam=`26229.96 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`23446.93 ms`, gpu_compute=`84.50 ms`, prefill_io=`0.00 ms`, checkpoint=`136.90 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a senior systems engineer specializing in extreme context MoE runtime benchmarks.<|im_end|> <|im_start|>user Context Document: Protocol 128K-Serving: The...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
The prompt checkpoint budget is locked at **1024 MB**.<|im_end|>
```

#### [2] Görev: `128k-turn2-cache-hit` | Context: 131072 | Turn: 2
- **Kategori / İş Yükü**: `extreme_context` / `ultra_context_cache`
- **Context & KV Belleği**: `131072` tokens (KV: `q4_0`)
- **Token Sayaçları**: Prompt=220, Üretilen=37, Cache Reused=184 (CACHE HIT (184 tokens reused), prompt_cache=230.44 MB)
- **Performans**: TTFT=`4897.30 ms`, Decode=`6.14 TPS`, Efektif=`3.28 TPS`, Toplam=`10980.28 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`4718.84 ms`, gpu_compute=`18.00 ms`, prefill_io=`0.00 ms`, checkpoint=`124.52 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a senior systems engineer specializing in extreme context MoE runtime benchmarks.<|im_end|> <|im_start|>user Context Document: Protocol 128K-Serving: The...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
Protocol 128K-Serving targets a decode throughput of **10+ TPS**, and prefill reuse provides up to a **100x TTFT reduction**.<|im_end|>
```


---
## Benchmark Koşusu: `bench-20260915-030930-2db3` (2026-09-15T03:09:30.012999)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-09-15T03:09:30.012999`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `57.91 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `128k-turn1-eval` | 131072 | `q4_0` | 169 | 15 | 0 | 22725.6 | 5.48 | 0.55 | COLD (0) [229.3MB] | PASS |
| `128k-turn2-cache-hit` | 131072 | `q4_0` | 220 | 37 | 184 | 4088.2 | 6.14 | 3.54 | HIT (184) [230.4MB] | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `128k-turn1-eval` | Context: 131072 | Turn: 1
- **Kategori / İş Yükü**: `extreme_context` / `ultra_context_cache`
- **Context & KV Belleği**: `131072` tokens (KV: `q4_0`)
- **Token Sayaçları**: Prompt=169, Üretilen=15, Cache Reused=0 (COLD PREFILL, prompt_cache=229.31 MB)
- **Performans**: TTFT=`22725.58 ms`, Decode=`5.48 TPS`, Efektif=`0.55 TPS`, Toplam=`25446.93 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`22573.05 ms`, gpu_compute=`84.50 ms`, prefill_io=`0.00 ms`, checkpoint=`129.57 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a senior systems engineer specializing in extreme context MoE runtime benchmarks.<|im_end|> <|im_start|>user Context Document: Protocol 128K-Serving: The...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
The prompt checkpoint budget is locked at **1024 MB**.<|im_end|>
```

#### [2] Görev: `128k-turn2-cache-hit` | Context: 131072 | Turn: 2
- **Kategori / İş Yükü**: `extreme_context` / `ultra_context_cache`
- **Context & KV Belleği**: `131072` tokens (KV: `q4_0`)
- **Token Sayaçları**: Prompt=220, Üretilen=37, Cache Reused=184 (CACHE HIT (184 tokens reused), prompt_cache=230.44 MB)
- **Performans**: TTFT=`4088.18 ms`, Decode=`6.14 TPS`, Efektif=`3.54 TPS`, Toplam=`10170.17 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`3905.31 ms`, gpu_compute=`18.00 ms`, prefill_io=`0.00 ms`, checkpoint=`128.41 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a senior systems engineer specializing in extreme context MoE runtime benchmarks.<|im_end|> <|im_start|>user Context Document: Protocol 128K-Serving: The...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
Protocol 128K-Serving targets a decode throughput of **10+ TPS**, and prefill reuse provides up to a **100x TTFT reduction**.<|im_end|>
```


---
## Benchmark Koşusu: `bench-20260915-031223-7372` (2026-09-15T03:12:23.787280)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-09-15T03:12:23.787280`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `55.53 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `128k-turn1-eval` | 131072 | `q4_0` | 169 | 15 | 0 | 23606.4 | 4.90 | 0.53 | COLD (0) [229.3MB] | PASS |
| `128k-turn2-cache-hit` | 131072 | `q4_0` | 220 | 35 | 184 | 1831.9 | 6.08 | 4.47 | HIT (184) [230.4MB] | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `128k-turn1-eval` | Context: 131072 | Turn: 1
- **Kategori / İş Yükü**: `extreme_context` / `ultra_context_cache`
- **Context & KV Belleği**: `131072` tokens (KV: `q4_0`)
- **Token Sayaçları**: Prompt=169, Üretilen=15, Cache Reused=0 (COLD PREFILL, prompt_cache=229.31 MB)
- **Performans**: TTFT=`23606.39 ms`, Decode=`4.90 TPS`, Efektif=`0.53 TPS`, Toplam=`26620.96 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`23454.56 ms`, gpu_compute=`84.50 ms`, prefill_io=`0.00 ms`, checkpoint=`133.73 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a senior systems engineer specializing in extreme context MoE runtime benchmarks.<|im_end|> <|im_start|>user Context Document: Protocol 128K-Serving: The...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
The prompt checkpoint budget is locked at **1024 MB**.<|im_end|>
```

#### [2] Görev: `128k-turn2-cache-hit` | Context: 131072 | Turn: 2
- **Kategori / İş Yükü**: `extreme_context` / `ultra_context_cache`
- **Context & KV Belleği**: `131072` tokens (KV: `q4_0`)
- **Token Sayaçları**: Prompt=220, Üretilen=35, Cache Reused=184 (CACHE HIT (184 tokens reused), prompt_cache=230.42 MB)
- **Performans**: TTFT=`1831.88 ms`, Decode=`6.08 TPS`, Efektif=`4.47 TPS`, Toplam=`7612.78 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`1663.32 ms`, gpu_compute=`18.00 ms`, prefill_io=`0.00 ms`, checkpoint=`119.49 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a senior systems engineer specializing in extreme context MoE runtime benchmarks.<|im_end|> <|im_start|>user Context Document: Protocol 128K-Serving: The...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
Protocol 128K-Serving targets **10+ TPS** for decode, and prefill reuse provides up to **100x TTFT reduction**.<|im_end|>
```


---
## Benchmark Koşusu: `bench-20260915-031329-3f45` (2026-09-15T03:13:29.747485)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-09-15T03:13:29.747485`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `49.36 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `32k-turn1-cold` | 32768 | `q8_0` | 186 | 17 | 0 | 20477.0 | 5.86 | 0.68 | COLD (0) [232.8MB] | PASS |
| `32k-turn2-warm` | 32768 | `q8_0` | 237 | 19 | 203 | 2259.4 | 6.73 | 3.53 | HIT (203) [234.6MB] | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `32k-turn1-cold` | Context: 32768 | Turn: 1
- **Kategori / İş Yükü**: `architecture` / `multi_turn_cache`
- **Context & KV Belleği**: `32768` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=186, Üretilen=17, Cache Reused=0 (COLD PREFILL, prompt_cache=232.77 MB)
- **Performans**: TTFT=`20476.99 ms`, Decode=`5.86 TPS`, Efektif=`0.68 TPS`, Toplam=`23394.73 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`20382.89 ms`, gpu_compute=`93.00 ms`, prefill_io=`0.00 ms`, checkpoint=`133.16 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are an expert AI software architect for high-performance low-latency MoE inference engines.<|im_end|> <|im_start|>user Context Document: Tendou launcher is a...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
NVIDIA RTX 5060 Laptop, 8 GB GDDR6<|im_end|>
```

#### [2] Görev: `32k-turn2-warm` | Context: 32768 | Turn: 2
- **Kategori / İş Yükü**: `architecture` / `multi_turn_cache`
- **Context & KV Belleği**: `32768` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=237, Üretilen=19, Cache Reused=203 (CACHE HIT (203 tokens reused), prompt_cache=234.55 MB)
- **Performans**: TTFT=`2259.41 ms`, Decode=`6.73 TPS`, Efektif=`3.53 TPS`, Toplam=`5092.22 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2155.78 ms`, gpu_compute=`17.00 ms`, prefill_io=`0.00 ms`, checkpoint=`110.92 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are an expert AI software architect for high-performance low-latency MoE inference engines.<|im_end|> <|im_start|>user Context Document: Tendou launcher is a...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
P3 Asynchronous H2D Transfer Pipeline uses pinned staging buffers and CUDA event fences.<|im_end|>
```


