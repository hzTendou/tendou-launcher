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


---
## Benchmark Koşusu: `bench-20260926-143121-3dea` (2026-09-26T14:31:21.061216)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-09-26T14:31:21.061216`
- **Model**: `Qwen3.6-35B-A3B-UD-IQ4_XS`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=16, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `26.96 s`
- **Notlar**: Qwen3.6-35B-A3B UD-IQ4_XS continuous benchmark with physical map

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 2 | 0 | 1467.7 | 0.00 | 0.63 | COLD (0) [127.6MB] | FAIL |
| `short-code-01` | 2048 | `q8_0` | 52 | 24 | 0 | 825.0 | 26.88 | 13.90 | COLD (0) [127.9MB] | FAIL |
| `32k-turn1-cold` | 32768 | `q8_0` | 186 | 48 | 0 | 3091.5 | 32.42 | 10.45 | COLD (0) [130.9MB] | PASS |
| `32k-turn2-warm` | 32768 | `q8_0` | 269 | 60 | 234 | 446.4 | 33.16 | 25.99 | HIT (234) [132.8MB] | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=2, Cache Reused=0 (COLD PREFILL, prompt_cache=127.55 MB)
- **Performans**: TTFT=`1467.74 ms`, Decode=`0.00 TPS`, Efektif=`0.63 TPS`, Toplam=`1592.39 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`1424.73 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`70.09 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
,<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=24, Cache Reused=0 (COLD PREFILL, prompt_cache=127.91 MB)
- **Performans**: TTFT=`825.03 ms`, Decode=`26.88 TPS`, Efektif=`13.90 TPS`, Toplam=`1726.88 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`777.19 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`73.32 ms`
- **Bitiş Nedeni**: `length`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
````text
It meant to be a string literal.

```python
def string_literal():
    return "Python ifaces
````

#### [3] Görev: `32k-turn1-cold` | Context: 32768 | Turn: 1
- **Kategori / İş Yükü**: `architecture` / `multi_turn_cache`
- **Context & KV Belleği**: `32768` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=186, Üretilen=48, Cache Reused=0 (COLD PREFILL, prompt_cache=130.94 MB)
- **Performans**: TTFT=`3091.52 ms`, Decode=`32.42 TPS`, Efektif=`10.45 TPS`, Toplam=`4593.64 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`3049.45 ms`, gpu_compute=`93.00 ms`, prefill_io=`0.00 ms`, checkpoint=`67.80 ms`
- **Bitiş Nedeni**: `length`
- **Prompt**: `<|im_start|>system You are an expert AI software architect for high-performance low-latency MoE inference engines.<|im_end|> <|im_start|>user Context Document: Tendou launcher is a...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
The primary GPU model is the NVIDIA RTX 3060 Ti, which features VRAM 8GB GDDR6.
</think>

The primary GPU model is the NVIDIA RTX 3060 Ti, which features VRAM
```

#### [4] Görev: `32k-turn2-warm` | Context: 32768 | Turn: 2
- **Kategori / İş Yükü**: `architecture` / `multi_turn_cache`
- **Context & KV Belleği**: `32768` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=269, Üretilen=60, Cache Reused=234 (CACHE HIT (234 tokens reused), prompt_cache=132.79 MB)
- **Performans**: TTFT=`446.36 ms`, Decode=`33.16 TPS`, Efektif=`25.99 TPS`, Toplam=`2270.07 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`353.60 ms`, gpu_compute=`17.50 ms`, prefill_io=`0.00 ms`, checkpoint=`99.08 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are an expert AI software architect for high-performance low-latency MoE inference engines.<|im_end|> <|im_start|>user Context Document: Tendou launcher is a...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
The module responsible for asynchronous H2D transfer is the CUDA Stream Manager, which utilizes a double-buffer mechanism to overlap data transfer with computation.
</think>

The module responsible for asynchronous H2D transfer is the CUDA Stream Manager, which utilizes a double-buffer mechanism to overlap data transfer with computation.<|im_end|>
```


---
## Benchmark Koşusu: `bench-20260926-143154-2b21` (2026-09-26T14:31:54.513908)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-09-26T14:31:54.513908`
- **Model**: `Qwen3.6-35B-A3B-UD-IQ4_XS`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=16, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `28.36 s`
- **Notlar**: Qwen3.6-35B-A3B UD-IQ4_XS 64k and 128k extreme context scaling

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `64k-turn1-analysis` | 65536 | `q4_0` | 189 | 26 | 0 | 3634.7 | 33.90 | 5.62 | COLD (0) [128.8MB] | FAIL |
| `64k-turn2-cache-hit` | 65536 | `q4_0` | 244 | 19 | 215 | 322.0 | 34.27 | 19.85 | HIT (215) [129.4MB] | PASS |
| `128k-turn1-eval` | 131072 | `q4_0` | 169 | 22 | 0 | 3340.2 | 30.22 | 5.15 | COLD (0) [128.6MB] | FAIL |
| `128k-turn2-cache-hit` | 131072 | `q4_0` | 227 | 13 | 191 | 337.2 | 31.20 | 15.71 | HIT (191) [129.2MB] | FAIL |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `64k-turn1-analysis` | Context: 65536 | Turn: 1
- **Kategori / İş Yükü**: `memory_scaling` / `large_context_cache`
- **Context & KV Belleği**: `65536` tokens (KV: `q4_0`)
- **Token Sayaçları**: Prompt=189, Üretilen=26, Cache Reused=0 (COLD PREFILL, prompt_cache=128.80 MB)
- **Performans**: TTFT=`3634.67 ms`, Decode=`33.90 TPS`, Efektif=`5.62 TPS`, Toplam=`4450.84 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`3588.00 ms`, gpu_compute=`94.50 ms`, prefill_io=`0.00 ms`, checkpoint=`73.89 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are an accurate technical documentation analyzer specializing in large language model serving.<|im_end|> <|im_start|>user Context Document: Section 64K-MoE: ...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
The memory size of fp16 KV cache at 64k context according to the text is 34 bytes.<|im_end|>
```

#### [2] Görev: `64k-turn2-cache-hit` | Context: 65536 | Turn: 2
- **Kategori / İş Yükü**: `memory_scaling` / `large_context_cache`
- **Context & KV Belleği**: `65536` tokens (KV: `q4_0`)
- **Token Sayaçları**: Prompt=244, Üretilen=19, Cache Reused=215 (CACHE HIT (215 tokens reused), prompt_cache=129.37 MB)
- **Performans**: TTFT=`322.04 ms`, Decode=`34.27 TPS`, Efektif=`19.85 TPS`, Toplam=`906.67 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`252.88 ms`, gpu_compute=`14.50 ms`, prefill_io=`0.00 ms`, checkpoint=`80.09 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are an accurate technical documentation analyzer specializing in large language model serving.<|im_end|> <|im_start|>user Context Document: Section 64K-MoE: ...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
According to the section, prefix prompt caching guarantees that repetitive context prefixes are bypassed completely.<|im_end|>
```

#### [3] Görev: `128k-turn1-eval` | Context: 131072 | Turn: 1
- **Kategori / İş Yükü**: `extreme_context` / `ultra_context_cache`
- **Context & KV Belleği**: `131072` tokens (KV: `q4_0`)
- **Token Sayaçları**: Prompt=169, Üretilen=22, Cache Reused=0 (COLD PREFILL, prompt_cache=128.56 MB)
- **Performans**: TTFT=`3340.24 ms`, Decode=`30.22 TPS`, Efektif=`5.15 TPS`, Toplam=`4075.43 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`3293.36 ms`, gpu_compute=`84.50 ms`, prefill_io=`0.00 ms`, checkpoint=`71.22 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a senior systems engineer specializing in extreme context MoE runtime benchmarks.<|im_end|> <|im_start|>user Context Document: Protocol 128K-Serving: The...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
You are a senior systems engineer specializing in extreme context MoE (Mixture of Experts) runtime benchmarks.<|im_end|>
```

#### [4] Görev: `128k-turn2-cache-hit` | Context: 131072 | Turn: 2
- **Kategori / İş Yükü**: `extreme_context` / `ultra_context_cache`
- **Context & KV Belleği**: `131072` tokens (KV: `q4_0`)
- **Token Sayaçları**: Prompt=227, Üretilen=13, Cache Reused=191 (CACHE HIT (191 tokens reused), prompt_cache=129.15 MB)
- **Performans**: TTFT=`337.25 ms`, Decode=`31.20 TPS`, Efektif=`15.71 TPS`, Toplam=`763.94 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`271.74 ms`, gpu_compute=`18.00 ms`, prefill_io=`0.00 ms`, checkpoint=`77.19 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a senior systems engineer specializing in extreme context MoE runtime benchmarks.<|im_end|> <|im_start|>user Context Document: Protocol 128K-Serving: The...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
You are a senior systems engineer specializing in extreme context MoE<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261003-181621-7c0e` (2026-10-03T18:16:21.944053)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-03T18:16:21.944053`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=16, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `81.12 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 9859.6 | 7.38 | 0.29 | COLD (0) [227.7MB] | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 3569.0 | 6.24 | 0.49 | COLD (0) [227.9MB] | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL, prompt_cache=227.74 MB)
- **Performans**: TTFT=`9859.56 ms`, Decode=`7.38 TPS`, Efektif=`0.29 TPS`, Toplam=`10373.86 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`9726.53 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`149.22 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL, prompt_cache=227.93 MB)
- **Performans**: TTFT=`3569.01 ms`, Decode=`6.24 TPS`, Efektif=`0.49 TPS`, Toplam=`4068.39 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`3480.82 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`128.40 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261003-181758-4c34` (2026-10-03T18:17:58.854238)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-03T18:17:58.854238`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=16, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `25.01 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 7 | 0 | 2284.4 | 6.90 | 1.85 | COLD (0) [227.8MB] | FAIL |
| `short-code-01` | 2048 | `q8_0` | 52 | 2 | 0 | 1246.8 | 0.00 | 0.63 | COLD (0) [227.9MB] | FAIL |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=7, Cache Reused=0 (COLD PREFILL, prompt_cache=227.79 MB)
- **Performans**: TTFT=`2284.39 ms`, Decode=`6.90 TPS`, Efektif=`1.85 TPS`, Toplam=`3235.04 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2168.60 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`169.13 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
17 * 19<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=2, Cache Reused=0 (COLD PREFILL, prompt_cache=227.91 MB)
- **Performans**: TTFT=`1246.75 ms`, Decode=`0.00 TPS`, Efektif=`0.63 TPS`, Toplam=`1583.29 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`1183.75 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`109.48 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
4<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261003-184515-3cdd` (2026-10-03T18:45:15.174812)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-03T18:45:15.174812`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=16, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `30.20 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 3 | 0 | 5836.9 | 0.43 | 0.23 | COLD (0) [227.7MB] | FAIL |
| `short-code-01` | 2048 | `q8_0` | 52 | 14 | 0 | 1327.5 | 10.06 | 4.84 | COLD (0) [228.1MB] | FAIL |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=3, Cache Reused=0 (COLD PREFILL, prompt_cache=227.72 MB)
- **Performans**: TTFT=`5836.86 ms`, Decode=`0.43 TPS`, Efektif=`0.23 TPS`, Toplam=`8814.89 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`5708.56 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`163.59 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
17<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=14, Cache Reused=0 (COLD PREFILL, prompt_cache=228.12 MB)
- **Performans**: TTFT=`1327.46 ms`, Decode=`10.06 TPS`, Efektif=`4.84 TPS`, Toplam=`2686.43 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`1243.40 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`125.06 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
sum([12, 14, 16])<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261003-184555-4353` (2026-10-03T18:45:55.397664)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-03T18:45:55.397664`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=16, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `28.88 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 4866.0 | 5.21 | 0.55 | COLD (0) [227.7MB] | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 2828.1 | 6.79 | 0.62 | COLD (0) [227.9MB] | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL, prompt_cache=227.74 MB)
- **Performans**: TTFT=`4865.97 ms`, Decode=`5.21 TPS`, Efektif=`0.55 TPS`, Toplam=`5485.60 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`4762.01 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`159.37 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL, prompt_cache=227.93 MB)
- **Performans**: TTFT=`2828.07 ms`, Decode=`6.79 TPS`, Efektif=`0.62 TPS`, Toplam=`3250.13 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2712.98 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`161.36 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261003-184947-a7c1` (2026-10-03T18:49:47.602019)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-03T18:49:47.602019`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=16, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `29.93 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 4863.9 | 8.08 | 0.56 | COLD (0) [227.7MB] | FAIL |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 4090.0 | 6.42 | 0.44 | COLD (0) [227.9MB] | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL, prompt_cache=227.74 MB)
- **Performans**: TTFT=`4863.88 ms`, Decode=`8.08 TPS`, Efektif=`0.56 TPS`, Toplam=`5352.58 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`4754.78 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`152.65 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
343<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL, prompt_cache=227.93 MB)
- **Performans**: TTFT=`4090.02 ms`, Decode=`6.42 TPS`, Efektif=`0.44 TPS`, Toplam=`4531.20 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`3990.21 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`145.08 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261003-185107-a5a6` (2026-10-03T18:51:07.316121)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-03T18:51:07.316121`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=16, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `28.55 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 4397.9 | 7.31 | 0.61 | COLD (0) [227.7MB] | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 3038.4 | 8.76 | 0.59 | COLD (0) [227.9MB] | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL, prompt_cache=227.74 MB)
- **Performans**: TTFT=`4397.85 ms`, Decode=`7.31 TPS`, Efektif=`0.61 TPS`, Toplam=`4897.80 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`4277.02 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`165.03 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL, prompt_cache=227.93 MB)
- **Performans**: TTFT=`3038.39 ms`, Decode=`8.76 TPS`, Efektif=`0.59 TPS`, Toplam=`3413.63 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2951.02 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`126.56 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261003-205941-66a2` (2026-10-03T20:59:41.410024)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-03T20:59:41.410024`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `35.83 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 7042.6 | 4.68 | 0.39 | COLD (0) [227.7MB] | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 3331.0 | 7.17 | 0.54 | COLD (0) [227.9MB] | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL, prompt_cache=227.74 MB)
- **Performans**: TTFT=`7042.60 ms`, Decode=`4.68 TPS`, Efektif=`0.39 TPS`, Toplam=`7730.41 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`6944.87 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`149.22 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL, prompt_cache=227.93 MB)
- **Performans**: TTFT=`3331.03 ms`, Decode=`7.17 TPS`, Efektif=`0.54 TPS`, Toplam=`3713.68 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`3210.77 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`138.60 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261003-210100-3d85` (2026-10-03T21:01:00.185667)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-03T21:01:00.185667`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `43.96 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 4927.5 | 8.93 | 0.56 | COLD (0) [227.7MB] | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 6325.7 | 0.78 | 0.24 | COLD (0) [227.9MB] | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL, prompt_cache=227.74 MB)
- **Performans**: TTFT=`4927.46 ms`, Decode=`8.93 TPS`, Efektif=`0.56 TPS`, Toplam=`5351.61 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`4834.61 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`130.16 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL, prompt_cache=227.93 MB)
- **Performans**: TTFT=`6325.66 ms`, Decode=`0.78 TPS`, Efektif=`0.24 TPS`, Toplam=`8290.60 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`6194.56 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`168.68 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261003-211225-bfee` (2026-10-03T21:12:25.857804)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-03T21:12:25.857804`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `260.46 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 5477.6 | 7.47 | 0.50 | COLD (0) [227.7MB] | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 3912.9 | 4.65 | 0.44 | COLD (0) [227.9MB] | PASS |
| `32k-turn1-cold` | 32768 | `q8_0` | 186 | 17 | 0 | 31080.5 | 5.11 | 0.47 | COLD (0) [232.8MB] | PASS |
| `32k-turn2-warm` | 32768 | `q8_0` | 237 | 21 | 203 | 1986.5 | 5.73 | 3.55 | HIT (203) [234.6MB] | PASS |
| `64k-turn1-analysis` | 65536 | `q4_0` | 189 | 43 | 0 | 29144.6 | 4.49 | 1.09 | COLD (0) [229.9MB] | PASS |
| `64k-turn2-cache-hit` | 65536 | `q4_0` | 261 | 23 | 232 | 1968.7 | 4.95 | 3.34 | HIT (232) [231.1MB] | PASS |
| `128k-turn1-eval` | 131072 | `q4_0` | 169 | 15 | 0 | 28570.8 | 4.79 | 0.44 | COLD (0) [229.3MB] | PASS |
| `128k-turn2-cache-hit` | 131072 | `q4_0` | 220 | 35 | 184 | 2342.3 | 5.49 | 3.89 | HIT (184) [230.4MB] | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL, prompt_cache=227.74 MB)
- **Performans**: TTFT=`5477.64 ms`, Decode=`7.47 TPS`, Efektif=`0.50 TPS`, Toplam=`5951.80 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`5389.80 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`130.88 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL, prompt_cache=227.93 MB)
- **Performans**: TTFT=`3912.94 ms`, Decode=`4.65 TPS`, Efektif=`0.44 TPS`, Toplam=`4540.44 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`3812.37 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`134.39 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `32k-turn1-cold` | Context: 32768 | Turn: 1
- **Kategori / İş Yükü**: `architecture` / `multi_turn_cache`
- **Context & KV Belleği**: `32768` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=186, Üretilen=17, Cache Reused=0 (COLD PREFILL, prompt_cache=232.77 MB)
- **Performans**: TTFT=`31080.51 ms`, Decode=`5.11 TPS`, Efektif=`0.47 TPS`, Toplam=`34397.63 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`30966.42 ms`, gpu_compute=`93.00 ms`, prefill_io=`0.00 ms`, checkpoint=`151.17 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are an expert AI software architect for high-performance low-latency MoE inference engines.<|im_end|> <|im_start|>user Context Document: Tendou launcher is a...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
NVIDIA RTX 5060 Laptop, 8 GB GDDR6<|im_end|>
```

#### [4] Görev: `32k-turn2-warm` | Context: 32768 | Turn: 2
- **Kategori / İş Yükü**: `architecture` / `multi_turn_cache`
- **Context & KV Belleği**: `32768` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=237, Üretilen=21, Cache Reused=203 (CACHE HIT (203 tokens reused), prompt_cache=234.59 MB)
- **Performans**: TTFT=`1986.53 ms`, Decode=`5.73 TPS`, Efektif=`3.55 TPS`, Toplam=`5638.02 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`1871.91 ms`, gpu_compute=`17.00 ms`, prefill_io=`0.00 ms`, checkpoint=`122.13 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are an expert AI software architect for high-performance low-latency MoE inference engines.<|im_end|> <|im_start|>user Context Document: Tendou launcher is a...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
P3 Asynchronous H2D Transfer Pipeline; it uses pinned staging buffers and CUDA event fences.<|im_end|>
```

#### [5] Görev: `64k-turn1-analysis` | Context: 65536 | Turn: 1
- **Kategori / İş Yükü**: `memory_scaling` / `large_context_cache`
- **Context & KV Belleği**: `65536` tokens (KV: `q4_0`)
- **Token Sayaçları**: Prompt=189, Üretilen=43, Cache Reused=0 (COLD PREFILL, prompt_cache=229.93 MB)
- **Performans**: TTFT=`29144.56 ms`, Decode=`4.49 TPS`, Efektif=`1.09 TPS`, Toplam=`38666.55 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`29036.74 ms`, gpu_compute=`94.50 ms`, prefill_io=`0.00 ms`, checkpoint=`146.19 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are an accurate technical documentation analyzer specializing in large language model serving.<|im_end|> <|im_start|>user Context Document: Section 64K-MoE: ...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
According to the text, the memory sizes are:

*   **fp16 KV cache:** 3.4 GB
*   **Q4_0 KV cache:** approximately 850 MB<|im_end|>
```

#### [6] Görev: `64k-turn2-cache-hit` | Context: 65536 | Turn: 2
- **Kategori / İş Yükü**: `memory_scaling` / `large_context_cache`
- **Context & KV Belleği**: `65536` tokens (KV: `q4_0`)
- **Token Sayaçları**: Prompt=261, Üretilen=23, Cache Reused=232 (CACHE HIT (232 tokens reused), prompt_cache=231.06 MB)
- **Performans**: TTFT=`1968.73 ms`, Decode=`4.95 TPS`, Efektif=`3.34 TPS`, Toplam=`6593.41 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`1830.37 ms`, gpu_compute=`14.50 ms`, prefill_io=`0.00 ms`, checkpoint=`128.12 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are an accurate technical documentation analyzer specializing in large language model serving.<|im_end|> <|im_start|>user Context Document: Section 64K-MoE: ...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
According to the text, prefix prompt caching guarantees that **repetitive context prefix evaluation is bypassed completely**.<|im_end|>
```

#### [7] Görev: `128k-turn1-eval` | Context: 131072 | Turn: 1
- **Kategori / İş Yükü**: `extreme_context` / `ultra_context_cache`
- **Context & KV Belleği**: `131072` tokens (KV: `q4_0`)
- **Token Sayaçları**: Prompt=169, Üretilen=15, Cache Reused=0 (COLD PREFILL, prompt_cache=229.31 MB)
- **Performans**: TTFT=`28570.79 ms`, Decode=`4.79 TPS`, Efektif=`0.44 TPS`, Toplam=`31708.75 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`28402.72 ms`, gpu_compute=`84.50 ms`, prefill_io=`0.00 ms`, checkpoint=`148.31 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a senior systems engineer specializing in extreme context MoE runtime benchmarks.<|im_end|> <|im_start|>user Context Document: Protocol 128K-Serving: The...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
The prompt checkpoint budget is locked at **1024 MB**.<|im_end|>
```

#### [8] Görev: `128k-turn2-cache-hit` | Context: 131072 | Turn: 2
- **Kategori / İş Yükü**: `extreme_context` / `ultra_context_cache`
- **Context & KV Belleği**: `131072` tokens (KV: `q4_0`)
- **Token Sayaçları**: Prompt=220, Üretilen=35, Cache Reused=184 (CACHE HIT (184 tokens reused), prompt_cache=230.42 MB)
- **Performans**: TTFT=`2342.26 ms`, Decode=`5.49 TPS`, Efektif=`3.89 TPS`, Toplam=`8739.96 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2116.47 ms`, gpu_compute=`18.00 ms`, prefill_io=`0.00 ms`, checkpoint=`138.46 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a senior systems engineer specializing in extreme context MoE runtime benchmarks.<|im_end|> <|im_start|>user Context Document: Protocol 128K-Serving: The...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
Protocol 128K-Serving targets **10+ TPS** for decode, and prefill reuse provides up to **100x TTFT reduction**.<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261003-212007-16c3` (2026-10-03T21:20:07.764526)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-03T21:20:07.764526`
- **Model**: `Qwen3.6-35B-A3B-UD-IQ4_XS`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `11.21 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 2 | 0 | 1408.5 | 0.00 | 0.66 | COLD (0) [127.6MB] | FAIL |
| `short-code-01` | 2048 | `q8_0` | 52 | 24 | 0 | 930.7 | 35.59 | 14.79 | COLD (0) [127.9MB] | FAIL |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=2, Cache Reused=0 (COLD PREFILL, prompt_cache=127.55 MB)
- **Performans**: TTFT=`1408.48 ms`, Decode=`0.00 TPS`, Efektif=`0.66 TPS`, Toplam=`1512.03 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`1368.52 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`63.54 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=24, Cache Reused=0 (COLD PREFILL, prompt_cache=127.91 MB)
- **Performans**: TTFT=`930.66 ms`, Decode=`35.59 TPS`, Efektif=`14.79 TPS`, Toplam=`1622.27 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`885.39 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`70.06 ms`
- **Bitiş Nedeni**: `length`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
0
  123456789012345678
```


---
## Benchmark Koşusu: `bench-20261003-212059-1f4a` (2026-10-03T21:20:59.599063)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-03T21:20:59.599063`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `28.64 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 4475.6 | 6.05 | 0.59 | COLD (0) [227.7MB] | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 3934.1 | 7.08 | 0.46 | COLD (0) [227.9MB] | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL, prompt_cache=227.74 MB)
- **Performans**: TTFT=`4475.55 ms`, Decode=`6.05 TPS`, Efektif=`0.59 TPS`, Toplam=`5050.01 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`4374.62 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`145.42 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL, prompt_cache=227.93 MB)
- **Performans**: TTFT=`3934.13 ms`, Decode=`7.08 TPS`, Efektif=`0.46 TPS`, Toplam=`4367.29 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`3854.42 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`121.06 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261003-212141-7562` (2026-10-03T21:21:41.028412)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-03T21:21:41.028412`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `28.46 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 4414.0 | 7.46 | 0.61 | COLD (0) [227.7MB] | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 3587.3 | 1.87 | 0.42 | COLD (0) [227.9MB] | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL, prompt_cache=227.74 MB)
- **Performans**: TTFT=`4414.00 ms`, Decode=`7.46 TPS`, Efektif=`0.61 TPS`, Toplam=`4905.18 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`4312.70 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`142.39 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL, prompt_cache=227.93 MB)
- **Performans**: TTFT=`3587.35 ms`, Decode=`1.87 TPS`, Efektif=`0.42 TPS`, Toplam=`4762.00 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`3494.90 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`128.56 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261003-212418-3a34` (2026-10-03T21:24:18.935539)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-03T21:24:18.935539`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `29.75 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 5111.7 | 5.77 | 0.53 | COLD (0) [227.7MB] | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 3346.9 | 6.14 | 0.53 | COLD (0) [227.9MB] | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL, prompt_cache=227.74 MB)
- **Performans**: TTFT=`5111.69 ms`, Decode=`5.77 TPS`, Efektif=`0.53 TPS`, Toplam=`5673.47 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`4956.44 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`205.86 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL, prompt_cache=227.93 MB)
- **Performans**: TTFT=`3346.90 ms`, Decode=`6.14 TPS`, Efektif=`0.53 TPS`, Toplam=`3757.96 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`3261.14 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`129.92 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261003-212612-c647` (2026-10-03T21:26:12.427807)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-03T21:26:12.427807`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `29.66 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 5028.6 | 7.90 | 0.55 | COLD (0) [227.7MB] | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 3721.3 | 5.92 | 0.49 | COLD (0) [227.9MB] | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL, prompt_cache=227.74 MB)
- **Performans**: TTFT=`5028.55 ms`, Decode=`7.90 TPS`, Efektif=`0.55 TPS`, Toplam=`5481.21 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`4932.94 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`139.10 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL, prompt_cache=227.93 MB)
- **Performans**: TTFT=`3721.33 ms`, Decode=`5.92 TPS`, Efektif=`0.49 TPS`, Toplam=`4113.91 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`3635.53 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`125.19 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261003-225741-e9d5` (2026-10-03T22:57:41.038227)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-03T22:57:41.038227`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=16, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `29.47 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 4496.9 | 7.09 | 0.60 | COLD (0) [227.7MB] | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 4948.3 | 3.78 | 0.36 | COLD (0) [227.9MB] | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL, prompt_cache=227.74 MB)
- **Performans**: TTFT=`4496.85 ms`, Decode=`7.09 TPS`, Efektif=`0.60 TPS`, Toplam=`5020.66 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`4383.81 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`157.05 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL, prompt_cache=227.93 MB)
- **Performans**: TTFT=`4948.26 ms`, Decode=`3.78 TPS`, Efektif=`0.36 TPS`, Toplam=`5481.88 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`4855.63 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`135.01 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261003-225915-dfde` (2026-10-03T22:59:15.655844)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-03T22:59:15.655844`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=16, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `55.23 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `32k-turn1-cold` | 32768 | `q8_0` | 186 | 17 | 0 | 25776.1 | 4.13 | 0.54 | COLD (0) [232.8MB] | PASS |
| `32k-turn2-warm` | 32768 | `q8_0` | 237 | 21 | 203 | 2008.7 | 5.92 | 3.59 | HIT (203) [234.6MB] | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `32k-turn1-cold` | Context: 32768 | Turn: 1
- **Kategori / İş Yükü**: `architecture` / `multi_turn_cache`
- **Context & KV Belleği**: `32768` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=186, Üretilen=17, Cache Reused=0 (COLD PREFILL, prompt_cache=232.77 MB)
- **Performans**: TTFT=`25776.09 ms`, Decode=`4.13 TPS`, Efektif=`0.54 TPS`, Toplam=`29819.97 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`25646.04 ms`, gpu_compute=`93.00 ms`, prefill_io=`0.00 ms`, checkpoint=`167.53 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are an expert AI software architect for high-performance low-latency MoE inference engines.<|im_end|> <|im_start|>user Context Document: Tendou launcher is a...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
NVIDIA RTX 5060 Laptop, 8 GB GDDR6<|im_end|>
```

#### [2] Görev: `32k-turn2-warm` | Context: 32768 | Turn: 2
- **Kategori / İş Yükü**: `architecture` / `multi_turn_cache`
- **Context & KV Belleği**: `32768` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=237, Üretilen=21, Cache Reused=203 (CACHE HIT (203 tokens reused), prompt_cache=234.59 MB)
- **Performans**: TTFT=`2008.66 ms`, Decode=`5.92 TPS`, Efektif=`3.59 TPS`, Toplam=`5569.62 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`1852.17 ms`, gpu_compute=`17.00 ms`, prefill_io=`0.00 ms`, checkpoint=`146.09 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are an expert AI software architect for high-performance low-latency MoE inference engines.<|im_end|> <|im_start|>user Context Document: Tendou launcher is a...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
P3 Asynchronous H2D Transfer Pipeline; it uses pinned staging buffers and CUDA event fences.<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261004-112411-2bbe` (2026-10-04T11:24:11.830616)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-04T11:24:11.830616`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `41.82 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 4749.1 | 5.47 | 0.56 | COLD (0) [227.7MB] | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 5148.8 | 3.03 | 0.34 | COLD (0) [227.9MB] | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 13 | 0 | 10923.2 | 7.31 | 0.95 | COLD (0) [229.4MB] | FAIL |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL, prompt_cache=227.74 MB)
- **Performans**: TTFT=`4749.06 ms`, Decode=`5.47 TPS`, Efektif=`0.56 TPS`, Toplam=`5365.44 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`4651.88 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`142.44 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL, prompt_cache=227.93 MB)
- **Performans**: TTFT=`5148.75 ms`, Decode=`3.03 TPS`, Efektif=`0.34 TPS`, Toplam=`5822.95 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`5056.70 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`139.38 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=13, Cache Reused=0 (COLD PREFILL, prompt_cache=229.37 MB)
- **Performans**: TTFT=`10923.22 ms`, Decode=`7.31 TPS`, Efektif=`0.95 TPS`, Toplam=`12635.89 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`10620.59 ms`, gpu_compute=`44.50 ms`, prefill_io=`0.00 ms`, checkpoint=`300.67 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
337

17 karesi 19<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261004-112454-8972` (2026-10-04T11:24:54.518231)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-04T11:24:54.518231`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `47.36 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 5800.1 | 5.78 | 0.47 | COLD (0) [227.7MB] | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 4505.9 | 1.91 | 0.34 | COLD (0) [227.9MB] | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 13 | 0 | 7290.6 | 6.98 | 1.32 | COLD (0) [229.4MB] | FAIL |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL, prompt_cache=227.74 MB)
- **Performans**: TTFT=`5800.08 ms`, Decode=`5.78 TPS`, Efektif=`0.47 TPS`, Toplam=`6375.06 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`5707.04 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`128.33 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL, prompt_cache=227.93 MB)
- **Performans**: TTFT=`4505.87 ms`, Decode=`1.91 TPS`, Efektif=`0.34 TPS`, Toplam=`5823.94 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`4421.63 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`115.85 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=13, Cache Reused=0 (COLD PREFILL, prompt_cache=229.37 MB)
- **Performans**: TTFT=`7290.55 ms`, Decode=`6.98 TPS`, Efektif=`1.32 TPS`, Toplam=`9062.12 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`7199.24 ms`, gpu_compute=`44.50 ms`, prefill_io=`0.00 ms`, checkpoint=`119.81 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
337

17 karesi 19<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261004-112542-768b` (2026-10-04T11:25:42.814339)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-04T11:25:42.814339`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `42.51 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 4638.0 | 6.07 | 0.54 | COLD (0) [227.7MB] | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 2944.3 | 5.71 | 0.59 | COLD (0) [227.9MB] | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 13 | 0 | 7884.8 | 6.85 | 1.24 | COLD (0) [229.4MB] | FAIL |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL, prompt_cache=227.74 MB)
- **Performans**: TTFT=`4637.96 ms`, Decode=`6.07 TPS`, Efektif=`0.54 TPS`, Toplam=`5515.80 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`4544.06 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`463.12 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL, prompt_cache=227.93 MB)
- **Performans**: TTFT=`2944.32 ms`, Decode=`5.71 TPS`, Efektif=`0.59 TPS`, Toplam=`3373.63 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2861.47 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`125.10 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=13, Cache Reused=0 (COLD PREFILL, prompt_cache=229.37 MB)
- **Performans**: TTFT=`7884.82 ms`, Decode=`6.85 TPS`, Efektif=`1.24 TPS`, Toplam=`9688.38 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`7719.43 ms`, gpu_compute=`44.50 ms`, prefill_io=`0.00 ms`, checkpoint=`179.44 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
337

17 karesi 19<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261004-112626-2e4f` (2026-10-04T11:26:26.207886)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-04T11:26:26.207886`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `46.14 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 4840.0 | 6.63 | 0.56 | COLD (0) [227.7MB] | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 3667.5 | 7.37 | 0.49 | COLD (0) [227.9MB] | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 13 | 0 | 9492.1 | 5.31 | 1.02 | COLD (0) [229.4MB] | FAIL |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL, prompt_cache=227.74 MB)
- **Performans**: TTFT=`4840.04 ms`, Decode=`6.63 TPS`, Efektif=`0.56 TPS`, Toplam=`5337.78 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`4748.55 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`128.38 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL, prompt_cache=227.93 MB)
- **Performans**: TTFT=`3667.54 ms`, Decode=`7.37 TPS`, Efektif=`0.49 TPS`, Toplam=`4089.77 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`3584.32 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`121.55 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=13, Cache Reused=0 (COLD PREFILL, prompt_cache=229.37 MB)
- **Performans**: TTFT=`9492.08 ms`, Decode=`5.31 TPS`, Efektif=`1.02 TPS`, Toplam=`11788.71 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`9396.41 ms`, gpu_compute=`44.50 ms`, prefill_io=`0.00 ms`, checkpoint=`123.43 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
337

17 karesi 19<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261004-112713-764b` (2026-10-04T11:27:13.242300)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-04T11:27:13.242300`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `46.26 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 5470.6 | 5.83 | 0.50 | COLD (0) [227.7MB] | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 3493.7 | 6.72 | 0.51 | COLD (0) [227.9MB] | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 13 | 0 | 6666.8 | 3.43 | 1.19 | COLD (0) [229.4MB] | FAIL |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL, prompt_cache=227.74 MB)
- **Performans**: TTFT=`5470.62 ms`, Decode=`5.83 TPS`, Efektif=`0.50 TPS`, Toplam=`6023.69 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`5373.95 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`131.45 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL, prompt_cache=227.93 MB)
- **Performans**: TTFT=`3493.66 ms`, Decode=`6.72 TPS`, Efektif=`0.51 TPS`, Toplam=`3903.19 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`3394.51 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`135.20 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=13, Cache Reused=0 (COLD PREFILL, prompt_cache=229.37 MB)
- **Performans**: TTFT=`6666.77 ms`, Decode=`3.43 TPS`, Efektif=`1.19 TPS`, Toplam=`10106.97 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`6563.34 ms`, gpu_compute=`44.50 ms`, prefill_io=`0.00 ms`, checkpoint=`133.28 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
337

17 karesi 19<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261004-112800-6880` (2026-10-04T11:28:00.506500)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-04T11:28:00.506500`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `47.56 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 4932.3 | 6.82 | 0.51 | COLD (0) [227.7MB] | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 3288.1 | 6.77 | 0.54 | COLD (0) [227.9MB] | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 13 | 0 | 10293.0 | 3.21 | 0.86 | COLD (0) [229.4MB] | FAIL |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL, prompt_cache=227.74 MB)
- **Performans**: TTFT=`4932.26 ms`, Decode=`6.82 TPS`, Efektif=`0.51 TPS`, Toplam=`5923.41 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`4830.26 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`557.21 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL, prompt_cache=227.93 MB)
- **Performans**: TTFT=`3288.10 ms`, Decode=`6.77 TPS`, Efektif=`0.54 TPS`, Toplam=`3727.33 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`3209.08 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`118.68 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=13, Cache Reused=0 (COLD PREFILL, prompt_cache=229.37 MB)
- **Performans**: TTFT=`10293.02 ms`, Decode=`3.21 TPS`, Efektif=`0.86 TPS`, Toplam=`13911.26 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`10142.20 ms`, gpu_compute=`44.50 ms`, prefill_io=`0.00 ms`, checkpoint=`163.02 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
337

17 karesi 19<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261004-112848-2782` (2026-10-04T11:28:48.797986)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-04T11:28:48.797986`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `44.76 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 4765.1 | 7.29 | 0.57 | COLD (0) [227.7MB] | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 2963.9 | 8.30 | 0.60 | COLD (0) [227.9MB] | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 13 | 0 | 7621.4 | 7.59 | 1.29 | COLD (0) [229.4MB] | FAIL |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL, prompt_cache=227.74 MB)
- **Performans**: TTFT=`4765.07 ms`, Decode=`7.29 TPS`, Efektif=`0.57 TPS`, Toplam=`5242.64 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`4664.89 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`138.52 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL, prompt_cache=227.93 MB)
- **Performans**: TTFT=`2963.92 ms`, Decode=`8.30 TPS`, Efektif=`0.60 TPS`, Toplam=`3323.31 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2881.26 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`115.79 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=13, Cache Reused=0 (COLD PREFILL, prompt_cache=229.37 MB)
- **Performans**: TTFT=`7621.43 ms`, Decode=`7.59 TPS`, Efektif=`1.29 TPS`, Toplam=`9267.80 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`7496.29 ms`, gpu_compute=`44.50 ms`, prefill_io=`0.00 ms`, checkpoint=`143.36 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
337

17 karesi 19<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261004-112934-f684` (2026-10-04T11:29:34.477085)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-04T11:29:34.477085`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `47.90 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 5020.4 | 7.20 | 0.54 | COLD (0) [227.7MB] | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 4307.1 | 7.16 | 0.42 | COLD (0) [227.9MB] | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 13 | 0 | 9508.1 | 7.71 | 1.08 | COLD (0) [229.4MB] | FAIL |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL, prompt_cache=227.74 MB)
- **Performans**: TTFT=`5020.39 ms`, Decode=`7.20 TPS`, Efektif=`0.54 TPS`, Toplam=`5519.82 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`4925.62 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`136.03 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL, prompt_cache=227.93 MB)
- **Performans**: TTFT=`4307.08 ms`, Decode=`7.16 TPS`, Efektif=`0.42 TPS`, Toplam=`4713.97 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`4229.33 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`114.39 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=13, Cache Reused=0 (COLD PREFILL, prompt_cache=229.37 MB)
- **Performans**: TTFT=`9508.12 ms`, Decode=`7.71 TPS`, Efektif=`1.08 TPS`, Toplam=`11124.44 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`9309.93 ms`, gpu_compute=`44.50 ms`, prefill_io=`0.00 ms`, checkpoint=`212.94 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
337

17 karesi 19<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261004-113023-8195` (2026-10-04T11:30:23.217189)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-04T11:30:23.217189`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `45.95 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 4433.0 | 7.11 | 0.61 | COLD (0) [227.7MB] | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 3134.3 | 7.26 | 0.57 | COLD (0) [227.9MB] | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 13 | 0 | 8663.2 | 7.86 | 1.17 | COLD (0) [229.4MB] | FAIL |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL, prompt_cache=227.74 MB)
- **Performans**: TTFT=`4432.98 ms`, Decode=`7.11 TPS`, Efektif=`0.61 TPS`, Toplam=`4935.12 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`4332.96 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`138.22 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL, prompt_cache=227.93 MB)
- **Performans**: TTFT=`3134.34 ms`, Decode=`7.26 TPS`, Efektif=`0.57 TPS`, Toplam=`3519.44 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`3049.63 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`122.41 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=13, Cache Reused=0 (COLD PREFILL, prompt_cache=229.37 MB)
- **Performans**: TTFT=`8663.25 ms`, Decode=`7.86 TPS`, Efektif=`1.17 TPS`, Toplam=`10276.75 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`8521.03 ms`, gpu_compute=`44.50 ms`, prefill_io=`0.00 ms`, checkpoint=`184.14 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
337

17 karesi 19<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261004-113110-9e1e` (2026-10-04T11:31:10.083830)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-04T11:31:10.083830`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `44.53 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 4288.7 | 7.60 | 0.63 | COLD (0) [227.7MB] | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 2531.5 | 6.35 | 0.68 | COLD (0) [227.9MB] | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 13 | 0 | 8853.6 | 5.31 | 1.08 | COLD (0) [229.4MB] | FAIL |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL, prompt_cache=227.74 MB)
- **Performans**: TTFT=`4288.74 ms`, Decode=`7.60 TPS`, Efektif=`0.63 TPS`, Toplam=`4756.23 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`4197.13 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`137.69 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL, prompt_cache=227.93 MB)
- **Performans**: TTFT=`2531.53 ms`, Decode=`6.35 TPS`, Efektif=`0.68 TPS`, Toplam=`2939.02 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2447.92 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`122.16 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=13, Cache Reused=0 (COLD PREFILL, prompt_cache=229.37 MB)
- **Performans**: TTFT=`8853.61 ms`, Decode=`5.31 TPS`, Efektif=`1.08 TPS`, Toplam=`11130.03 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`8721.43 ms`, gpu_compute=`44.50 ms`, prefill_io=`0.00 ms`, checkpoint=`153.74 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
337

17 karesi 19<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261004-200505-4c34` (2026-10-04T20:05:05.720278)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-04T20:05:05.720278`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=16, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `37.18 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 6766.5 | 1.38 | 0.35 | COLD (0) [227.7MB] | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 5544.9 | 8.90 | 0.33 | COLD (0) [227.9MB] | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 13 | 0 | 7164.6 | 8.10 | 1.37 | COLD (0) [229.4MB] | FAIL |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL, prompt_cache=227.74 MB)
- **Performans**: TTFT=`6766.49 ms`, Decode=`1.38 TPS`, Efektif=`0.35 TPS`, Toplam=`8575.04 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`6615.10 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`163.16 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL, prompt_cache=227.93 MB)
- **Performans**: TTFT=`5544.95 ms`, Decode=`8.90 TPS`, Efektif=`0.33 TPS`, Toplam=`6036.02 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`5358.36 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`199.41 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=13, Cache Reused=0 (COLD PREFILL, prompt_cache=229.37 MB)
- **Performans**: TTFT=`7164.58 ms`, Decode=`8.10 TPS`, Efektif=`1.37 TPS`, Toplam=`8762.76 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`7016.98 ms`, gpu_compute=`44.50 ms`, prefill_io=`0.00 ms`, checkpoint=`160.22 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
337

17 karesi 19<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261004-201054-8540` (2026-10-04T20:10:54.893761)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-04T20:10:54.893761`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=16, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `43.23 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 6493.1 | 3.73 | 0.40 | COLD (0) [227.7MB] | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 7374.0 | 1.33 | 0.24 | COLD (0) [227.9MB] | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 13 | 0 | 8449.0 | 3.96 | 1.05 | COLD (0) [229.4MB] | FAIL |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL, prompt_cache=227.74 MB)
- **Performans**: TTFT=`6493.07 ms`, Decode=`3.73 TPS`, Efektif=`0.40 TPS`, Toplam=`7423.82 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`6329.52 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`143.60 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL, prompt_cache=227.93 MB)
- **Performans**: TTFT=`7374.04 ms`, Decode=`1.33 TPS`, Efektif=`0.24 TPS`, Toplam=`8506.41 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`7240.83 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`145.73 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=13, Cache Reused=0 (COLD PREFILL, prompt_cache=229.37 MB)
- **Performans**: TTFT=`8449.00 ms`, Decode=`3.96 TPS`, Efektif=`1.05 TPS`, Toplam=`11466.84 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`8149.23 ms`, gpu_compute=`44.50 ms`, prefill_io=`0.00 ms`, checkpoint=`183.54 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
337

17 karesi 19<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261004-201202-d86b` (2026-10-04T20:12:02.467631)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-04T20:12:02.467631`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=16, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `34.24 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 5078.4 | 3.51 | 0.49 | COLD (0) [227.7MB] | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 3747.1 | 2.97 | 0.42 | COLD (0) [227.9MB] | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 13 | 0 | 7438.7 | 7.85 | 1.32 | COLD (0) [229.4MB] | FAIL |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL, prompt_cache=227.74 MB)
- **Performans**: TTFT=`5078.40 ms`, Decode=`3.51 TPS`, Efektif=`0.49 TPS`, Toplam=`6078.45 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`4976.38 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`132.51 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL, prompt_cache=227.93 MB)
- **Performans**: TTFT=`3747.11 ms`, Decode=`2.97 TPS`, Efektif=`0.42 TPS`, Toplam=`4805.17 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`3632.76 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`151.32 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=13, Cache Reused=0 (COLD PREFILL, prompt_cache=229.37 MB)
- **Performans**: TTFT=`7438.71 ms`, Decode=`7.85 TPS`, Efektif=`1.32 TPS`, Toplam=`9089.03 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`7006.96 ms`, gpu_compute=`44.50 ms`, prefill_io=`0.00 ms`, checkpoint=`160.97 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
337

17 karesi 19<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261004-210549-a216` (2026-10-04T21:05:49.768052)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-04T21:05:49.768052`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=16, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `45.14 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 5614.9 | 2.15 | 0.43 | COLD (0) [227.7MB] | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 3205.5 | 5.57 | 0.54 | COLD (0) [227.9MB] | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 13 | 0 | 9220.2 | 4.13 | 0.96 | COLD (0) [229.4MB] | FAIL |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL, prompt_cache=227.74 MB)
- **Performans**: TTFT=`5614.93 ms`, Decode=`2.15 TPS`, Efektif=`0.43 TPS`, Toplam=`6981.09 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`5423.93 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`191.32 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL, prompt_cache=227.93 MB)
- **Performans**: TTFT=`3205.54 ms`, Decode=`5.57 TPS`, Efektif=`0.54 TPS`, Toplam=`3721.39 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`3074.43 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`146.58 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=13, Cache Reused=0 (COLD PREFILL, prompt_cache=229.37 MB)
- **Performans**: TTFT=`9220.17 ms`, Decode=`4.13 TPS`, Efektif=`0.96 TPS`, Toplam=`12448.44 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`8998.03 ms`, gpu_compute=`44.50 ms`, prefill_io=`0.00 ms`, checkpoint=`552.08 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
337

17 karesi 19<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261004-213522-c05f` (2026-10-04T21:35:22.462043)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-04T21:35:22.462043`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `40.59 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 5302.8 | 6.97 | 0.52 | COLD (0) [227.7MB] | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 4073.3 | 5.30 | 0.44 | COLD (0) [227.9MB] | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 13 | 0 | 7899.6 | 8.29 | 1.27 | COLD (0) [229.4MB] | FAIL |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL, prompt_cache=227.74 MB)
- **Performans**: TTFT=`5302.82 ms`, Decode=`6.97 TPS`, Efektif=`0.52 TPS`, Toplam=`5815.99 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`5168.53 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`147.42 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL, prompt_cache=227.93 MB)
- **Performans**: TTFT=`4073.32 ms`, Decode=`5.30 TPS`, Efektif=`0.44 TPS`, Toplam=`4550.25 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`3968.99 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`128.38 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=13, Cache Reused=0 (COLD PREFILL, prompt_cache=229.37 MB)
- **Performans**: TTFT=`7899.62 ms`, Decode=`8.29 TPS`, Efektif=`1.27 TPS`, Toplam=`9433.56 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`7721.56 ms`, gpu_compute=`44.50 ms`, prefill_io=`0.00 ms`, checkpoint=`177.35 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
337

17 karesi 19<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261004-213611-f862` (2026-10-04T21:36:11.339100)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-04T21:36:11.339100`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=16, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `50.10 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 5259.6 | 6.98 | 0.52 | COLD (0) [227.7MB] | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 3996.0 | 6.10 | 0.45 | COLD (0) [227.9MB] | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 13 | 0 | 9488.1 | 3.32 | 0.92 | COLD (0) [229.4MB] | FAIL |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL, prompt_cache=227.74 MB)
- **Performans**: TTFT=`5259.55 ms`, Decode=`6.98 TPS`, Efektif=`0.52 TPS`, Toplam=`5787.01 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`5120.35 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`148.44 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL, prompt_cache=227.93 MB)
- **Performans**: TTFT=`3996.04 ms`, Decode=`6.10 TPS`, Efektif=`0.45 TPS`, Toplam=`4459.15 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`3894.72 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`146.35 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=13, Cache Reused=0 (COLD PREFILL, prompt_cache=229.37 MB)
- **Performans**: TTFT=`9488.09 ms`, Decode=`3.32 TPS`, Efektif=`0.92 TPS`, Toplam=`13005.86 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`9285.13 ms`, gpu_compute=`44.50 ms`, prefill_io=`0.00 ms`, checkpoint=`217.54 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
337

17 karesi 19<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261004-214912-8c5b` (2026-10-04T21:49:12.511035)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-04T21:49:12.511035`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `54.55 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 11568.1 | 5.74 | 0.25 | COLD (0) [227.7MB] | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 4440.6 | 6.49 | 0.41 | COLD (0) [227.9MB] | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 13 | 0 | 9303.2 | 8.40 | 1.11 | COLD (0) [229.4MB] | FAIL |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL, prompt_cache=227.74 MB)
- **Performans**: TTFT=`11568.11 ms`, Decode=`5.74 TPS`, Efektif=`0.25 TPS`, Toplam=`12201.51 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`11398.61 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`140.09 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL, prompt_cache=227.93 MB)
- **Performans**: TTFT=`4440.61 ms`, Decode=`6.49 TPS`, Efektif=`0.41 TPS`, Toplam=`4912.78 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`4354.32 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`121.13 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=13, Cache Reused=0 (COLD PREFILL, prompt_cache=229.37 MB)
- **Performans**: TTFT=`9303.20 ms`, Decode=`8.40 TPS`, Efektif=`1.11 TPS`, Toplam=`10818.39 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`9068.04 ms`, gpu_compute=`44.50 ms`, prefill_io=`0.00 ms`, checkpoint=`264.22 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
337

17 karesi 19<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261004-215013-9fbd` (2026-10-04T21:50:13.191706)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-04T21:50:13.191706`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=16, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `48.25 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 5142.2 | 7.14 | 0.53 | COLD (0) [227.7MB] | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 3814.5 | 6.03 | 0.47 | COLD (0) [227.9MB] | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 13 | 0 | 9474.8 | 3.09 | 0.91 | COLD (0) [229.4MB] | FAIL |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL, prompt_cache=227.74 MB)
- **Performans**: TTFT=`5142.19 ms`, Decode=`7.14 TPS`, Efektif=`0.53 TPS`, Toplam=`5667.92 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`4997.43 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`159.54 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL, prompt_cache=227.93 MB)
- **Performans**: TTFT=`3814.49 ms`, Decode=`6.03 TPS`, Efektif=`0.47 TPS`, Toplam=`4277.39 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`3698.35 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`157.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=13, Cache Reused=0 (COLD PREFILL, prompt_cache=229.37 MB)
- **Performans**: TTFT=`9474.84 ms`, Decode=`3.09 TPS`, Efektif=`0.91 TPS`, Toplam=`13245.53 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`9303.82 ms`, gpu_compute=`44.50 ms`, prefill_io=`0.00 ms`, checkpoint=`208.80 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
337

17 karesi 19<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261004-220326-19c1` (2026-10-04T22:03:26.872198)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-04T22:03:26.872198`
- **Model**: `Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `48.14 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 5282.6 | 5.87 | 0.51 | COLD (0) [227.7MB] | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 5073.8 | 3.83 | 0.35 | COLD (0) [227.9MB] | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 13 | 0 | 7511.4 | 8.24 | 1.33 | COLD (0) [229.4MB] | FAIL |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL, prompt_cache=227.74 MB)
- **Performans**: TTFT=`5282.64 ms`, Decode=`5.87 TPS`, Efektif=`0.51 TPS`, Toplam=`5900.34 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`5152.97 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`132.22 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL, prompt_cache=227.93 MB)
- **Performans**: TTFT=`5073.78 ms`, Decode=`3.83 TPS`, Efektif=`0.35 TPS`, Toplam=`5675.31 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`4994.56 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`116.33 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=13, Cache Reused=0 (COLD PREFILL, prompt_cache=229.37 MB)
- **Performans**: TTFT=`7511.43 ms`, Decode=`8.24 TPS`, Efektif=`1.33 TPS`, Toplam=`9051.21 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`7364.45 ms`, gpu_compute=`44.50 ms`, prefill_io=`0.00 ms`, checkpoint=`176.81 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
337

17 karesi 19<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261015-000812-7257` (2026-10-15T00:08:12.566190)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-15T00:08:12.566190`
- **Model**: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `27.43 s`
- **Notlar**: Benchmark on new Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS model

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 12 | 0 | 3291.2 | 8.00 | 2.32 | COLD (0) [227.9MB] | FAIL |
| `short-code-01` | 2048 | `q8_0` | 52 | 24 | 0 | 2885.9 | 11.63 | 4.79 | COLD (0) [228.3MB] | FAIL |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 33 | 0 | 4496.9 | 14.12 | 4.65 | COLD (0) [229.7MB] | FAIL |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=12, Cache Reused=0 (COLD PREFILL, prompt_cache=227.88 MB)
- **Performans**: TTFT=`3291.16 ms`, Decode=`8.00 TPS`, Efektif=`2.32 TPS`, Toplam=`4748.86 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`3166.82 ms`, gpu_compute=`11.50 ms`, prefill_io=`0.00 ms`, checkpoint=`128.64 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
I asked to compute 17×19.<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=24, Cache Reused=0 (COLD PREFILL, prompt_cache=228.29 MB)
- **Performans**: TTFT=`2885.86 ms`, Decode=`11.63 TPS`, Efektif=`4.79 TPS`, Toplam=`5007.86 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2825.81 ms`, gpu_compute=`13.00 ms`, prefill_io=`0.00 ms`, checkpoint=`98.35 ms`
- **Bitiş Nedeni**: `length`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
## Python function call result

Python function call `sum([12, 14, 16])`
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=33, Cache Reused=0 (COLD PREFILL, prompt_cache=229.71 MB)
- **Performans**: TTFT=`4496.89 ms`, Decode=`14.12 TPS`, Efektif=`4.65 TPS`, Toplam=`6879.27 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`4415.52 ms`, gpu_compute=`22.25 ms`, prefill_io=`0.00 ms`, checkpoint=`107.03 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
Sayılın carpımını hesapla: 17 × 19 = 333.

Sayi yaz: 333.<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261015-074014-6d6c` (2026-10-15T07:40:14.411227)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-15T07:40:14.411227`
- **Model**: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=16, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `36.33 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 16 | 0 | 3060.0 | 8.01 | 2.97 | COLD (0) [227.9MB] | FAIL |
| `short-code-01` | 2048 | `q8_0` | 52 | 30 | 0 | 2444.6 | 11.42 | 5.66 | COLD (0) [228.4MB] | FAIL |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 128 | 0 | 4171.7 | 11.39 | 8.28 | COLD (0) [231.3MB] | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=16, Cache Reused=0 (COLD PREFILL, prompt_cache=227.94 MB)
- **Performans**: TTFT=`3060.03 ms`, Decode=`8.01 TPS`, Efektif=`2.97 TPS`, Toplam=`5047.34 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2925.78 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`151.55 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
17 ile 19 sayılarının çarpımını hesapla<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=30, Cache Reused=0 (COLD PREFILL, prompt_cache=228.39 MB)
- **Performans**: TTFT=`2444.63 ms`, Decode=`11.42 TPS`, Efektif=`5.66 TPS`, Toplam=`5121.50 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2389.35 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`104.90 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=128, Cache Reused=0 (COLD PREFILL, prompt_cache=231.34 MB)
- **Performans**: TTFT=`4171.66 ms`, Decode=`11.39 TPS`, Efektif=`8.28 TPS`, Toplam=`15452.95 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`4072.26 ms`, gpu_compute=`44.50 ms`, prefill_io=`0.00 ms`, checkpoint=`138.33 ms`
- **Bitiş Nedeni**: `length`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
I'll assume you mean "17 and 19" (17 ile 19) rather than "1719" (1719). Let me calculate step by step.

**Step 1: Calculate 17 × 19**

17 × 19 = 17 × (20 - 1) = 17 × 20 - 17 × 1 = 340 - 17 = **323**

**Step 2: Calculate 1719 × 1719**

17
```


---
## Benchmark Koşusu: `bench-20261015-074131-03e5` (2026-10-15T07:41:31.094204)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-15T07:41:31.094204`
- **Model**: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=16, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `29.00 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 16 | 0 | 2745.0 | 9.36 | 3.37 | COLD (0) [227.9MB] | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 33 | 0 | 2315.8 | 11.19 | 6.04 | COLD (0) [228.4MB] | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 19 | 0 | 4010.3 | 11.66 | 3.18 | COLD (0) [229.5MB] | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=16, Cache Reused=0 (COLD PREFILL, prompt_cache=227.94 MB)
- **Performans**: TTFT=`2745.00 ms`, Decode=`9.36 TPS`, Efektif=`3.37 TPS`, Toplam=`4449.48 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2651.49 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`139.61 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
````text
```
17 × 19 = 323
```<|im_end|>
````

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=33, Cache Reused=0 (COLD PREFILL, prompt_cache=228.44 MB)
- **Performans**: TTFT=`2315.77 ms`, Decode=`11.19 TPS`, Efektif=`6.04 TPS`, Toplam=`5297.54 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2253.15 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`103.75 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
````text
**Python'da `sum([12, 14, 16])` ifadesinin sonucu:**

```python
42
```<|im_end|>
````

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=19, Cache Reused=0 (COLD PREFILL, prompt_cache=229.47 MB)
- **Performans**: TTFT=`4010.34 ms`, Decode=`11.66 TPS`, Efektif=`3.18 TPS`, Toplam=`5663.51 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`3887.19 ms`, gpu_compute=`44.50 ms`, prefill_io=`0.00 ms`, checkpoint=`137.46 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
17 ile 19 sayilarinin carpimi 323'tür.<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261015-075042-36ac` (2026-10-15T07:50:42.794282)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-15T07:50:42.794282`
- **Model**: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=16, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `28.43 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 16 | 0 | 2417.1 | 9.27 | 3.63 | COLD (0) [227.9MB] | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 33 | 0 | 2257.7 | 11.36 | 6.14 | COLD (0) [228.4MB] | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 19 | 0 | 4030.3 | 12.24 | 3.21 | COLD (0) [229.5MB] | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=16, Cache Reused=0 (COLD PREFILL, prompt_cache=227.94 MB)
- **Performans**: TTFT=`2417.09 ms`, Decode=`9.27 TPS`, Efektif=`3.63 TPS`, Toplam=`4129.47 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2322.60 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`136.54 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
````text
```
17 × 19 = 323
```<|im_end|>
````

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=33, Cache Reused=0 (COLD PREFILL, prompt_cache=228.44 MB)
- **Performans**: TTFT=`2257.71 ms`, Decode=`11.36 TPS`, Efektif=`6.14 TPS`, Toplam=`5207.64 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2199.79 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`103.88 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
````text
**Python'da `sum([12, 14, 16])` ifadesinin sonucu:**

```python
42
```<|im_end|>
````

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=19, Cache Reused=0 (COLD PREFILL, prompt_cache=229.47 MB)
- **Performans**: TTFT=`4030.26 ms`, Decode=`12.24 TPS`, Efektif=`3.21 TPS`, Toplam=`5613.36 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`3929.52 ms`, gpu_compute=`44.50 ms`, prefill_io=`0.00 ms`, checkpoint=`137.46 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
17 ile 19 sayilarinin carpimi 323'tür.<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261015-080106-c3ab` (2026-10-15T08:01:06.067229)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-15T08:01:06.067229`
- **Model**: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=16, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `29.33 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 16 | 0 | 2379.4 | 8.68 | 3.58 | COLD (0) [227.9MB] | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 33 | 0 | 2247.8 | 10.54 | 5.92 | COLD (0) [228.4MB] | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 19 | 0 | 4044.9 | 12.30 | 3.20 | COLD (0) [229.5MB] | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=16, Cache Reused=0 (COLD PREFILL, prompt_cache=227.94 MB)
- **Performans**: TTFT=`2379.37 ms`, Decode=`8.68 TPS`, Efektif=`3.58 TPS`, Toplam=`4189.99 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2284.22 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`141.86 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
````text
```
17 × 19 = 323
```<|im_end|>
````

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=33, Cache Reused=0 (COLD PREFILL, prompt_cache=228.44 MB)
- **Performans**: TTFT=`2247.78 ms`, Decode=`10.54 TPS`, Efektif=`5.92 TPS`, Toplam=`5405.93 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2190.02 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`98.75 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
````text
**Python'da `sum([12, 14, 16])` ifadesinin sonucu:**

```python
42
```<|im_end|>
````

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=19, Cache Reused=0 (COLD PREFILL, prompt_cache=229.47 MB)
- **Performans**: TTFT=`4044.92 ms`, Decode=`12.30 TPS`, Efektif=`3.20 TPS`, Toplam=`5627.01 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`3944.93 ms`, gpu_compute=`44.50 ms`, prefill_io=`0.00 ms`, checkpoint=`139.42 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
17 ile 19 sayilarinin carpimi 323'tür.<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261015-112007-8870` (2026-10-15T11:20:07.614885)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-15T11:20:07.614885`
- **Model**: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `25.73 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 5 | 0 | 2421.5 | 7.81 | 1.32 | COLD (0) [227.8MB] | FAIL |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 2412.5 | 6.83 | 0.72 | COLD (0) [227.9MB] | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 10 | 0 | 4163.3 | 7.70 | 1.66 | COLD (0) [229.3MB] | FAIL |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=5, Cache Reused=0 (COLD PREFILL, prompt_cache=227.76 MB)
- **Performans**: TTFT=`2421.50 ms`, Decode=`7.81 TPS`, Efektif=`1.32 TPS`, Toplam=`3019.46 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2342.50 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`118.27 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
1719<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL, prompt_cache=227.93 MB)
- **Performans**: TTFT=`2412.53 ms`, Decode=`6.83 TPS`, Efektif=`0.72 TPS`, Toplam=`2792.46 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2327.51 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`116.58 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=10, Cache Reused=0 (COLD PREFILL, prompt_cache=229.32 MB)
- **Performans**: TTFT=`4163.26 ms`, Decode=`7.70 TPS`, Efektif=`1.66 TPS`, Toplam=`5425.34 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`4076.27 ms`, gpu_compute=`44.50 ms`, prefill_io=`0.00 ms`, checkpoint=`124.50 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
I'm unable to assist with this request.<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261015-153608-fde8` (2026-10-15T15:36:08.842709)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-15T15:36:08.842709`
- **Model**: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `36.59 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 3 | 0 | 2647.0 | 5.70 | 0.66 | COLD (0) [227.7MB] | FAIL |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 2525.0 | 6.93 | 0.69 | COLD (0) [227.9MB] | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 128 | 0 | 4144.5 | 12.35 | 8.84 | COLD (0) [231.3MB] | FAIL |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=3, Cache Reused=0 (COLD PREFILL, prompt_cache=227.70 MB)
- **Performans**: TTFT=`2646.98 ms`, Decode=`5.70 TPS`, Efektif=`0.66 TPS`, Toplam=`3029.36 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2568.32 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`115.10 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
32<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL, prompt_cache=227.91 MB)
- **Performans**: TTFT=`2524.97 ms`, Decode=`6.93 TPS`, Efektif=`0.69 TPS`, Toplam=`2879.40 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2441.81 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`126.65 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=128, Cache Reused=0 (COLD PREFILL, prompt_cache=231.33 MB)
- **Performans**: TTFT=`4144.46 ms`, Decode=`12.35 TPS`, Efektif=`8.84 TPS`, Toplam=`14476.04 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`4069.47 ms`, gpu_compute=`44.50 ms`, prefill_io=`0.00 ms`, checkpoint=`112.18 ms`
- **Bitiş Nedeni**: `length`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
333

17 ile 19 sayilarinin carpimini hesaplamak icin once 17 sayisinin 19 sayisi ile carpimini ele aliriz. Bu islemde 17 sayisi 19 sayisi ile carpilir ve sonucun 333 oldugunu goruruz. Daha sonra bu islemi uzunca aciklayarak adim adim isler. Ilk olarak 17 sayisini 19 sayisi ile carpmaya basladigimizde, 17 sayisinin 19 sayisi ile
```


---
## Benchmark Koşusu: `bench-20261016-080254-8d92` (2026-10-16T08:02:54.339861)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-16T08:02:54.339861`
- **Model**: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `34.30 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 3 | 0 | 2478.8 | 6.37 | 0.71 | COLD (0) [114.3MB] | FAIL |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 2686.5 | 6.70 | 0.66 | COLD (0) [114.4MB] | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 128 | 0 | 4682.1 | 12.29 | 8.50 | COLD (0) [117.2MB] | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=3, Cache Reused=0 (COLD PREFILL, prompt_cache=114.34 MB)
- **Performans**: TTFT=`2478.75 ms`, Decode=`6.37 TPS`, Efektif=`0.71 TPS`, Toplam=`2826.65 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2466.58 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`43.31 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
32<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL, prompt_cache=114.45 MB)
- **Performans**: TTFT=`2686.54 ms`, Decode=`6.70 TPS`, Efektif=`0.66 TPS`, Toplam=`3017.54 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2674.34 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`37.38 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=128, Cache Reused=0 (COLD PREFILL, prompt_cache=117.23 MB)
- **Performans**: TTFT=`4682.05 ms`, Decode=`12.29 TPS`, Efektif=`8.50 TPS`, Toplam=`15060.09 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`4664.48 ms`, gpu_compute=`44.50 ms`, prefill_io=`0.00 ms`, checkpoint=`45.25 ms`
- **Bitiş Nedeni**: `length`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323

17 ile 19 sayılarının çarpımını hesaplamak için, öncelikle 17 sayısını 19 ile çarpma işlemini gerçekleştirmemiz gerekir. Bu işlem, iki basamaklı bir sayının, yine iki basamaklı bir sayıya çarpılmasıdır. 17 sayısı, 10 + 7 şeklinde ifade edilebilirken, 19 sayısı da 10 + 9 şeklinde düşünülebilir. Bu yapısal parçalar, çarpımın daha basit alt işlemlere ayrılmasını sağlay
```


---
## Benchmark Koşusu: `bench-20261016-082603-bd3c` (2026-10-16T08:26:03.211934)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-16T08:26:03.211934`
- **Model**: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `36.39 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 2909.1 | 6.59 | 0.89 | COLD (0) [114.4MB] | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 2959.9 | 8.54 | 0.62 | COLD (0) [114.4MB] | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 128 | 0 | 5068.5 | 12.19 | 8.24 | COLD (0) [117.2MB] | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL, prompt_cache=114.36 MB)
- **Performans**: TTFT=`2909.13 ms`, Decode=`6.59 TPS`, Efektif=`0.89 TPS`, Toplam=`3352.69 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2898.05 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`41.42 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL, prompt_cache=114.45 MB)
- **Performans**: TTFT=`2959.94 ms`, Decode=`8.54 TPS`, Efektif=`0.62 TPS`, Toplam=`3244.02 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2948.78 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`41.89 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=128, Cache Reused=0 (COLD PREFILL, prompt_cache=117.23 MB)
- **Performans**: TTFT=`5068.48 ms`, Decode=`12.19 TPS`, Efektif=`8.24 TPS`, Toplam=`15536.68 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`5055.68 ms`, gpu_compute=`44.50 ms`, prefill_io=`0.00 ms`, checkpoint=`48.82 ms`
- **Bitiş Nedeni**: `length`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323

17 ile 19 sayılarının çarpımının sonucu 323'tür. Bu işlemi daha ayrıntılı bir şekilde açıklamak ve sürecin mantığını anlamak için, çarpma işleminin temel kurallarını ve adım adım nasıl uygulandığını ele alalım. Çarpma, iki veya daha fazla sayının çarpılmasında, çarpılan sayılardan birinin diğer sayıya eklenerek veya sayıya eklenerek elde edilen bir sonuçtur. Bu işlemde, 17 sayısını 19 ile çarpmak
```


---
## Benchmark Koşusu: `bench-20261016-083332-283d` (2026-10-16T08:33:32.193861)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-16T08:33:32.193861`
- **Model**: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `36.54 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 3205.1 | 6.50 | 0.82 | COLD (0) [114.4MB] | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 2944.8 | 8.84 | 0.62 | COLD (0) [114.4MB] | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 128 | 0 | 5090.2 | 12.28 | 8.27 | COLD (0) [117.2MB] | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL, prompt_cache=114.36 MB)
- **Performans**: TTFT=`3205.11 ms`, Decode=`6.50 TPS`, Efektif=`0.82 TPS`, Toplam=`3655.56 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`3197.22 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`39.26 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL, prompt_cache=114.45 MB)
- **Performans**: TTFT=`2944.76 ms`, Decode=`8.84 TPS`, Efektif=`0.62 TPS`, Toplam=`3231.10 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2936.82 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`40.32 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=128, Cache Reused=0 (COLD PREFILL, prompt_cache=117.23 MB)
- **Performans**: TTFT=`5090.16 ms`, Decode=`12.28 TPS`, Efektif=`8.27 TPS`, Toplam=`15485.94 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`5079.39 ms`, gpu_compute=`44.50 ms`, prefill_io=`0.00 ms`, checkpoint=`48.99 ms`
- **Bitiş Nedeni**: `length`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323

17 ile 19 sayılarının çarpımının sonucu 323'tür. Bu işlemi daha ayrıntılı bir şekilde açıklamak ve sürecin mantığını anlamak için, çarpma işleminin temel kurallarını ve adım adım nasıl uygulandığını ele alalım. Çarpma, iki veya daha fazla sayının çarpılmasında, çarpılan sayılardan birinin diğer sayıya eklenerek veya sayıya eklenerek elde edilen bir sonuçtur. Bu işlemde, 17 sayısını 19 ile çarpmak
```


---
## Benchmark Koşusu: `bench-20261016-085456-3ffd` (2026-10-16T08:54:56.544013)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-16T08:54:56.544013`
- **Model**: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `36.06 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 2832.9 | 6.92 | 0.92 | COLD (0) [114.4MB] | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 2877.1 | 8.70 | 0.63 | COLD (0) [114.4MB] | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 128 | 0 | 5024.4 | 12.46 | 8.39 | COLD (0) [117.2MB] | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL, prompt_cache=114.36 MB)
- **Performans**: TTFT=`2832.92 ms`, Decode=`6.92 TPS`, Efektif=`0.92 TPS`, Toplam=`3260.24 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2825.71 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`41.44 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL, prompt_cache=114.45 MB)
- **Performans**: TTFT=`2877.10 ms`, Decode=`8.70 TPS`, Efektif=`0.63 TPS`, Toplam=`3173.36 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2869.15 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`43.47 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=128, Cache Reused=0 (COLD PREFILL, prompt_cache=117.23 MB)
- **Performans**: TTFT=`5024.45 ms`, Decode=`12.46 TPS`, Efektif=`8.39 TPS`, Toplam=`15260.80 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`5011.74 ms`, gpu_compute=`44.50 ms`, prefill_io=`0.00 ms`, checkpoint=`41.45 ms`
- **Bitiş Nedeni**: `length`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323

17 ile 19 sayılarının çarpımının sonucu 323'tür. Bu işlemi daha ayrıntılı bir şekilde açıklamak ve sürecin mantığını anlamak için, çarpma işleminin temel kurallarını ve adım adım nasıl uygulandığını ele alalım. Çarpma, iki veya daha fazla sayının çarpılmasında, çarpılan sayılardan birinin diğer sayıya eklenerek veya sayıya eklenerek elde edilen bir sonuçtur. Bu işlemde, 17 sayısını 19 ile çarpmak
```


---
## Benchmark Koşusu: `bench-20261016-090811-5fb3` (2026-10-16T09:08:11.500304)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-16T09:08:11.500304`
- **Model**: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `23.63 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 3 | 0 | 2440.8 | 7.10 | 0.74 | COLD (0) | FAIL |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 2395.7 | 9.51 | 0.77 | COLD (0) | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 13 | 0 | 4219.4 | 15.21 | 2.40 | COLD (0) | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=3, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`2440.84 ms`, Decode=`7.10 TPS`, Efektif=`0.74 TPS`, Toplam=`2712.48 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2433.12 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
32<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`2395.68 ms`, Decode=`9.51 TPS`, Efektif=`0.77 TPS`, Toplam=`2608.44 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2388.75 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=13, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`4219.44 ms`, Decode=`15.21 TPS`, Efektif=`2.40 TPS`, Toplam=`5000.69 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`4209.99 ms`, gpu_compute=`44.50 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323

17 çarp 19.<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261016-091549-d697` (2026-10-16T09:15:49.005799)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-16T09:15:49.005799`
- **Model**: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `31.07 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 2 | 0 | 2724.0 | 0.00 | 0.35 | COLD (0) | FAIL |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 2518.6 | 10.08 | 0.73 | COLD (0) | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 128 | 0 | 4700.1 | 17.72 | 10.79 | COLD (0) | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=2, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`2724.03 ms`, Decode=`0.00 TPS`, Efektif=`0.35 TPS`, Toplam=`2862.19 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2716.71 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
3<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`2518.63 ms`, Decode=`10.08 TPS`, Efektif=`0.73 TPS`, Toplam=`2742.68 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2510.97 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=128, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`4700.10 ms`, Decode=`17.72 TPS`, Efektif=`10.79 TPS`, Toplam=`11867.50 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`4688.47 ms`, gpu_compute=`44.50 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `length`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323

17 ile 19 sayılarının çarpımını hesaplamak için, çarpma işleminin temel kurallarını ve bu sayıların nasıl çarpılacağını ele alarak, adım adım bir çözümleme yapalım. Bu işlem, iki tek haneli veya iki basaklı sayının çarpımını içerir. 17 sayısı, 17 rakamının 10'lar basanağındaki değeri olan 10 ve 7 rakamının 1'ler basanağındaki değeri olan 7'den oluşur.
```


---
## Benchmark Koşusu: `bench-20261016-091641-bc2c` (2026-10-16T09:16:41.515813)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-16T09:16:41.515813`
- **Model**: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `32.71 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 2907.2 | 8.17 | 0.92 | COLD (0) | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 2926.1 | 9.62 | 0.64 | COLD (0) | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 128 | 0 | 5226.1 | 18.63 | 10.63 | COLD (0) | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`2907.22 ms`, Decode=`8.17 TPS`, Efektif=`0.92 TPS`, Toplam=`3274.57 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2900.73 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`2926.12 ms`, Decode=`9.62 TPS`, Efektif=`0.64 TPS`, Toplam=`3129.01 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2920.52 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=128, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`5226.12 ms`, Decode=`18.63 TPS`, Efektif=`10.63 TPS`, Toplam=`12043.53 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`5216.59 ms`, gpu_compute=`44.50 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `length`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323

17 ile 19 sayılarının çarpımını hesaplamak için ilk olarak 17 sayısını 19 ile çarpmak gerekir. Bu işlem, iki sayının çarpımını bulmak için standart bir çarpma prosedürü gerektirir. 17 sayısını 19 ile çarpmak, 17 sayısını 19 ile çarpmak anlamına gelir. 17 sayısını 19 ile çarpmak, 17 sayısını 19 ile çarpmak anlamına gelir. 17 say
```


---
## Benchmark Koşusu: `bench-20261016-092940-6d10` (2026-10-16T09:29:40.419651)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-16T09:29:40.419651`
- **Model**: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `33.05 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 2864.5 | 8.24 | 0.93 | COLD (0) | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 2879.4 | 9.40 | 0.65 | COLD (0) | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 128 | 0 | 5096.4 | 17.16 | 10.24 | COLD (0) | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`2864.49 ms`, Decode=`8.24 TPS`, Efektif=`0.93 TPS`, Toplam=`3226.58 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2856.40 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`2879.39 ms`, Decode=`9.40 TPS`, Efektif=`0.65 TPS`, Toplam=`3076.01 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2870.27 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=128, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`5096.45 ms`, Decode=`17.16 TPS`, Efektif=`10.24 TPS`, Toplam=`12496.43 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`5082.90 ms`, gpu_compute=`44.50 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `length`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323

17 ile 19 sayılarının çarpımını hesaplamak için ilk olarak 17 sayısını 19 ile çarpmak gerekir. Bu işlem, iki sayının çarpımını bulmak için standart bir çarpma prosedürü gerektirir. 17 sayısını 19 ile çarpmak, 17 sayısını 19 ile çarpmak anlamına gelir. 17 sayısını 19 ile çarpmak, 17 sayısını 19 ile çarpmak anlamına gelir. 17 say
```


---
## Benchmark Koşusu: `bench-20261016-140106-149d` (2026-10-16T14:01:06.411501)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-16T14:01:06.411501`
- **Model**: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `38.10 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 3198.1 | 8.32 | 0.85 | COLD (0) | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 3151.5 | 10.24 | 0.60 | COLD (0) | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 128 | 0 | 5989.8 | 16.30 | 9.29 | COLD (0) | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`3198.08 ms`, Decode=`8.32 TPS`, Efektif=`0.85 TPS`, Toplam=`3544.79 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`3189.74 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`3151.54 ms`, Decode=`10.24 TPS`, Efektif=`0.60 TPS`, Toplam=`3355.27 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`3144.41 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=128, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`5989.81 ms`, Decode=`16.30 TPS`, Efektif=`9.29 TPS`, Toplam=`13782.34 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`5977.38 ms`, gpu_compute=`44.50 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `length`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323

17 ile 19 sayılarının çarpımını hesaplamak için ilk olarak 17 sayısını 19 ile çarpmak gerekir. Bu işlem, iki sayının çarpımını bulmak için standart bir çarpma prosedürü gerektirir. 17 sayısını 19 ile çarpmak, 17 sayısını 19 ile çarpmak anlamına gelir. 17 sayısını 19 ile çarpmak, 17 sayısını 19 ile çarpmak anlamına gelir. 17 say
```


---
## Benchmark Koşusu: `bench-20261019-002056-21f7` (2026-10-19T00:20:56.296504)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-19T00:20:56.296504`
- **Model**: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `36.17 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 2938.4 | 8.16 | 0.91 | COLD (0) | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 4088.4 | 4.12 | 0.43 | COLD (0) | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 128 | 0 | 7339.4 | 17.73 | 8.83 | COLD (0) | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`2938.45 ms`, Decode=`8.16 TPS`, Efektif=`0.91 TPS`, Toplam=`3293.30 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2931.23 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`4088.45 ms`, Decode=`4.12 TPS`, Efektif=`0.43 TPS`, Toplam=`4608.43 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`4081.55 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=128, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`7339.36 ms`, Decode=`17.73 TPS`, Efektif=`8.83 TPS`, Toplam=`14503.73 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`7324.79 ms`, gpu_compute=`44.50 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `length`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323

17 ile 19 sayılarının çarpımını hesaplamak için ilk olarak 17 sayısını 19 ile çarpmak gerekir. Bu işlem, iki sayının çarpımını bulmak için standart bir çarpma prosedürü gerektirir. 17 sayısını 19 ile çarpmak, 17 sayısını 19 ile çarpmak anlamına gelir. 17 sayısını 19 ile çarpmak, 17 sayısını 19 ile çarpmak anlamına gelir. 17 say
```


---
## Benchmark Koşusu: `bench-20261019-003345-3312` (2026-10-19T00:33:45.353320)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-19T00:33:45.353320`
- **Model**: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `30.01 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 2888.2 | 7.60 | 0.92 | COLD (0) | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 3793.5 | 9.13 | 0.50 | COLD (0) | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 17 | 0 | 5763.0 | 14.22 | 2.33 | COLD (0) | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`2888.19 ms`, Decode=`7.60 TPS`, Efektif=`0.92 TPS`, Toplam=`3260.46 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2877.64 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`3793.49 ms`, Decode=`9.13 TPS`, Efektif=`0.50 TPS`, Toplam=`4009.08 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`3783.78 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=17, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`5763.02 ms`, Decode=`14.22 TPS`, Efektif=`2.33 TPS`, Toplam=`6870.72 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`5752.78 ms`, gpu_compute=`44.50 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323

323

17 çarpı 19<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261019-003552-8499` (2026-10-19T00:35:52.213769)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-19T00:35:52.213769`
- **Model**: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `33.59 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 2875.4 | 7.23 | 0.92 | COLD (0) | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 3089.5 | 8.38 | 0.60 | COLD (0) | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 128 | 0 | 6241.9 | 20.04 | 10.18 | COLD (0) | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`2875.41 ms`, Decode=`7.23 TPS`, Efektif=`0.92 TPS`, Toplam=`3274.24 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2868.74 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`3089.52 ms`, Decode=`8.38 TPS`, Efektif=`0.60 TPS`, Toplam=`3328.53 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`3082.80 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=128, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`6241.85 ms`, Decode=`20.04 TPS`, Efektif=`10.18 TPS`, Toplam=`12579.43 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`6229.55 ms`, gpu_compute=`44.50 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `length`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323

17 ile 19 sayılarının çarpımını hesapla. Yalnızca sayıyı yaz. Sonrasında boşluk bırakarak aynı işlemi uzunca açıkla. Bunu yaparken mümkün olduğunca fazla detay ver ki token sayısı 64'ü geçsin.

323

17 ile 19 sayılarının çarpımını hesapla. Yalnızca sayıyı yaz. Sonrasında boşluk bırakarak aynı işlemi uzunca açıkla. Bunu yaparken mümkün olduğunca fazla detay ver ki token sayısı 64'ü geçsin
```


---
## Benchmark Koşusu: `bench-20261019-004518-0951` (2026-10-19T00:45:18.997570)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-19T00:45:18.997570`
- **Model**: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `33.74 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 2862.9 | 7.79 | 0.92 | COLD (0) | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 2981.3 | 9.65 | 0.63 | COLD (0) | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 128 | 0 | 6716.8 | 20.00 | 9.80 | COLD (0) | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`2862.95 ms`, Decode=`7.79 TPS`, Efektif=`0.92 TPS`, Toplam=`3252.78 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2855.68 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`2981.28 ms`, Decode=`9.65 TPS`, Efektif=`0.63 TPS`, Toplam=`3180.05 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2975.28 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=128, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`6716.77 ms`, Decode=`20.00 TPS`, Efektif=`9.80 TPS`, Toplam=`13067.78 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`6705.84 ms`, gpu_compute=`44.50 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `length`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323

17 ile 19 sayılarının çarpımını hesapla. Yalnızca sayıyı yaz. Sonrasında boşluk bırakarak aynı işlemi uzunca açıkla. Bunu yaparken mümkün olduğunca fazla detay ver ki token sayısı 64'ü geçsin.

323

17 ile 19 sayılarının çarpımını hesapla. Yalnızca sayıyı yaz. Sonrasında boşluk bırakarak aynı işlemi uzunca açıkla. Bunu yaparken mümkün olduğunca fazla detay ver ki token sayısı 64'ü geçsin
```


---
## Benchmark Koşusu: `bench-20261019-005622-d5e0` (2026-10-19T00:56:22.019163)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-19T00:56:22.019163`
- **Model**: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `33.52 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 2924.1 | 7.59 | 0.91 | COLD (0) | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 3695.4 | 7.90 | 0.51 | COLD (0) | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 128 | 0 | 5928.1 | 20.23 | 10.49 | COLD (0) | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`2924.11 ms`, Decode=`7.59 TPS`, Efektif=`0.91 TPS`, Toplam=`3314.68 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2916.91 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`3695.44 ms`, Decode=`7.90 TPS`, Efektif=`0.51 TPS`, Toplam=`3933.20 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`3688.82 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=128, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`5928.10 ms`, Decode=`20.23 TPS`, Efektif=`10.49 TPS`, Toplam=`12205.48 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`5918.55 ms`, gpu_compute=`44.50 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `length`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323

17 ile 19 sayılarının çarpımını hesapla. Yalnızca sayıyı yaz. Sonrasında boşluk bırakarak aynı işlemi uzunca açıkla. Bunu yaparken mümkün olduğunca fazla detay ver ki token sayısı 64'ü geçsin.

323

17 ile 19 sayılarının çarpımını hesapla. Yalnızca sayıyı yaz. Sonrasında boşluk bırakarak aynı işlemi uzunca açıkla. Bunu yaparken mümkün olduğunca fazla detay ver ki token sayısı 64'ü geçsin
```


---
## Benchmark Koşusu: `bench-20261019-005852-1eb6` (2026-10-19T00:58:52.817275)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-19T00:58:52.817275`
- **Model**: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `32.60 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 2959.5 | 7.36 | 0.90 | COLD (0) | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 2994.5 | 8.97 | 0.62 | COLD (0) | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 128 | 0 | 5861.9 | 20.13 | 10.52 | COLD (0) | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`2959.53 ms`, Decode=`7.36 TPS`, Efektif=`0.90 TPS`, Toplam=`3340.04 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2952.65 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`2994.49 ms`, Decode=`8.97 TPS`, Efektif=`0.62 TPS`, Toplam=`3209.70 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2987.99 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=128, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`5861.91 ms`, Decode=`20.13 TPS`, Efektif=`10.52 TPS`, Toplam=`12172.41 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`5851.69 ms`, gpu_compute=`44.50 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `length`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323

17 ile 19 sayılarının çarpımını hesapla. Yalnızca sayıyı yaz. Sonrasında boşluk bırakarak aynı işlemi uzunca açıkla. Bunu yaparken mümkün olduğunca fazla detay ver ki token sayısı 64'ü geçsin.

323

17 ile 19 sayılarının çarpımını hesapla. Yalnızca sayıyı yaz. Sonrasında boşluk bırakarak aynı işlemi uzunca açıkla. Bunu yaparken mümkün olduğunca fazla detay ver ki token sayısı 64'ü geçsin
```


---
## Benchmark Koşusu: `bench-20261019-010059-4db4` (2026-10-19T01:00:59.129317)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-19T01:00:59.129317`
- **Model**: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `32.55 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 2883.3 | 7.77 | 0.92 | COLD (0) | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 3157.6 | 8.60 | 0.59 | COLD (0) | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 128 | 0 | 6099.7 | 20.04 | 10.29 | COLD (0) | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`2883.31 ms`, Decode=`7.77 TPS`, Efektif=`0.92 TPS`, Toplam=`3260.30 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`2874.41 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`3157.56 ms`, Decode=`8.60 TPS`, Efektif=`0.59 TPS`, Toplam=`3381.88 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`3149.26 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=128, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`6099.68 ms`, Decode=`20.04 TPS`, Efektif=`10.29 TPS`, Toplam=`12436.51 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`6085.67 ms`, gpu_compute=`44.50 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `length`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323

17 ile 19 sayılarının çarpımını hesapla. Yalnızca sayıyı yaz. Sonrasında boşluk bırakarak aynı işlemi uzunca açıkla. Bunu yaparken mümkün olduğunca fazla detay ver ki token sayısı 64'ü geçsin.

323

17 ile 19 sayılarının çarpımını hesapla. Yalnızca sayıyı yaz. Sonrasında boşluk bırakarak aynı işlemi uzunca açıkla. Bunu yaparken mümkün olduğunca fazla detay ver ki token sayısı 64'ü geçsin
```


---
## Benchmark Koşusu: `bench-20261019-084422-e402` (2026-10-19T08:44:22.501654)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-19T08:44:22.501654`
- **Model**: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=14, Prompt Cache=256 MB
- **Toplam Koşu Süresi**: `23.99 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 44 | 422.0 | 6.40 | 3.45 | HIT (44) | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 50 | 346.4 | 8.79 | 3.41 | HIT (50) | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 65 | 0 | 7122.2 | 20.26 | 6.23 | COLD (0) | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=44 (CACHE HIT (44 tokens reused))
- **Performans**: TTFT=`422.02 ms`, Decode=`6.40 TPS`, Efektif=`3.45 TPS`, Toplam=`870.34 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`292.87 ms`, gpu_compute=`1.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=50 (CACHE HIT (50 tokens reused))
- **Performans**: TTFT=`346.41 ms`, Decode=`8.79 TPS`, Efektif=`3.41 TPS`, Toplam=`585.90 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`212.46 ms`, gpu_compute=`1.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=65, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`7122.19 ms`, Decode=`20.26 TPS`, Efektif=`6.23 TPS`, Toplam=`10273.02 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`6369.41 ms`, gpu_compute=`44.50 ms`, prefill_io=`0.00 ms`, checkpoint=`738.57 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323

17 ile 19 sayılarının çarpımını hesapla. Yalnızca sayıyı yaz. Sonrasında boşluk bırakarak aynı işlemi uzunca açıkla. Bunu yaparken mümkün olduğunca fazla detay ver ki token sayısı 64'ü geçsin.<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261019-084456-65c8` (2026-10-19T08:44:56.800565)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-19T08:44:56.800565`
- **Model**: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=14, Prompt Cache=256 MB
- **Toplam Koşu Süresi**: `15.25 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 44 | 447.4 | 6.18 | 3.36 | HIT (44) | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 50 | 390.0 | 7.23 | 3.08 | HIT (50) | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 16 | 87 | 380.3 | 13.68 | 10.35 | HIT (87) | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=44 (CACHE HIT (44 tokens reused))
- **Performans**: TTFT=`447.44 ms`, Decode=`6.18 TPS`, Efektif=`3.36 TPS`, Toplam=`893.56 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`295.66 ms`, gpu_compute=`1.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=50 (CACHE HIT (50 tokens reused))
- **Performans**: TTFT=`390.00 ms`, Decode=`7.23 TPS`, Efektif=`3.08 TPS`, Toplam=`650.24 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`233.78 ms`, gpu_compute=`1.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=16, Cache Reused=87 (CACHE HIT (87 tokens reused))
- **Performans**: TTFT=`380.33 ms`, Decode=`13.68 TPS`, Efektif=`10.35 TPS`, Toplam=`1449.80 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`227.96 ms`, gpu_compute=`1.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323

17 * 19 = 323<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261019-084528-db5d` (2026-10-19T08:45:28.729408)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-19T08:45:28.729408`
- **Model**: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=14, Prompt Cache=256 MB
- **Toplam Koşu Süresi**: `34.61 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `32k-turn1-cold` | 32768 | `q8_0` | 186 | 20 | 0 | 13766.1 | 14.11 | 1.26 | COLD (0) | PASS |
| `32k-turn2-warm` | 32768 | `q8_0` | 240 | 22 | 184 | 3408.2 | 13.80 | 4.28 | HIT (184) | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `32k-turn1-cold` | Context: 32768 | Turn: 1
- **Kategori / İş Yükü**: `architecture` / `multi_turn_cache`
- **Context & KV Belleği**: `32768` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=186, Üretilen=20, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`13766.07 ms`, Decode=`14.11 TPS`, Efektif=`1.26 TPS`, Toplam=`15094.95 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`13593.83 ms`, gpu_compute=`93.00 ms`, prefill_io=`0.00 ms`, checkpoint=`153.78 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are an expert AI software architect for high-performance low-latency MoE inference engines.<|im_end|> <|im_start|>user Context Document: Tendou launcher is a...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
NVIDIA RTX 5060 Laptop with 8 GB GDDR6 VRAM.<|im_end|>
```

#### [2] Görev: `32k-turn2-warm` | Context: 32768 | Turn: 2
- **Kategori / İş Yükü**: `architecture` / `multi_turn_cache`
- **Context & KV Belleği**: `32768` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=240, Üretilen=22, Cache Reused=184 (CACHE HIT (184 tokens reused))
- **Performans**: TTFT=`3408.23 ms`, Decode=`13.80 TPS`, Efektif=`4.28 TPS`, Toplam=`4903.88 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`3188.93 ms`, gpu_compute=`28.00 ms`, prefill_io=`0.00 ms`, checkpoint=`151.47 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are an expert AI software architect for high-performance low-latency MoE inference engines.<|im_end|> <|im_start|>user Context Document: Tendou launcher is a...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
The P3 module uses pinned staging buffers and CUDA event fences to enable non-blocking host-to-device transfers.<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261019-090927-e406` (2026-10-19T09:09:27.764857)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-19T09:09:27.764857`
- **Model**: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=256 MB
- **Toplam Koşu Süresi**: `18.57 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 44 | 450.9 | 6.33 | 3.30 | HIT (44) | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 50 | 401.4 | 8.69 | 3.12 | HIT (50) | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 38 | 87 | 372.0 | 14.95 | 13.03 | HIT (87) | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=44 (CACHE HIT (44 tokens reused))
- **Performans**: TTFT=`450.91 ms`, Decode=`6.33 TPS`, Efektif=`3.30 TPS`, Toplam=`908.58 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`309.93 ms`, gpu_compute=`1.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=50 (CACHE HIT (50 tokens reused))
- **Performans**: TTFT=`401.35 ms`, Decode=`8.69 TPS`, Efektif=`3.12 TPS`, Toplam=`641.31 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`255.13 ms`, gpu_compute=`1.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=38, Cache Reused=87 (CACHE HIT (87 tokens reused))
- **Performans**: TTFT=`371.97 ms`, Decode=`14.95 TPS`, Efektif=`13.03 TPS`, Toplam=`2839.99 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`231.89 ms`, gpu_compute=`1.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323

17 çarpım tablosu ile 19 sayısal yapısı ve çarpım tablosu ile 17 sayısal yapısı<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261019-090951-0651` (2026-10-19T09:09:51.831706)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-19T09:09:51.831706`
- **Model**: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=14, Prompt Cache=256 MB
- **Toplam Koşu Süresi**: `16.65 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 44 | 460.2 | 5.96 | 3.26 | HIT (44) | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 50 | 359.2 | 9.28 | 3.40 | HIT (50) | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 16 | 87 | 360.9 | 13.53 | 10.42 | HIT (87) | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=44 (CACHE HIT (44 tokens reused))
- **Performans**: TTFT=`460.16 ms`, Decode=`5.96 TPS`, Efektif=`3.26 TPS`, Toplam=`921.24 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`310.86 ms`, gpu_compute=`1.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=50 (CACHE HIT (50 tokens reused))
- **Performans**: TTFT=`359.17 ms`, Decode=`9.28 TPS`, Efektif=`3.40 TPS`, Toplam=`588.43 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`210.66 ms`, gpu_compute=`1.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=16, Cache Reused=87 (CACHE HIT (87 tokens reused))
- **Performans**: TTFT=`360.85 ms`, Decode=`13.53 TPS`, Efektif=`10.42 TPS`, Toplam=`1439.71 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`215.08 ms`, gpu_compute=`1.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323

17 * 19 = 323<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261019-092121-f104` (2026-10-19T09:21:21.598075)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-19T09:21:21.598075`
- **Model**: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=0 MB
- **Toplam Koşu Süresi**: `22.45 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 44 | 453.5 | 6.75 | 3.48 | HIT (44) | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 50 | 381.8 | 9.31 | 3.31 | HIT (50) | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 128 | 87 | 373.1 | 15.32 | 14.78 | HIT (87) | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=44 (CACHE HIT (44 tokens reused))
- **Performans**: TTFT=`453.54 ms`, Decode=`6.75 TPS`, Efektif=`3.48 TPS`, Toplam=`863.30 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`311.10 ms`, gpu_compute=`1.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=50 (CACHE HIT (50 tokens reused))
- **Performans**: TTFT=`381.77 ms`, Decode=`9.31 TPS`, Efektif=`3.31 TPS`, Toplam=`603.90 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`229.22 ms`, gpu_compute=`1.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=128, Cache Reused=87 (CACHE HIT (87 tokens reused))
- **Performans**: TTFT=`373.15 ms`, Decode=`15.32 TPS`, Efektif=`14.78 TPS`, Toplam=`8660.83 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`221.08 ms`, gpu_compute=`1.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `length`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323

17 ve 19 sayılarının çarpımını hesaplamak için, iki sayının da birim ve onluk basaklarını içeren bir sayısal yapıya sahip olduğu göz önüne alınmalıdır. Bu iki sayılar, tam olarak 17 ve 19, her ikisi de bir ve onluk basaklarını içeren iki basaklı sayılar olarak tanımlanabilir. Bu bağlamda, 17 sayisi, birim basaklarında 7 ve onluk basaklarında 1 olmakla tanımlanabilir. Benzer şekilde, 1
```


---
## Benchmark Koşusu: `bench-20261019-092155-a1d7` (2026-10-19T09:21:55.235915)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-19T09:21:55.235915`
- **Model**: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=256 MB
- **Toplam Koşu Süresi**: `32.91 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 0 | 3210.9 | 7.97 | 0.85 | COLD (0) | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 0 | 3176.5 | 8.28 | 0.59 | COLD (0) | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 65 | 0 | 5753.6 | 19.71 | 7.11 | COLD (0) | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`3210.86 ms`, Decode=`7.97 TPS`, Efektif=`0.85 TPS`, Toplam=`3549.22 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`3074.58 ms`, gpu_compute=`23.00 ms`, prefill_io=`0.00 ms`, checkpoint=`127.91 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`3176.46 ms`, Decode=`8.28 TPS`, Efektif=`0.59 TPS`, Toplam=`3403.96 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`3040.40 ms`, gpu_compute=`26.00 ms`, prefill_io=`0.00 ms`, checkpoint=`127.95 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=65, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`5753.58 ms`, Decode=`19.71 TPS`, Efektif=`7.11 TPS`, Toplam=`8996.26 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`5568.38 ms`, gpu_compute=`44.50 ms`, prefill_io=`0.00 ms`, checkpoint=`145.56 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323

17 ile 19 sayılarının çarpımını hesapla. Yalnızca sayıyı yaz. Sonrasında boşluk bırakarak aynı işlemi uzunca açıkla. Bunu yaparken mümkün olduğunca fazla detay ver ki token sayısı 64'ü geçsin.<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261019-092235-049e` (2026-10-19T09:22:35.328049)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-19T09:22:35.328049`
- **Model**: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=256 MB
- **Toplam Koşu Süresi**: `22.30 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 44 | 440.4 | 6.59 | 3.48 | HIT (44) | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 50 | 370.3 | 9.48 | 3.43 | HIT (50) | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 128 | 87 | 385.4 | 15.43 | 14.86 | HIT (87) | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=44 (CACHE HIT (44 tokens reused))
- **Performans**: TTFT=`440.43 ms`, Decode=`6.59 TPS`, Efektif=`3.48 TPS`, Toplam=`862.18 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`301.19 ms`, gpu_compute=`1.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=50 (CACHE HIT (50 tokens reused))
- **Performans**: TTFT=`370.30 ms`, Decode=`9.48 TPS`, Efektif=`3.43 TPS`, Toplam=`582.84 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`247.07 ms`, gpu_compute=`1.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=128, Cache Reused=87 (CACHE HIT (87 tokens reused))
- **Performans**: TTFT=`385.39 ms`, Decode=`15.43 TPS`, Efektif=`14.86 TPS`, Toplam=`8613.75 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`235.56 ms`, gpu_compute=`1.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `length`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323

17 ve 19 sayılarının çarpımını hesaplamak için, iki sayının da birim ve onluk basaklarını içeren bir sayısal yapıya sahip olduğu göz önüne alınmalıdır. Bu iki sayılar, tam olarak 17 ve 19, her ikisi de bir ve onluk basaklarını içeren iki basaklı sayılar olarak tanımlanabilir. Bu bağlamda, 17 sayisi, birim basaklarında 7 ve onluk basaklarında 1 olmakla tanımlanabilir. Benzer şekilde, 1
```


---
## Benchmark Koşusu: `bench-20261019-092305-b0f1` (2026-10-19T09:23:05.288050)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-19T09:23:05.288050`
- **Model**: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=256 MB
- **Toplam Koşu Süresi**: `36.14 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `32k-turn1-cold` | 32768 | `q8_0` | 186 | 20 | 0 | 14428.7 | 14.06 | 1.21 | COLD (0) | PASS |
| `32k-turn2-warm` | 32768 | `q8_0` | 240 | 22 | 184 | 4643.2 | 14.02 | 3.43 | HIT (184) | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `32k-turn1-cold` | Context: 32768 | Turn: 1
- **Kategori / İş Yükü**: `architecture` / `multi_turn_cache`
- **Context & KV Belleği**: `32768` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=186, Üretilen=20, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`14428.73 ms`, Decode=`14.06 TPS`, Efektif=`1.21 TPS`, Toplam=`15761.84 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`14273.21 ms`, gpu_compute=`93.00 ms`, prefill_io=`0.00 ms`, checkpoint=`135.12 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are an expert AI software architect for high-performance low-latency MoE inference engines.<|im_end|> <|im_start|>user Context Document: Tendou launcher is a...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
NVIDIA RTX 5060 Laptop with 8 GB GDDR6 VRAM.<|im_end|>
```

#### [2] Görev: `32k-turn2-warm` | Context: 32768 | Turn: 2
- **Kategori / İş Yükü**: `architecture` / `multi_turn_cache`
- **Context & KV Belleği**: `32768` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=240, Üretilen=22, Cache Reused=184 (CACHE HIT (184 tokens reused))
- **Performans**: TTFT=`4643.22 ms`, Decode=`14.02 TPS`, Efektif=`3.43 TPS`, Toplam=`6126.02 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`4411.99 ms`, gpu_compute=`28.00 ms`, prefill_io=`0.00 ms`, checkpoint=`135.02 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are an expert AI software architect for high-performance low-latency MoE inference engines.<|im_end|> <|im_start|>user Context Document: Tendou launcher is a...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
The P3 module uses pinned staging buffers and CUDA event fences to enable asynchronous Host-to-Device transfers.<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261019-092959-3df0` (2026-10-19T09:29:59.261438)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-19T09:29:59.261438`
- **Model**: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `22.56 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 44 | 423.3 | 6.23 | 3.48 | HIT (44) | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 50 | 352.7 | 9.39 | 3.47 | HIT (50) | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 128 | 87 | 356.2 | 15.43 | 14.91 | HIT (87) | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=44 (CACHE HIT (44 tokens reused))
- **Performans**: TTFT=`423.33 ms`, Decode=`6.23 TPS`, Efektif=`3.48 TPS`, Toplam=`861.01 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`297.49 ms`, gpu_compute=`1.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=50 (CACHE HIT (50 tokens reused))
- **Performans**: TTFT=`352.69 ms`, Decode=`9.39 TPS`, Efektif=`3.47 TPS`, Toplam=`575.68 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`228.32 ms`, gpu_compute=`1.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=128, Cache Reused=87 (CACHE HIT (87 tokens reused))
- **Performans**: TTFT=`356.16 ms`, Decode=`15.43 TPS`, Efektif=`14.91 TPS`, Toplam=`8586.27 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`220.29 ms`, gpu_compute=`1.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `length`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323

17 ve 19 sayılarının çarpımını hesaplamak için, iki sayının da birim ve onluk basaklarını içeren bir sayısal yapıya sahip olduğu göz önüne alınmalıdır. Bu iki sayılar, tam olarak 17 ve 19, her ikisi de bir ve onluk basaklarını içeren iki basaklı sayılar olarak tanımlanabilir. Bu bağlamda, 17 sayisi, birim basaklarında 7 ve onluk basaklarında 1 olmakla tanımlanabilir. Benzer şekilde, 1
```


---
## Benchmark Koşusu: `bench-20261019-183255-41cb` (2026-10-19T18:32:55.902483)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-19T18:32:55.902483`
- **Model**: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `21.11 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 36 | 44 | 543.6 | 24.31 | 18.15 | HIT (44) | FAIL |
| `short-code-01` | 2048 | `q8_0` | 52 | 36 | 50 | 431.3 | 25.40 | 19.90 | HIT (50) | FAIL |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 128 | 87 | 349.8 | 26.15 | 24.58 | HIT (87) | FAIL |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=36, Cache Reused=44 (CACHE HIT (44 tokens reused))
- **Performans**: TTFT=`543.56 ms`, Decode=`24.31 TPS`, Efektif=`18.15 TPS`, Toplam=`1983.40 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`346.10 ms`, gpu_compute=`1.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `length`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
3!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=36, Cache Reused=50 (CACHE HIT (50 tokens reused))
- **Performans**: TTFT=`431.33 ms`, Decode=`25.40 TPS`, Efektif=`19.90 TPS`, Toplam=`1809.26 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`261.68 ms`, gpu_compute=`1.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `length`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
4!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=128, Cache Reused=87 (CACHE HIT (87 tokens reused))
- **Performans**: TTFT=`349.76 ms`, Decode=`26.15 TPS`, Efektif=`24.58 TPS`, Toplam=`5207.21 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`222.84 ms`, gpu_compute=`1.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `length`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
3!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```


---
## Benchmark Koşusu: `bench-20261019-183505-d129` (2026-10-19T18:35:05.816175)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-19T18:35:05.816175`
- **Model**: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `20.19 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 44 | 451.3 | 2.01 | 1.90 | HIT (44) | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 50 | 377.4 | 10.60 | 3.41 | HIT (50) | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 63 | 87 | 362.7 | 14.54 | 13.47 | HIT (87) | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=44 (CACHE HIT (44 tokens reused))
- **Performans**: TTFT=`451.34 ms`, Decode=`2.01 TPS`, Efektif=`1.90 TPS`, Toplam=`1575.23 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`331.52 ms`, gpu_compute=`1.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=50 (CACHE HIT (50 tokens reused))
- **Performans**: TTFT=`377.45 ms`, Decode=`10.60 TPS`, Efektif=`3.41 TPS`, Toplam=`586.36 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`248.32 ms`, gpu_compute=`1.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=63, Cache Reused=87 (CACHE HIT (87 tokens reused))
- **Performans**: TTFT=`362.68 ms`, Decode=`14.54 TPS`, Efektif=`13.47 TPS`, Toplam=`4603.43 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`224.19 ms`, gpu_compute=`1.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323

17 çarp 19, 17 çarp 19, 17 çarp 19, 17 çarp 19, 17 çarp 19, 17 çarp 19, 17 çarp<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261019-183742-0a43` (2026-10-19T18:37:42.911396)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-19T18:37:42.911396`
- **Model**: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `49.90 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `32k-turn1-cold` | 32768 | `q8_0` | 186 | 20 | 0 | 14065.9 | 12.55 | 1.22 | COLD (0) | PASS |
| `32k-turn2-warm` | 32768 | `q8_0` | 240 | 31 | 0 | 16782.8 | 12.61 | 1.57 | COLD (0) | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `32k-turn1-cold` | Context: 32768 | Turn: 1
- **Kategori / İş Yükü**: `architecture` / `multi_turn_cache`
- **Context & KV Belleği**: `32768` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=186, Üretilen=20, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`14065.91 ms`, Decode=`12.55 TPS`, Efektif=`1.22 TPS`, Toplam=`15555.43 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`13387.45 ms`, gpu_compute=`93.00 ms`, prefill_io=`0.00 ms`, checkpoint=`538.86 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are an expert AI software architect for high-performance low-latency MoE inference engines.<|im_end|> <|im_start|>user Context Document: Tendou launcher is a...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
NVIDIA RTX 5060 Laptop with 8 GB GDDR6 VRAM.<|im_end|>
```

#### [2] Görev: `32k-turn2-warm` | Context: 32768 | Turn: 2
- **Kategori / İş Yükü**: `architecture` / `multi_turn_cache`
- **Context & KV Belleği**: `32768` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=240, Üretilen=31, Cache Reused=0 (COLD PREFILL)
- **Performans**: TTFT=`16782.85 ms`, Decode=`12.61 TPS`, Efektif=`1.57 TPS`, Toplam=`19140.83 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`16461.58 ms`, gpu_compute=`120.00 ms`, prefill_io=`0.00 ms`, checkpoint=`120.59 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are an expert AI software architect for high-performance low-latency MoE inference engines.<|im_end|> <|im_start|>user Context Document: Tendou launcher is a...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
Module P3 (Asynchronous H2D Transfer Pipeline) uses pinned staging buffers and CUDA event fences to enable non-blocking host-to-device data transfer.<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261019-183857-2f23` (2026-10-19T18:38:57.475574)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-19T18:38:57.475574`
- **Model**: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `23.69 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 44 | 432.7 | 6.87 | 3.62 | HIT (44) | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 50 | 352.2 | 11.62 | 3.77 | HIT (50) | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 113 | 87 | 385.2 | 16.42 | 15.56 | HIT (87) | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=44 (CACHE HIT (44 tokens reused))
- **Performans**: TTFT=`432.73 ms`, Decode=`6.87 TPS`, Efektif=`3.62 TPS`, Toplam=`828.72 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`301.33 ms`, gpu_compute=`1.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=50 (CACHE HIT (50 tokens reused))
- **Performans**: TTFT=`352.20 ms`, Decode=`11.62 TPS`, Efektif=`3.77 TPS`, Toplam=`531.10 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`227.52 ms`, gpu_compute=`1.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=113, Cache Reused=87 (CACHE HIT (87 tokens reused))
- **Performans**: TTFT=`385.22 ms`, Decode=`16.42 TPS`, Efektif=`15.56 TPS`, Toplam=`7196.46 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`244.53 ms`, gpu_compute=`1.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323

17 çarp 19, yani 17 sayısının 19 ile çarpılması, sonucun 323 olmasına neden olur. Bu işlem, iki sayının çarpımını, yani 17 ve 19'un çarpımını, sonucun 323 olmasına neden olur. Bu işlem, iki sayının çarpımını, yani 17 ve 19'un çarpımını, sonucun 323 olmasına neden olur.<|im_end|>
```


---
## Benchmark Koşusu: `bench-20261019-185552-c65e` (2026-10-19T18:55:52.515164)

### 1. Çalışma Konfigürasyonu ve Donanım
- **Zaman**: `2026-10-19T18:55:52.515164`
- **Model**: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002`
- **Donanım**: CPU: `AMD Ryzen 7 260 w/ Radeon 780M Graphics (16 threads)` | GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB` | RAM: `15.29 GB`
- **Parametreler**: Threads=8, Prompt Cache=1024 MB
- **Toplam Koşu Süresi**: `18.34 s`

### 2. Metrik Özeti Tablosu
| Görev ID | Context | KV Quant | Prompt Tok | Gen Tok | Cache Tok | TTFT (ms) | Decode TPS | Efektif TPS | Cache Durumu | Doğruluk |
|---|---|---|---|---|---|---|---|---|---|---|
| `short-math-01` | 2048 | `q8_0` | 46 | 4 | 44 | 410.4 | 7.31 | 3.92 | HIT (44) | PASS |
| `short-code-01` | 2048 | `q8_0` | 52 | 3 | 50 | 358.9 | 11.32 | 3.65 | HIT (50) | PASS |
| `short-math-01-measure` | 2048 | `q8_0` | 89 | 65 | 87 | 326.6 | 13.75 | 12.88 | HIT (87) | PASS |

### 3. Model Çıktıları ve Detaylı Metrik Dökümü (AI Generated Outputs)

#### [1] Görev: `short-math-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=46, Üretilen=4, Cache Reused=44 (CACHE HIT (44 tokens reused))
- **Performans**: TTFT=`410.36 ms`, Decode=`7.31 TPS`, Efektif=`3.92 TPS`, Toplam=`764.64 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`289.73 ms`, gpu_compute=`1.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz.<|im_end|> <|im_start|>assistant <think> </t...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323<|im_end|>
```

#### [2] Görev: `short-code-01` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `code` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=52, Üretilen=3, Cache Reused=50 (CACHE HIT (50 tokens reused))
- **Performans**: TTFT=`358.85 ms`, Decode=`11.32 TPS`, Efektif=`3.65 TPS`, Toplam=`547.59 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`238.15 ms`, gpu_compute=`1.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Python ifadesinin sonucunu yalnizca deger olarak yaz: sum([12, 14, 16])<|im_end|> <|im_start|>assistant <...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
42<|im_end|>
```

#### [3] Görev: `short-math-01-measure` | Context: 2048 | Turn: 0
- **Kategori / İş Yükü**: `math` / `short_request`
- **Context & KV Belleği**: `2048` tokens (KV: `q8_0`)
- **Token Sayaçları**: Prompt=89, Üretilen=65, Cache Reused=87 (CACHE HIT (87 tokens reused))
- **Performans**: TTFT=`326.61 ms`, Decode=`13.75 TPS`, Efektif=`12.88 TPS`, Toplam=`4968.03 ms`
- **C++ Prefill & Bellek Ayrımı**: prefill_compute=`213.10 ms`, gpu_compute=`1.00 ms`, prefill_io=`0.00 ms`, checkpoint=`0.00 ms`
- **Bitiş Nedeni**: `stop`
- **Prompt**: `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user 17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzu...`
- **Modelin Ürettiği Çıktı (AI Generated Output)**:
```text
323

17 ile 19 sayılarının çarpımını hesapla. Yalnızca sayıyı yaz. Sonrasında boşluk bırakarak aynı işlemi uzunca açıkla. Bunu yaparken mümkün olduğunca fazla detay ver ki token sayısı 64'ü geçsin.<|im_end|>
```


