# Tendou Launcher — Qwen3.6-35B-A3B (UD-IQ4_XS) Benchmark Raporu

**Tarih:** 2026-09-26 14:29:59  
**Model:** `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` (16.51 GB, 40 Katman, 256 Uzman, K=8)  
**Fiziksel Harita:** `config/atlas_physical_map_35b.json` (10,240 uzman parçası)  
**Donanım:** AMD Ryzen 7 260 (8C/16T Zen 4), NVIDIA RTX 5060 Laptop (8151 MiB GDDR6), 16 GB DDR5 RAM  

## 1. Performans Özeti (Metrik Karşılaştırma)

| Senaryo | Görev | Prompt Tok | Gen Tok | TTFT (ms) | Decode TPS | Prefill TPS | RAM (GB) | VRAM (MB) | Sonuç |
|---|---|---|---|---|---|---|---|---|---|
| `8-Threads_GPU2_NativeK` | `math_cold` | 33 | 24 | 1268.62 | **33.04** | 26.78 | 13.343 | 3699.0 | FAIL |
| `8-Threads_GPU2_NativeK` | `math_warm_cache (HIT)` | 92 | 23 | 354.95 | **39.38** | 196.89 | 13.709 | 3700.0 | FAIL |
| `8-Threads_GPU2_NativeK` | `code_generation` | 31 | 4 | 507.23 | **28.3** | 63.0 | 14.302 | 3700.0 | FAIL |
| `8-Threads_GPU2_NativeK` | `deep_reasoning_128t` | 40 | 45 | 740.64 | **25.94** | 55.2 | 14.897 | 3702.0 | - |
| `8-Threads_GPU2_NativeK` | `medium_prefix_prefill` | 1081 | 16 | 11693.02 | **25.06** | 93.32 | 15.195 | 3682.0 | FAIL |
| `16-Threads_GPU2_NativeK` | `math_cold` | 33 | 24 | 1132.22 | **34.15** | 30.16 | 11.86 | 3678.0 | FAIL |
| `16-Threads_GPU2_NativeK` | `math_warm_cache (HIT)` | 93 | 24 | 297.31 | **35.63** | 149.75 | 12.212 | 3678.0 | FAIL |
| `16-Threads_GPU2_NativeK` | `code_generation` | 31 | 21 | 532.48 | **28.14** | 60.28 | 13.237 | 3680.0 | FAIL |
| `16-Threads_GPU2_NativeK` | `deep_reasoning_128t` | 40 | 128 | 630.64 | **36.49** | 65.26 | 14.337 | 3680.0 | - |
| `16-Threads_GPU2_NativeK` | `medium_prefix_prefill` | 1081 | 16 | 12162.33 | **23.4** | 89.81 | 15.2 | 3700.0 | FAIL |
| `16-Threads_GPU6_NativeK` | `math_cold` | 33 | 2 | 1049.88 | **19.24** | 32.64 | 12.164 | 5106.0 | FAIL |
| `16-Threads_GPU6_NativeK` | `math_warm_cache (HIT)` | 72 | 3 | 317.56 | **22.29** | 139.97 | 12.435 | 5129.0 | FAIL |
| `16-Threads_GPU6_NativeK` | `code_generation` | 31 | 64 | 565.41 | **32.22** | 56.65 | 13.83 | 5110.0 | PASS |
| `16-Threads_GPU6_NativeK` | `deep_reasoning_128t` | 40 | 128 | 593.72 | **36.62** | 69.46 | 14.455 | 5104.0 | - |
| `16-Threads_GPU6_NativeK` | `medium_prefix_prefill` | 1081 | 3 | 11454.89 | **25.09** | 95.3 | 15.159 | 5106.0 | PASS |
| `16-Threads_GPU6_BoostK5` | `math_cold` | 33 | 3 | 1041.92 | **22.55** | 32.99 | 11.826 | 5098.0 | FAIL |
| `16-Threads_GPU6_BoostK5` | `math_warm_cache (HIT)` | 73 | 0 | 316.24 | **0.0** | 140.5 | 12.078 | 5098.0 | FAIL |
| `16-Threads_GPU6_BoostK5` | `code_generation` | 31 | 64 | 575.79 | **29.22** | 55.5 | 13.625 | 5104.0 | FAIL |
| `16-Threads_GPU6_BoostK5` | `deep_reasoning_128t` | 40 | 128 | 580.98 | **39.88** | 71.17 | 14.562 | 5104.0 | - |
| `16-Threads_GPU6_BoostK5` | `medium_prefix_prefill` | 1081 | 5 | 10941.16 | **31.18** | 99.91 | 15.059 | 5106.0 | FAIL |
| `16-Threads_GPU6_8kContext` | `math_cold` | 33 | 24 | 1007.13 | **32.7** | 34.17 | 11.73 | 5148.0 | FAIL |
| `16-Threads_GPU6_8kContext` | `math_warm_cache (HIT)` | 93 | 2 | 302.74 | **24.31** | 148.51 | 11.979 | 5148.0 | FAIL |
| `16-Threads_GPU6_8kContext` | `code_generation` | 31 | 64 | 599.9 | **32.06** | 53.27 | 13.304 | 5150.0 | PASS |
| `16-Threads_GPU6_8kContext` | `deep_reasoning_128t` | 40 | 128 | 592.83 | **34.62** | 69.6 | 14.475 | 5150.0 | - |
| `16-Threads_GPU6_8kContext` | `medium_prefix_prefill` | 1081 | 4 | 10543.65 | **23.53** | 103.48 | 15.045 | 5152.0 | PASS |

## 2. Üretilen Model Yanıtları ve Kalite Doğrulaması

### `8-Threads_GPU2_NativeK` — `math_cold`
- **Decode TPS:** 33.04 | **TTFT:** 1268.62 ms | **Toplam Süre:** 2018.35 ms
- **Çıktı:**
```text
17 ile 19 sayılarının carpimi = 227

Yalnizca result:
```

### `8-Threads_GPU2_NativeK` — `math_warm_cache`
- **Decode TPS:** 39.38 | **TTFT:** 354.95 ms | **Toplam Süre:** 980.86 ms
- **Çıktı:**
```text
17 ile 19 sayılarının carpimi = 227

Yalnizca result:<|im_end|>
```

### `8-Threads_GPU2_NativeK` — `code_generation`
- **Decode TPS:** 28.3 | **TTFT:** 507.23 ms | **Toplam Süre:** 662.65 ms
- **Çıktı:**
```text
Here's a Python<|im_end|>
```

### `8-Threads_GPU2_NativeK` — `deep_reasoning_128t`
- **Decode TPS:** 25.94 | **TTFT:** 740.64 ms | **Toplam Süre:** 2493.63 ms
- **Çıktı:**
```text
The text you provided seems to be a mix of English and some other language (possibly Turkish or another language with similar script). It looks like a fragment of a sentence or a technical description.

Let's break it down:<|im_end|>
```

### `8-Threads_GPU2_NativeK` — `medium_prefix_prefill`
- **Decode TPS:** 25.06 | **TTFT:** 11693.02 ms | **Toplam Süre:** 12363.49 ms
- **Çıktı:**
```text
The text appears to be a series of nonsensical sentences or phrases, likely
```

### `16-Threads_GPU2_NativeK` — `math_cold`
- **Decode TPS:** 34.15 | **TTFT:** 1132.22 ms | **Toplam Süre:** 1860.87 ms
- **Çıktı:**
```text
**17 ile 19 sayilarinin carpimi**

**Yalnizca sonucu yaz**

**
```

### `16-Threads_GPU2_NativeK` — `math_warm_cache`
- **Decode TPS:** 35.63 | **TTFT:** 297.31 ms | **Toplam Süre:** 986.68 ms
- **Çıktı:**
```text
**17 ile 19 sayilarinin carpimi**

**Yalnizca sonucu yaz**

**
```

### `16-Threads_GPU2_NativeK` — `code_generation`
- **Decode TPS:** 28.14 | **TTFT:** 532.48 ms | **Toplam Süre:** 1310.51 ms
- **Çıktı:**
```text
Bonjour!

Vous avez besoin de calculer la median (médian) pour une Python list.<|im_end|>
```

### `16-Threads_GPU2_NativeK` — `deep_reasoning_128t`
- **Decode TPS:** 36.49 | **TTFT:** 630.64 ms | **Toplam Süre:** 4157.32 ms
- **Çıktı:**
```text
# Mixture-of-Experts (MoE) Mimari ve Gating Yapısı

Mixture-of-Experts (MoE), der learning mimari bir variantıdır ve bu mimari, **gating yapısı** (gizli gate) ile **expert (expert)** yapısı** bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir
```

### `16-Threads_GPU2_NativeK` — `medium_prefix_prefill`
- **Decode TPS:** 23.4 | **TTFT:** 12162.33 ms | **Toplam Süre:** 12872.39 ms
- **Çıktı:**
```text
<think>

</think>

This is a very strange request. The user seems to be
```

### `16-Threads_GPU6_NativeK` — `math_cold`
- **Decode TPS:** 19.24 | **TTFT:** 1049.88 ms | **Toplam Süre:** 1216.48 ms
- **Çıktı:**
```text
15<|im_end|>
```

### `16-Threads_GPU6_NativeK` — `math_warm_cache`
- **Decode TPS:** 22.29 | **TTFT:** 317.56 ms | **Toplam Süre:** 499.73 ms
- **Çıktı:**
```text
102<|im_end|>
```

### `16-Threads_GPU6_NativeK` — `code_generation`
- **Decode TPS:** 32.22 | **TTFT:** 565.41 ms | **Toplam Süre:** 2572.16 ms
- **Çıktı:**
```text
Python中，计算列表的median（中位数）通常有两种方法：手动实现和用内置函数。这里提供一个简短的自定义函数，用于计算列表的中位数。

```python
def calculate_median(lst):
    """
    计算列表的中位数
    :param lst: 列表
```

### `16-Threads_GPU6_NativeK` — `deep_reasoning_128t`
- **Decode TPS:** 36.62 | **TTFT:** 593.72 ms | **Toplam Süre:** 4113.52 ms
- **Çıktı:**
```text
# 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
```

### `16-Threads_GPU6_NativeK` — `medium_prefix_prefill`
- **Decode TPS:** 25.09 | **TTFT:** 11454.89 ms | **Toplam Süre:** 11641.99 ms
- **Çıktı:**
```text
Tendou<|im_end|>
```

### `16-Threads_GPU6_BoostK5` — `math_cold`
- **Decode TPS:** 22.55 | **TTFT:** 1041.92 ms | **Toplam Süre:** 1228.65 ms
- **Çıktı:**
```text
313<|im_end|>
```

### `16-Threads_GPU6_BoostK5` — `math_warm_cache`
- **Decode TPS:** 0.0 | **TTFT:** 316.24 ms | **Toplam Süre:** 383.66 ms
- **Çıktı:**
```text
<|im_end|>
```

### `16-Threads_GPU6_BoostK5` — `code_generation`
- **Decode TPS:** 29.22 | **TTFT:** 575.79 ms | **Toplam Süre:** 2782.85 ms
- **Çıktı:**
```text
Here is the answer for this question.

The prompt asks for a Python function to calculate the median of a list. However, the question is incomplete and has some grammatical errors (e.g., "med" instead of "median", "kas" instead of "short"). Let's assume the user wants a simple
```

### `16-Threads_GPU6_BoostK5` — `deep_reasoning_128t`
- **Decode TPS:** 39.88 | **TTFT:** 580.98 ms | **Toplam Süre:** 3813.16 ms
- **Çıktı:**
```text
Mixture-of-Experts (MoE) yapısı, bir input (input) için en uygun expert'leri (expert'ler) seçer ve bu expert'ler, bu input'u farklı olarak (farklı) işler. Bu, her bir expert'in (expert'ler) input'u (input) farklı bir şekilde (farklı) işleyerek (işleyerek) en optimal (optimal) output'u (output) üreseinin (üreseinin) en optimal (optimal) output'u (output) üreseinin (üreseinin) en optimal (optimal
```

### `16-Threads_GPU6_BoostK5` — `medium_prefix_prefill`
- **Decode TPS:** 31.18 | **TTFT:** 10941.16 ms | **Toplam Süre:** 11151.05 ms
- **Çıktı:**
```text
Bilgi Belgesi<|im_end|>
```

### `16-Threads_GPU6_8kContext` — `math_cold`
- **Decode TPS:** 32.7 | **TTFT:** 1007.13 ms | **Toplam Süre:** 1758.58 ms
- **Çıktı:**
```text
15.000.000.000.000.000.0
```

### `16-Threads_GPU6_8kContext` — `math_warm_cache`
- **Decode TPS:** 24.31 | **TTFT:** 302.74 ms | **Toplam Süre:** 433.82 ms
- **Çıktı:**
```text
77<|im_end|>
```

### `16-Threads_GPU6_8kContext` — `code_generation`
- **Decode TPS:** 32.06 | **TTFT:** 599.9 ms | **Toplam Süre:** 2599.38 ms
- **Çıktı:**
```text
# 计算 Median 的函数

```python
def calculate_median(numbers):
    """
    计算一组数字的 median
    
    Args:
        numbers: 数字列表
        
    Returns:
        median 值
    """
    if not numbers:
        return None
    
    sorted_numbers
```

### `16-Threads_GPU6_8kContext` — `deep_reasoning_128t`
- **Decode TPS:** 34.62 | **TTFT:** 592.83 ms | **Toplam Süre:** 4310.68 ms
- **Çıktı:**
```text
Mixture-of-Experts (MoE) 架构中，路由机制（routing）是如何工作的？

MoE 的核心在于“路由”（routing）机制。它通过一个**路由网络**（通常是 softmax 或类似函数）来动态分配输入样本到不同的专家（Expert）模块。

具体工作流程如下：

1. **输入**：每个样本进入 MoE 层时，首先被送入路由网络。
2. **路由决策**：路由网络根据样本的特征，计算每个专家（Expert）的“激活分数”（activation scores）。
3. **选择专家**：系统
```

### `16-Threads_GPU6_8kContext` — `medium_prefix_prefill`
- **Decode TPS:** 23.53 | **TTFT:** 10543.65 ms | **Toplam Süre:** 10747.65 ms
- **Çıktı:**
```text
Tendou launcher<|im_end|>
```
