# Tendou Launcher — Test Mimarisi ve Dizin Yapısı (tests/)

Bu dizin, Tendou Launcher platformunun tüm birim, entegrasyon, aşama (phase), native C++ ve kıyaslama (benchmark) testlerini tek bir merkezi yapıda toplar.

## Klasör ve Aşama Haritası

### 1. `unit/` (Çekirdek Birim Testleri)
- `test_atlas_qwen38.py`: Qwen 3.8 model parametreleri, katman ve uzman topolojisi doğrulama testleri.
- `test_benchmark_logger.py`: Kıyaslama ve metrik kayıt tutma testleri.
- `test_freetoken_policy.py`: Serbest token önbellek ve kullanım politikası testleri.
- `test_odmoe_spice_tutti.py`: Expert prefetch ve kuyruk derinliği mantık testleri.

### 2. `integration/` (Sistem ve API Entegrasyon Testleri)
- `test_openai_server.py`: OpenAI uyumlu FastAPI HTTP sunucu uç noktaları (`/v1/models`, `/v1/chat/completions`).
- `test_engine_lifecycle.py`: IPC motor başlatma, hazır olma kontrolü ve sonlandırma yaşam döngüsü.
- `test_runtime_e2e.py`: Uçtan uca token üretim ve yanıt bütünlüğü testi.

### 3. `phases/` (Geliştirme Aşamalarına Göre Düzenlenmiş Testler)
Her bir alt klasör, projenin ilgili mimari fazında geliştirilmiş denetim (audit), doğrulama ve analiz testlerini içerir:

| Aşama Klasörü | İlgili Faz | İçerik ve Testler |
|---|---|---|
| **`phase0_exploration/`** | Phase 0 | İlk keşif ve temel DDR5 bant genişliği testleri (`test_p0_benchmark.py`, `p0_benchmark.py`). |
| **`phase1_ttft/`** | Phase 1 | Prefill ve TTFT profil çıkarma testleri (`test_p1_ttft_profiler.py`, `p1_ttft_profiler.py`). |
| **`phase2_gpu_binding/`** | Phase 2 | GPU katman bağlama, VRAM denetimleri ve bellek tavanı testleri (`test_p2_gpu_binder.py`, `audit_tlep_phase2.py`, `test_gpu_layers_1.py`, `test_vram_gpu1.py`, `test_vram_inspection.py`). |
| **`phase3_async_transfer/`**| Phase 3 | Asenkron PCIe H2D/D2H transfer testleri ve karşılaştırmaları (`test_p3_async_transfer.py`, `audit_tlep_phase3.py`, `bench_mode_a.py`, `bench_tlep.py`). |
| **`phase4_placement/`** | Phase 4 | Dinamik katman ve uzman yerleşim algoritmaları testleri (`test_p4_placement.py`, `audit_tlep_phase4.py`). |
| **`phase5_mtp_eval/`** | Phase 5 | MTP (Multi-Token Prediction) spekülasyonu ve Dynamic-K denetimleri (`test_p5_mtp_evaluator.py`, `audit_tlep_phase5.py`, `benchmark_mtp.py`, `bench_vram_mtp.py`, `test_hybrid_mtp.py`, `test_mtp_exploration.py`). |
| **`phase6_parity/`** | Phase 6 | Sıfır-hata matematiksel logit paritesi, Mode-C ve kalite dilimleri (`test_parity_now.py`, `test_parity_explain.py`, `test_mode_c.py`, `test_mode_c_splits.py`, `test_quality_slice.py`, `quick_test.py`, `find_perfect_parity_prompt.py`, `debug_step_690.py`). |
| **`phase7_cross_layer_pinning/`** | Phase 7 | 48-katman çapraz GPU sıcak uzman pinleme (1,008 VRAM uzman) ve 20+ TPS denetimleri (`audit_tlep_phase7.py`, `benchmark_resident.py`, `evaluate_resident.py`, `benchmark_qwen36_comprehensive.py`). |
| **`phase8_stable_baseline/`** | Phase 8 | `v0.8.0-stable` temel sürüm dondurma ve sert kapı (gate) denetimleri (`audit_tlep_phase8.py`). |
| **`phase9_overlap_quants/`** | Phase 9 | `v0.9.0-stable` GPU/CPU zamanlayıcı örtüşmesi, 1x4 AVX2 IQ3_XXS kernel ve 100-token x 5 prompt karşılaştırmalı benchmark (`audit_phase9_milestones.py`, `benchmark_phase8_vs_phase9.py`). |

### 4. `native/` (C++ Native Testler)
- `test_async_prefetch.cpp`: Native asenkron prefetch ve 0-ULP doğrulama testi.
- `test_expert_prefetch.cpp`: Native uzman önbellek yükleme testi.
- `test_vnni_parity.cpp`: VNNI / AVX2 çekirdek parite testi.
- `mem_bench.cpp`: DDR5 ve bellek bant genişliği mikro-benchmark'ı.
- `bench_q5k_isolation.cpp`: Q5_K_S izolasyon testi.
- `build_partition_tests.cmd`: Partition testlerini derleme betiği.

### 5. `benchmarks/` (Sürekli Performans Testleri)
- `run_continuous_benchmark.py`: Sürekli TTFT, TPS ve DDR5 MB/token izleme aracı.
- `run_benchmarks.py`: Genel kıyaslama paketi.
- `run_gpu_first_benchmark.py`: GPU-first mimari kıyaslaması.
- `run_grouped_gemm_benchmark.py`: Gruplu matris çarpımı kıyaslaması.
- `run_speed_matrix.py`: Kapsamlı hız matrisi ölçümü.
- `benchmark_native_api.py`: Native API gecikme ölçümü.
- `analyze_token_waterfall.py`: Token waterfall gecikme dökümü.

### 6. `legacy/` (Eski Sürüm Testleri)
- `legacy_v6/`: Eski v0.6 test takımı.
- `legacy_v7/`: Eski v0.7 simülatör test takımı.
- `legacy_v7_shims/`: Legacy simülatör import köprüleri (`atlas_simulator.py` vb.).

### 7. `bin/` (Test Binary ve Obje Dosyaları)
- Derlenmiş test çalıştırılabilirleri (`.exe`) ve ara derleme objeleri (`.obj`).
