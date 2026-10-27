# Tendou Launcher — Araştırma ve Doğrulama Verileri (docs/json)

Bu dizin, Tendou Launcher mimari geliştirmeleri, hızlandırma fazları ve teknik dokümantasyonların doğrulama kanıtı olan JSON çıktılarını içerir.

## Dosya Haritası ve Doğrulama Eşleşmeleri

| JSON Dosyası | Doğruladığı Doküman / Aşama | Açıklama |
|---|---|---|
| `benchmark_phase8_vs_phase9_results.json` | `docs/releases/release-phase9.md`, `AGENT.md` | Phase 8 vs Phase 9 100-token x 5 prompt karşılaştırmalı doğrulama; TTFT, TPS, VRAM ve parite kanıtı |
| `phase5_results.json` | `docs/audits/`, Phase 5 | Phase 5 MTP ve dynamic-K gecikme & parite ölçüm sonuçları |
| `phase4_results.json` | Phase 4 Placement | Dynamic Layer/Expert yerleşim ve bellek ayak izi çıktısı |
| `phase3_results.json` | Phase 3 Transfer | Asenkron H2D transfer ve transfer gecikme metrikleri |
| `v7_baseline_report.json` | Phase 7 Baseline | Phase 7 öncesi 48-katman CPU/GPU durum raporu |
| `v7_benchmark_report.json` | Phase 7 Pinning | 48-katman çapraz GPU pinleme ve eşzamanlı split-MoE sonuçları |
| `qwen36_benchmark_report.json` | `docs/reports/qwen36_benchmark_report.md` | Qwen 3.6 35B model çıkarım ve bellek profili |
| `qwen36_q8_benchmark_report.json` | `docs/reports/qwen36_q8_benchmark_report.md` | Qwen 3.6 35B Q8_K_XL yüksek hassasiyetli MoE testi |
| `p1_ttft_breakdown_baseline.json` | Phase 1 TTFT Profiling | Prefill darboğazı ve TTFT bileşen dökümü |
| `runtime_audit_baseline.json` | `docs/audits/RUNTIME_AUDIT_2026-09-09.md` | Orijinal referans çalışma zamanı denetim tablosu |
| `runtime_audit_optimized.json` | `docs/audits/RUNTIME_AUDIT_2026-09-09.md` | Optimize edilmiş motor çalışma zamanı denetim tablosu |
| `runtime_audit_comparison.json` | `docs/audits/RUNTIME_AUDIT_2026-09-09.md` | Baseline vs Optimize karşılaştırması |
| `speed_matrix_summary.json` | `docs/benchmarks/BENCHMARK_LOG.md` | Hız matrisi ve throughput kırılımları |
| `breakthrough_matrix_summary.json` | `docs/benchmarks/BENCHMARK_LOG.md` | Atılım hedefleri ve TPS kazanımları |
| `task_a_summary.json` | Track A | Asenkron H2D/D2H & Shared DAG çıktısı |
| `task_b_summary.json` | Track B | Boost Level & Monoton DDR5 akışı |
| `task_c_summary.json` | Track C | VRAM-Resident MTP Draft araştırması |
