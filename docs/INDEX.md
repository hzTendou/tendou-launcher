# Tendou Launcher — Dokümantasyon Merkezi (docs/)

Tendou Launcher, bağımsız MoE çıkarım platformu için hazırlanmış tüm mimari belgeleri, teknik makaleleri, denetim raporlarını, sürüm notlarını ve kullanım kılavuzlarını içerir.

---

## 📚 Kapsamlı Teknik Makaleler (`docs/articles/`)
Geliştirme sürecinin derinlemesine analizi, mühendislik kararları ve Ar-Ge sonuçları:
- 📖 [`RESEARCH_HISTORY_AND_METHODOLOGY.md`](articles/RESEARCH_HISTORY_AND_METHODOLOGY.md): Phase 0'dan Phase 9'a gelişim yolculuğu, kronolojik Ar-Ge analizi ve kanıta dayalı çalışma metodolojisi.
- 🔬 [`HYPOTHESIS_ANALYSIS_PROVEN_AND_REFUTED.md`](articles/HYPOTHESIS_ANALYSIS_PROVEN_AND_REFUTED.md): Çürütülen hipotezler (Memory Wall, SMT, MTP, döngü içi prefetch) ve kanıtlanan sistem mimarileri (monoton akış, 48-katman GPU pinleme, determinizm).
- ⚙️ [`ARCHITECTURE_AND_INFERENCE_ENGINE.md`](articles/ARCHITECTURE_AND_INFERENCE_ENGINE.md): `PhysicalExpert`, `CrossLayerGPUPinning`, `AtlasHybridStateCache` ve 1x4 AVX2/FMA tensör çekirdekleri iç mimarisi.
- 🚀 [`ROADMAP_AND_OPEN_PROBLEMS.md`](articles/ROADMAP_AND_OPEN_PROBLEMS.md): Gelecek yol haritası, Phase 10 (64k/128k bağlam ölçekleme), AVX-512 VNNI ve ucu açık araştırma konuları.

---

## 🛠️ Kullanıcı ve Entegrasyon Kılavuzları
- 📘 [`USER_GUIDE.md`](USER_GUIDE.md): Tendou Launcher kapsamlı kurulum, konfigürasyon, API kullanımı, OpenCode entegrasyonu ve problem giderme kılavuzu.

---

## 🏛️ Mimari ve Sistem Şartnameleri
- **`architecture/`**:
  - [`BACKEND_INTEGRATION.md`](architecture/BACKEND_INTEGRATION.md): Alt seviye tensör hesaplama motoru (`llama.cpp` / `ggml`) bağımlılık arayüzü.
- **`specs/`**:
  - [`V1_RUNTIME_SPEC.md`](specs/V1_RUNTIME_SPEC.md): Çalışma zamanı mimari spesifikasyonu.
  - [`LLAMA4_ARCHITECTURE_AND_OPTIMIZATION_PLAN.md`](specs/LLAMA4_ARCHITECTURE_AND_OPTIMIZATION_PLAN.md): Llama-4 Scout (Top-1 + Shared Expert) mimari desteği, optimizasyon, stabilite ve benchmark planı.

---

## 📦 Sürümler ve Denetim Raporları
- **`releases/`**:
  - [`release-phase9.md`](releases/release-phase9.md): Phase 9 Kararlı Sürüm (`v0.9.0-stable`) mimari doğrulaması.
  - [`release-phase6.md`](releases/release-phase6.md): Phase 6 Parite ve Çıkarım Mimarisi.
- **`audits/`**:
  - [`RUNTIME_AUDIT_2026-09-09.md`](audits/RUNTIME_AUDIT_2026-09-09.md): Çalışma zamanı donanım denetimi.
  - [`RUNTIME_AUDIT_2026-09-10.md`](audits/RUNTIME_AUDIT_2026-09-10.md): Ayrıntılı bellek ve iş parçacığı analizi.
- **`reports/`**:
  - [`qwen36_benchmark_report.md`](reports/qwen36_benchmark_report.md): Qwen 3.6 35B model analizi.
  - [`qwen36_q8_benchmark_report.md`](reports/qwen36_q8_benchmark_report.md): Qwen 3.6 Q8_K_XL yüksek hassasiyet testi.
- **`benchmarks/`**:
  - [`BENCHMARK_LOG.md`](benchmarks/BENCHMARK_LOG.md): Tarihsel TPS, TTFT ve DDR5 ayak izi logları.

---

## 📊 Araştırma ve Doğrulama Verileri (`docs/json/`)
Dokümantasyonlardaki tüm iddia ve metrikleri doğrulayan 18 adet kanıt JSON veri seti:
- [`docs/json/README.md`](json/README.md): Doğrulama dosyaları kataloğu.

---

## 📜 Tarihsel Kayıtlar ve Devir Arşivi
- [`NEXT_STEPS.md`](NEXT_STEPS.md): Phase 10 görev listesi.
- [`agent-history.md`](agent-history.md): Ajan geçmiş devir kayıtları arşivi.
- `history/`: Eski geliştirme dokümanları.
