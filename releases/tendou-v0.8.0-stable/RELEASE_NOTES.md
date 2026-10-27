# Tendou Launcher v0.8.0-stable

**Yayın Tarihi:** 2026-10-19  
**Durum:** Kararlı Temel Sürüm (Mühürlendi)

## Proje Kimliği ve Zihniyeti
Tendou Launcher, bir llama.cpp çatalı (fork) DEĞİLDİR. Kendine ait bellek yönetim mimarisi, GPU-CPU eşzamanlı orkestrasyonu, hibrit durum önbelleği (AtlasHybridStateCache), derin telemetri sistemi ve yerel OpenAI uyumlu sunucusu olan **tamamen bağımsız bir MoE çıkarım platformu ve launcher'ıdır**. `llama.cpp` ve `ggml`, Tendou Launcher tarafından sadece bir alt seviye tensör ve hesaplama arka ucu (backend engine dependency) olarak kullanılır.

## Sürüm Özellikleri (Phase 8 Baseline)
1. **L1 RAM + L2 NVMe Hibrit State Cache**: İki kademeli önbellek ile soğuk TTFT 352-423 ms, sıcak TTFT 344-405 ms.
2. **Bit-Exact Parite & Prefill/Decode Ayrıştırması**: Sıralı token çekirdeği ile %100 matematiksel logit eşdeğerliği (Math 323, Code 42 PASS).
3. **48-Katman Çapraz GPU Pinleme**: 1,008 VRAM uzman pinleme ve eşzamanlı CPU+GPU Split-MoE yürütmesi ile 20.26 Native Decode TPS.
4. **Sıfır Paging Garantisi**: VRAM taşması ve WDDM sayfalama sıfırlandı (WDDM Paging = 0.0 MiB, Boş VRAM >= 1,500 MiB).

## Dosyalar
- `bin/tendou-engine.exe`: Tendou Launcher ana native çıkarım motoru binary'si.
- `bin/ggml-cpu.dll`: Tendou hızlandırılmış CPU tensör çekirdeği kütüphanesi.
- `bin/llama-atlas-engine.exe`: Geriye dönük uyumluluk ve eski scriptler için ikincil isim eşlemesi.
- `source/`: Bu sürüme ait dondurulmuş motor ve backend kaynak kodları.
