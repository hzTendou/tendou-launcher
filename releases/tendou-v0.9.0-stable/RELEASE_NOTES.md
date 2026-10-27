# Tendou Launcher v0.9.0-stable (LATEST STABLE RELEASE)

**Yayın Tarihi:** 2026-10-19  
**Durum:** En Güncel Kararlı Sürüm (Mühürlendi ve Doğrulandı)

## Proje Kimliği ve Zihniyeti
Tendou Launcher, bir llama.cpp sürümü, modu veya çatalı (fork) DEĞİLDİR; kendine ait bellek yöneticisi (`PhysicalExpert`, `CrossLayerGPUPinning`), asenkron orkestrasyonu, sıfır-kopya hibrit durum önbelleği (`AtlasHybridStateCache`), telemetri izlemesi ve yerel API sunucusu olan **tamamen bağımsız bir MoE çıkarım platformu ve launcher'ıdır**. `llama.cpp` ve `ggml`, yalnızca alt seviye tensör ve hesaplama arka ucu (backend engine dependency) olarak kullanılır.

## Sürüm Özellikleri (Phase 9 Milestone 1, 2, 3)
1. **GPU ffn_shexp <-> CPU MoE Zamanlayıcı Örtüşmesi (Milestone 1)**:
   - Split 0 Router bitiminde 10.2 KB D2H erken senkronizasyonu ile CPU MoE `MUL_MAT_ID` ve GPU `ffn_shexp` eşzamanlı örtüştürüldü.
2. **1x4 AVX2/FMA IQ3_XXS Çekirdeği & Decode Syscall Eliminasyonu (Milestone 1 & 2)**:
   - 1x4 yazmaç bloklu `ggml_vec_dot_iq3_xxs_q8_K_x4` çekirdeği ile 4 satır paralel çarpım ve 0-ULP bit-exact parite.
   - Tek-token decode'da `PrefetchVirtualMemory` syscall'ları ve GPU `cb_eval` duraklamaları kaldırılarak 21.58 - 23.86 TPS decode sağlandı.
3. **32K Asimetrik KV & Sıfır-Kopya State (Milestone 3)**:
   - `n_ctx > 8192` için `params.cache_type_v = GGML_TYPE_Q4_0` asimetrik KV mimarisi.
   - `AtlasHybridStateCache` sıfır-kopya ve O(1) `llama_memory_clear`.
4. **100-Token Karşılaştırmalı Doğrulama Sonuçları**:
   - Soğuk TTFT: **315.6–399.6 ms** (ortalama **353.9 ms** - Phase 8'e göre 17 kat hızlanma).
   - Sıcak TTFT: **182.3–207.6 ms** (ortalama **192.4 ms**).
   - Warm Native Decode: **21.58 TPS** (kod üretimi: **22.82 TPS**, yaratıcı sohbet: **23.86 TPS**).
   - DDR5 Ayak İzi: Cold 29.34 MB/tok, Warm 22.47 MB/tok (%-30 düşüş).
   - Parite: 5/5 Tam Deterministik Parite (Math 323, Code 42, Explain, Chat PASS).

## Dosyalar
- `bin/tendou-engine.exe`: Tendou Launcher ana native çıkarım motoru binary'si.
- `bin/ggml-cpu.dll`: Tendou hızlandırılmış CPU tensör çekirdeği kütüphanesi.
- `bin/llama-atlas-engine.exe`: Geriye dönük uyumluluk ve eski scriptler için ikincil isim eşlemesi.
- `source/`: Bu sürüme ait dondurulmuş motor ve backend kaynak kodları.
