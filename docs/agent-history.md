# tendou launcher — geçmiş ajan devir kayıtları

2026-09-14 | Antigravity / Zen 4 NTA Prefetch, 8-Row Unroll, Sıfır D2H Readback & 8.77 TPS | DOĞRULANDI
- İş/değişen dosyalar: `ggml-cpu.c`, `atlas-engine.cpp`, `server.py`, `agent-history.md`, `AGENT.md`.
- Kanıt ve sonuç: 1) `ggml-cpu.c`: `src1` tek-token kuantizasyonunda 3 kat iç içe döngü atlanarak 1D fast-path yapıldı, `src1_col` aktivasyonu L1'e `_MM_HINT_T0` ile sabitlendi, Q5_K için 8-satır unrolling eklendi, ardışık `_MM_HINT_NTA` non-temporal prefetching ile DDR5 L2/L3 kirlenmesi önlendi. 2) `atlas-engine.cpp`: `fast_parse_int` ile token başı 96 CRT string parse elendi; `min_k == max_k` katmanlarında GPU `weights_norm` D2H okuması tamamen baypas edildi (%100 D2H elendi); katman 44-47 `max_k = 2` desteği aktifleştirildi; boost readback-warmup 0 yapıldı. 3) Gerçek RTX 5060 + Ryzen 7 260 donanımında ortalama decode hızı 7.10 TPS'den 8.76-8.77 TPS'ye (+%23.5) yükseldi, pik hız 11.15 TPS'ye ulaştı; kalite kesiti 9/10 PASS (%90 doğruluk, %80 kuralı sağlandı). 179 pytest ve native testler geçti.
- Devam: Katmanlar arası dinamik K ve donanım pipeline paralelliğinin genişletilmesi.

2026-09-13 | Antigravity / Dynamic K End 47 Aktivasyonu, Katman 44-47 Koruma & 8.32 TPS | DOĞRULANDI
- İş/değişen dosyalar: `atlas-engine.h`, `atlas-engine.cpp`, `test_quality_slice.py`, `agent-history.md`, `AGENT.md`.
- Kanıt ve sonuç: 1) `dynamic_k_end=43` kısıtlaması nedeniyle ulaşılamaz kalan katman 44-47 koruma kodu `dynamic_k_end=47` yapılarak aktifleştirildi; çıkış katmanlarında gereksiz 10 expert hesaplaması yerine kalibre edilmiş K=2-3 budaması devreye girdi. 2) Windows asyncio IPC pipe kapanış ve taskkill sonlandırması sertleştirildi. 3) RTX 5060 + Ryzen 7 260 üzerinde `test_quality_slice` (10 görev) %90 doğruluk (9/10 PASS, %80 tabanının üzerinde) ile ortalama decode hızını 6.99 TPS'den 8.32 TPS'ye (pik 10.66-12.64 TPS) çıkardı (+%19-60 TPS artışı). 179 pytest ve tüm native testler geçti.
- Devam: P0 tam 100 görevlik benchmark suite'i üzerinde agresif profil ölçümü.

2026-09-13 | Antigravity / Katman Uyarlamalı Agresif K Budaması, GPU Bypass & 5.40-7.09 TPS | DOĞRULANDI
- İş/değişen dosyalar: `atlas-engine.h`, `atlas-engine.cpp`, `test_quality_slice.py`, `agent-history.md`.
- Kanıt ve sonuç: 1) Clipgfy dinamik K budamasına dinamik K_min=1, K_max=3, min_weight=0.10 ve katman uyarlamalı koruma (akıl yürütme katmanları 16-32 ve token formatlama 44-47 korumalı) eklendi. 2) `topk` budandığında (`-1`) `ggml_compute_forward_mul_mat_id` ilgili sütunu sıfırladığı için gereksiz GPU `weights_norm` tensor yazımı tamamen elendi (token başı 44 CUDA API çağrısı tasarruf edildi). 3) Sampled router readback aralığı 64'e çekilerek GPU senkronizasyonları decode sırasında asgariye indirildi. RTX 5060 + Ryzen 7 260 üzerinde `test_quality_slice` varsayılan modda 9/10 PASS (%90 doğruluk, %80 tabanının üzerinde) ile ortalama decode hızını 2.91 TPS'den 5.40 TPS'ye (pik 7.18 TPS), hiper agresif modda ise 7.09 TPS'ye (pik 10.27 TPS) çıkardı (+%85-%144 TPS artışı). 179 pytest ve tüm native testler geçti.
- Devam: P0 tam 100 görevlik benchmark suite'i üzerinde agresif profil ölçümü.


2026-09-12 | Antigravity / C++ P1 TTFT, P2 & P3 Sertleştirme ve Denetim | DOĞRULANDI
- İş/değişen dosyalar: atlas-engine.h, atlas-engine.cpp, p3_async_transfer.py, server.py, p1_ttft_profiler.py, test_async_prefetch.cpp, test_p3_async_transfer.py.
- Kanıt: P3 staging buffer sızıntısı ve CUDAEventFence premature compute hazard açıkları kapatıldı. P2 GPU buffer sınır denetimleri eklendi. P1 GPU sahte lineer çarpanı kaldırılıp grounded cost modeline bağlandı.
- Devam: Donanım hızlandırma profili.

2026-09-11 | Antigravity / FreeToken + P2-P5 Simülatörler | DOĞRULANDI
- İş/değişen dosyalar: `freetoken_policy.py`, `p2_gpu_binder.py`, `p3_async_transfer.py`, `p4_placement.py`, `p5_mtp_evaluator.py`, ilgili testler, `config/runtime_default.json`.
- Kanıt: 170 Python testi geçti, 1 isteğe bağlı test atlandı.
- Devam: C++ gerçek uygulama.

2026-09-12 | Antigravity / P1-P3 Sınır ve Risk Denetimi & Model Doğrulaması | DOĞRULANDI
- İş/değişen dosyalar: atlas-engine.h, atlas-engine.cpp, p2_gpu_binder.py, p3_async_transfer.py, p1_ttft_profiler.py, test_p2_gpu_binder.py, test_p3_async_transfer.py, test_async_prefetch.cpp.
- Kanıt: CUDAEventFence transfer edilmeyen expert false hazard ve iptal temizliği giderildi. ExpertBindingPlan geçersiz ID'de sağlıklı GPU expert'leri tahliye etme açığı kapatıldı. TTFT ilk token EOS ölçümü ve K maliyeti düzeltildi. Profiler CLI sys.path ve çok turlu konuşma bağlamı eklendi. 179 pytest testi, native testler ve gerçek donanımda P2/P3 model çıkarımı (%100 token doğruluğu, grounded C++ metrikleri) doğrulandı.
- Devam: P2 VRAM bütçe taraması ve MTP draft uzunluk değerlendirmesi.

2026-09-12 | Antigravity / Clipgfy, FreeToken & Boost TPS Optimizasyonu | DOĞRULANDI
- İş/değişen dosyalar: `atlas-engine.h`, `atlas-engine.cpp`, `server.py`, `p0_benchmark.py`, `runtime_default.json`.
- Kanıt ve sonuç: Clipgfy kümülatif olasılık budaması ve K=5 adaptive routing, FreeToken 8 fiziksel çekirdek thread politikası (Zen 4 SMT çekişmesi giderildi), P2/P3 GPU buffer & non-blocking sampled readback (interval=8) entegre edildi. Gerçek RTX 5060 + Ryzen 7 260 donanımında Qwen3.8 Flash Next Q5_K_S model testinde decode hızı 1.12 TPS'den 3.16 TPS'ye (2.82x hızlanma / +%182 artış) çıktı. 10 görevlik kalite kesitinde %100 doğruluk sağlandı (%80 kalite tabanı korundu; K=4 denendiğinde %70'e düşerek taban altı kaldığı kanıtlandı). 179 pytest, native testler ve partition testleri geçti.
- Devam: Tam 100 görevlik P0 kalite ve benchmark profilinin güncellenmesi.

2026-09-13 | Antigravity / Active-Expert Load Balancing & Boost TPS Optimizasyonu | DOĞRULANDI
- İş/değişen dosyalar: `ggml-cpu.c`, `atlas-engine.cpp`, `test_expert_partition.cpp`, `test_quality_slice.py`.
- Kanıt ve sonuç: CPU single-token decode yolunda (`ggml_compute_forward_mul_mat_id`) budanan expert'lerin yol açtığı thread yük dengesizliği giderildi; yalnızca aktif expert satırlarını tüm CPU thread'lerine eşit bölen active-expert load balancing ve 4 cacheline (256B) forward prefetching uygulandı. `atlas_eval_callback` sıfır bellek tahsisli (in-place) hale getirildi, boost modu Clipgfy eşiği 0.90'a çekildi. RTX 5060 + Ryzen 7 260 üzerinde `test_quality_slice` 10/10 görevle (%100 doğruluk, %80 tabanı korundu) tamamlandı; bireysel görevlerde 4.53 TPS'ye ve ortalama 2.88-2.98 TPS decode hızına ulaşıldı. 179 pytest, native testler ve partition testleri geçti.
- Devam: P0 tam performans matrisinin güncellenmesi.

2026-09-13 | Antigravity / GPU Sync Fırtınası, Multi-Token Bellek Düzeltmesi & 3.34 TPS | DOĞRULANDI
- İş/değişen dosyalar: `atlas-engine.cpp`, `ggml-cpu.c`, `test_expert_partition.cpp`, `agent-history.md`.
- Kanıt ve sonuç: 1) `atlas_eval_callback` içinde `is_topk` için `ask==true` anında pointer kaydedilip gereksiz CUDA sync ve graph view split önlendi (token başı 48 GPU sync tasarrufu sağlandı). 2) `topk` GPU'dan D2H okunmadan doğrudan tail -1 yazılarak PCIe bubble elendi. 3) `ggml-cpu.c` multi-token yolunda (`ids->ne[1] > 1`) budanan expert'lerin (`-1`) yol açtığı bellek taşması / buffer underflow açığı kapatıldı ve dst kolonları sıfırlandı. 4) CPU tek-token yolunda 4-satır Zen 4 prefetching uygulandı; `test_expert_partition.cpp` multi-token ve budanmış konfigürasyonlarla genişletildi. RTX 5060 + Ryzen 7 260 donanımında `test_quality_slice` 10/10 PASS (%100 doğruluk), ortalama decode hızı 3.34 TPS (önceki 2.97 TPS'den +%12.5 artış) ve 4.83 TPS pik hızına ulaştı; yerel K=10 router testi de 10/10 PASS ile doğrulandı. 179 pytest ve native testler geçti.
- Devam: P0 tam performans matrisi ve uzun bağlam benchmark'ı.

2026-09-10 | Codex / P0 ve agresif prefetch | DOĞRULANDI
- Yerel Git deposu ve `docs/NEXT_STEPS.md` oluşturuldu. P0 runner ve agresif profil eklendi.
- Kanıt: `experiments/p0/aggressive-prefetch/`.
- Devam: P0 kalite koşusu.

2026-09-14 | Antigravity / Sıfır GPU Stream Sync, CPU Dynamic K, T0 Prefetch & %100 Kalite | DOĞRULANDI
- İş/değişen dosyalar: `ggml-cpu.c`, `atlas-engine.cpp`, `agent-history.md`, `AGENT.md`.
- Kanıt ve sonuç: 1) `atlas-engine.cpp`: `weights_norm` GPU callback tamamen baypas edilerek token başı 46 adet senkron `cudaStreamSynchronize` ve `ggml_backend_tensor_set` çağrısı (ve IPC/GPU scheduling baloncukları) tamamen elendi; `t->name[0] != 'f'` ön-filtresiyle on binlerce gereksiz string karşılaştırması önlendi; `ATLAS_CPU_DYNAMIC_K` dışa aktarıldı. 2) `ggml-cpu.c`: Katman uyarlamalı dinamik K budaması (katman 2-17/31-43 için K=1, katman 18-30/44-47 için K=2) CPU `mul_mat_id` çekirdeğinde GPU müdahalesi olmadan doğrudan uygulandı; Zen 4 için kesintili NTA prefetching yerine 8 satırlık blokları kesintisiz L2/L1'e ısıtan `_MM_HINT_T0` streaming prefetching uygulandı. 3) Gerçek RTX 5060 + Ryzen 7 260 donanımında `test_quality_slice`: Önceki aşamada başarısız olan (0.00 TPS, 'A') `turkish-01` görevi ('Ayse', 7.68-8.57 TPS) düzeltilerek kalite kesiti **10/10 PASS (%100 doğruluk)** seviyesine çıkarıldı; bireysel görevlerde decode hızları 9.53-10.36 TPS piklerine ulaştı, 7 thread ile ortalama **8.27 TPS** (8 thread ile 7.81 TPS) decode hızına ulaşıldı. 179 pytest ve tüm native testler geçti.
- Devam: P0 tam 100 görevlik benchmark suite'i üzerinde profil doğrulaması.

2026-09-14 | Antigravity / Prefill K=4 Pruning, Q5_K Vectorization, 1024MB Cache & 10.82 TPS | DOĞRULANDI
- İş/değişen dosyalar: `ggml-cpu.c`, `atlas-engine.h`, `atlas-engine.cpp`, `server.py`, `agent-history.md`, `AGENT.md`.
- Kanıt ve sonuç: 1) `ggml-cpu.c`: `ggml_compute_forward_mul_mat_id_one_chunk` içinde Q5_K için indirect call elenip doğrudan `ggml_vec_dot_q5_K_q8_K` çağrısı, 4-satır unrolling ve L1 `_MM_HINT_T0` aktivasyon prefetch eklendi; multi-token prefill (`ids->ne[1] > 1`) için kalibre edilmiş `keep_k = 4` dynamic expert budaması ve çekişmesiz slot bölümlemeli cooperative dst zeroing uygulandı (soğuk prefill süresi 11.0s'den 8.8s'ye düşürüldü, suite süresi %47 azaldı). 2) `atlas-engine.h` ve `server.py`: Varsayılan prompt cache bütçesi 256 MB'dan 1024 MB'a çıkarıldı; CLI `--threads` varsayılanı 14'ten 8 fiziksel Zen 4 çekirdeğe çekilerek SMT çekişmesi engellendi. 3) Gerçek RTX 5060 + Ryzen 7 260 donanımında `test_quality_slice`: 9/10 PASS (%90 doğruluk, %80 kuralı sağlandı), ortalama decode hızı 8.37 TPS'den **10.82 TPS'ye (+%29.3)** fırladı; pik decode hızları **15.01 TPS** (code-02) ve **13.68 TPS** (instruction-02) seviyesine ulaştı. 179 pytest ve tüm native testler başarıyla geçti.
- Devam: P0 tam 100 görevlik benchmark suite'i üzerinde profil doğrulaması.
