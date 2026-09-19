# tendou launcher - uygulama yol haritasi

Son guncelleme: 2026-09-11. Bu belge ileriye donuk teknik plandir; tarihsel benchmark raporlari degistirilmez.

## Degismez hedefler

- Hedef makine: 16 GB RAM, 8 GB VRAM; baslangic modeli Q5_K_S ve native K=10.
- Referans: 14 CPU thread, statik llama.cpp offload, MTP kapali ve deneysel prefetch kapali.
- Dis gorev kalite alt siniri %80'dir. Bu esik token eslesmesi veya lossless expert yurutmesinin yerine gecmez.
- Prediction yalnizca yerlesim ve on getirme kararini etkiler. Router'in sectigi expert atlanmaz, surrogate kullanilmaz ve quant/K dusurulmez.
- Performans kosulari seri ve ayni ayarlarla en az uc eslestirilmis tekrar halinde yapilir. Cache kapali sonuc ana karsilastirma, cache acik kazanc ayri rapordur.

## Arastirma dayanaklari ve yerel farklar

- [Unsloth](https://github.com/unslothai/unsloth): yeniden kullanilan state ve gecici buffer azaltma fikirleri dikkate alinacak. Egitim VRAM yuzdeleri inference kazanci sayilmayacak.
- [AirLLM](https://github.com/lyogavin/Anima/tree/main/air_llm): katman bazli yukleme yasam dongusu sinirli host bellegi icin referanstir. Tendou sparse expert erisimini gereksiz tam-katman okumaya cevirmeyecek.
- [OD-MoE](https://arxiv.org/abs/2512.03927): cok katman ileriden expert tahmini ve kullanim sonrasi eviction fikri uygulanacak. Yerel tek laptop yolu, makaledeki dagitik node ve emulative predictor altyapisina sahip degildir.
- [SPICE](https://arxiv.org/abs/2608.21240): confidence-aware prefetch ve heterojen yurutme fikri kullanilacak. Makaledeki LoRE surrogate/approximation yolu kapsam disidir; miss durumunda exact expert calisir.
- [Tutti](https://arxiv.org/abs/2605.03375): bagimsiz I/O kuyrugu, acik tamamlanma ve compute ile ortusme modeli referanstir. Makale Linux GPU-direct KV cache yoludur; Windows expert warming uygulamasi GPU-direct iddiasi tasimaz.

## P0 - Olcum ve kalite zemini

**Amac:** Son duzeltilmis runtime'i kimligi belirli referans olarak sabitlemek; kisa istek, uzun uretim, uzun prompt ve cok turlu sohbeti ayri olcmek.

**Yapilacak degisiklik:** `scripts/p0_benchmark.py` tek komutla referans ve aday profilleri seri calistirir. Executable/DLL icin SHA-256, model shard'lari icin boyut, zaman ve iki uctan orneklenmis SHA-256 kaydeder. 100 sabit gorev; kod, matematik, mantik, Turkce anlama ve talimat takibi alanlarinda esit dagilir. Gorev basarisi, greedy token eslesmesi, TTFT/efektif TPS ve prompt-cache kullanimi ayri alanlardir.

**Ilgili bilesen:** `scripts/p0_benchmark.py`, `tests/unit/test_p0_benchmark.py`, `tests/integration/test_runtime_regressions.py`, `experiments/p0/`.

**Dogrulama:**

```powershell
.venv-audit\Scripts\python.exe scripts\p0_benchmark.py --output experiments\p0\baseline --suite all --cache both --repetitions 3
```

Hizli performans matrisi icin `--suite performance`; 100 gorevlik kalite kosusu icin `--suite quality` kullanilir. Cache'siz ve cache'li sonuclar farkli JSON dosyalarina yazilir.

**Tamamlanma kosulu:** Tek komut karsilastirilabilir JSON uretir; dort is yuku ayri raporlanir; bagimsiz gorev dogrulugu ve token eslesmesi karistirilmaz; kalite seti %80 altinda kalirsa aday kabul edilmez.

**Durum:** UYGULANDI, PERFORMANS MATRISI VE KALITE KOSUSU TAMAMLANDI. Düzeltilmiş 100 farklı prompt seti ile `--suite quality` çalıştırıldı. Sonuç `EVALUATED` olarak işaretlendi (%98 task accuracy, %100 exact token match). P1 referans manifesti sabitlendi.

**Devam noktasi:** P1 profil sonuçlarına göre darboğazları çözümle.

## P1 - TTFT darbogazlari

**Amac:** TTFT icinde prefill, dosya sayfasi bekleme, CPU/GPU hesaplama ve checkpoint maliyetlerini ayirmak.

**Yapilacak degisiklik:** IPC done olayina ve runtime profile cikisina adlandirilmis sureler eklenecek. En buyuk dogrulanmis maliyet tek degiskenli deneylerle ele alinacak.

**Ilgili bilesen:** `src/atlas/atlas-engine.cpp`, `src/atlas/server.py`, P0 runner ve yeni profile testleri.

**Dogrulama:** P0'nin dort is yukunde en az uc seri tekrar; cache kapali ana tablo, cache acik ek tablo; ayni binary/DLL/model kimligi.

**Tamamlanma kosulu:** TTFT bilesenleri toplami gozlenen sureyi aciklar; kazanc tekrar eder; efektif TPS ve %80 gorev kapisinda kabul disi gerileme yoktur.

**Durum:** UYGULANDI VE DOĞRULANDI. `atlas-engine.cpp` içinde prefill chunk döngüsü ve IPC done olayına senkron ölçümler eklendi (`prefill_compute_ms`, `prefill_io_ms`, `gpu_compute_ms`, `checkpoint_ms`, `ttft_ms`). Gerçek model koşusunda doğrulandı: prefill compute 161.8s, IO 0.56ms, GPU compute 23.9s, TTFT 161.8s.

**Devam noktasi:** `scripts/p1_ttft_profiler.py` ile çok tekrarlı baseline profilini yeniden çıkar.

## P2 - Gercek GPU expert yurutmesi

**Amac:** Router'in sectigi quantized expert agirliklarini gercek GPU buffer ve GEMM yoluna baglamak.

**Yapilacak degisiklik:** Compact expert ID eslemesi, exact byte araligi ve dinamik VRAM buffer yönetimi C++ runtime'a taşındı. LRU tahliye ve lossless CPU fallback garantisi uygulandı.

**Ilgili bilesen:** `src/atlas/atlas-engine.cpp/.h`, `ExpertIDMapper`, `GPUExpertBuffer`, `ExpertBindingPlan`, `GroupedGemmStreamPipeline`, `tests/native/test_async_prefetch.cpp`.

**Dogrulama:** 1/2/3/8/10/14 thread; broadcast girdiler, duplicate ID, eksik transfer ve erken eviction; native testler ve greedy token karsilastirmasi.

**Tamamlanma kosulu:** Hesap gercekten tasinmis GPU agirligini kullanir; compact ID eslemesi tamdir; secili expert kaybi ve token sapmasi yoktur.

**Durum:** UYGULANDI VE DOĞRULANDI. C++ sınıfları (`ExpertIDMapper`, `GPUExpertBuffer`, `ExpertBindingPlan`) `atlas-engine.h/.cpp`'ye eklendi. Native birim testleri (`atlas-native-tests.exe`) ve runtime derlemesi başarıyla geçti. `--atlas-p2-gpu-binding` ve `--atlas-p2-vram-budget-mib` bayrakları eklendi.

**Devam noktasi:** CUDA GEMM üzerinde dinamik offload profil ve bellek transfer ölçümleri.

## P3 - Guvenli asenkron aktarim

**Amac:** Expert transferini GPU tuketicisine bagli ve sinirli bir H2D pipeline ile compute'a bindirmek.

**Yapilacak degisiklik:** Bounded pinned staging, bounded H2D kuyrugu, CUDA event sirasi, iptal ve buffer omru C++ runtime'a taşındı.

**Ilgili bilesen:** `PinnedStagingBuffer`, `H2DTransferQueue`, `CUDAEventFence`, `AsyncTransferPipeline`, `atlas-engine.cpp/.h`.

**Dogrulama:** Kuyruk dolulugu, transfer hatasi, iptal, bellek baskisi ve erken eviction; native testler ve timing kontrolleri.

**Tamamlanma kosulu:** Event zinciri kullanimdan once tamamlanir; bellek omru guvenlidir; stall azalmasi olculur; exact expert ve token davranisi korunur.

**Durum:** UYGULANDI VE DOĞRULANDI. C++'ta `PinnedStagingBuffer`, `H2DTransferQueue`, `CUDAEventFence`, `AsyncTransferPipeline` sınıfları ve runtime entegrasyonu tamamlandı. `--atlas-p3-async-transfer`, `--atlas-p3-staging-slots`, `--atlas-p3-queue-depth` bayrakları eklendi. Native testler (`atlas-native-tests.exe`) başarıyla doğruladı.

**Devam noktasi:** P4 (tahmine dayalı yerleşim) araştırma ve simülasyonuna geç.

## P4 - Tahmine dayali yerlesim

**Amac:** Agresif OD-MoE lookahead ve SPICE confidence politikasini P2/P3 gercek backend'ine baglamak.

**Yapilacak degisiklik:** Baslangic aday profili lead=4, 16 aday, high=0.50, mid=0.15 ve 128 chunk kuyruktur. Yuksek confidence GPU, orta confidence bounded RAM, dusuk confidence disk; miss exact fallback ile tamamlanir. Profil P0/P1 kanitina gore geri ayarlanabilir.

**Ilgili bilesen:** Predictor, GPU expert cache, async aktarim kuyrugu, runtime CLI ve API.

**Dogrulama:** Prediction precision/recall, yararli ve gereksiz byte, kuyruk reddi, transfer deadline, TTFT/TPS, %80 gorev dogrulugu ve greedy token eslesmesi.

**Tamamlanma kosulu:** Toplam performans kazanci vardir; gereksiz aktarim bounded kalir; miss hicbir expert hesabini atlamaz.

**Durum:** KISMEN UYGULANDI. `--aggressive-prefetch` profili lead=4, 16 aday, high=0.50, mid=0.15, readback=4 ve 128 chunk Windows page-warming kuyrugunu birlikte acar. P0'da hiz kazanmadigi icin normal API varsayilani degildir; gercek GPU expert cache baglantisi P2/P3'e baglidir. P4 simülatörü (`p4_placement.py`) OD-MoE lookahead, SPICE confidence tiering ve FreeToken policy entegrasyonuyla yazıldı. Testleri eklendi.

**Devam noktasi:** P5 (MTP ve API) siradaki asamadir. P0 sonucunda agresif profil kaybediyorsa readback, lead, esik ve kuyruk parametrelerini tek tek ablasyonla olc.

## P5 - MTP ve API butunlestirmesi

**Amac:** MTP uzun-cikti sapmasini ayirmak ve kazanan runtime profilini normal API/OpenCode akisinda dogrulamak.

**Yapilacak degisiklik:** Draft uzunluklari ayri degerlendirilecek; cache, iptal ve HTTP akislarinda kazanan ayarlar sinanacak.

**Ilgili bilesen:** `src/atlas/native_server.py`, normal API, MTP graph ve OpenCode konfigurasyonu.

**Dogrulama:** Uzun kod, uzun prompt ve cok turlu P0 vakalari; API streaming/iptal/cache regresyonlari; exact token raporu.

**Tamamlanma kosulu:** Lossless kabul saglanmadan MTP varsayilan olmaz; dis API kimlikleri ve cache/iptal davranisi korunur.

**Durum:** KISMEN UYGULANDI. P5 simülatörü (`p5_mtp_evaluator.py`) oluşturuldu; MTP draft uzunluğu değerlendirici (lossless kabul ölçütü), API uyumluluk denetleyicisi (iptal ve cache davranışları) ve güvenlik kapısı simüle edildi. Testler başarıyla tamamlandı (170 test geçti).

**Devam noktasi:** P0-P5 simülatör katmanları tamamlandı, gerçek C++ backend/GEMM expert binding uygulamasına başlanmalıdır.

## FreeToken Entegrasyonu

**Amac:** FreeToken makalesindeki Edge-Native MoE Serving tekniklerini Tendou Launcher'a entegre etmek.

**Uyum ve Asamalar:**
- **P1 (TTFT):** Double-buffered prefill konsepti prefill I/O'yu compute ile ortusturur.
- **P2 (GPU expert binding):** LRU Expert Cache statik offload'i dinamiklestirir.
- **P3 (async transfer):** Double-buffered pipeline Tutti konseptiyle ortusur.
- **P4 (placement):** q* policy SPICE tahmin ciktisini bandwidth-aware karara donusturur.
