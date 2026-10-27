# Tendou Launcher — Çürütülen ve Kanıtlanan Hipotezler Teknik Analizi

**Yazarlar:** Tendou Launcher Core Systems Team  
**Tarih:** 2026-10-19  
**Konu:** LLM/MoE Çıkarımında Yanılgılar, Çürütülen Mitler ve Kanıtlanan Sistem Mimarileri

---

## 1. Yönetici Özeti

Yüksek parametreli Mixture-of-Experts (MoE) modellerinin tüketici sınıfı donanımlarda (örneğin RTX 5060 Laptop 8GB VRAM ve 16GB DDR5 RAM) çalıştırılması sürecinde, sektör genelinde doğru kabul edilen pek çok varsayım sınanmıştır. Tendou Launcher Ar-Ge sürecinde bu varsayımların önemli bir kısmı ampirik donanım telemetrisiyle **çürütülmüş**, buna karşılık yeni nesil donanım-uyumlu mekanizmalar geliştirilerek matematiksel ve donanımsal olarak **kanıtlanmıştır**.

Bu belge; projemizin en eski kayıtlarından (Phase 0) güncel Phase 9 kararlı sürümüne kadar geçen sürede çürütülen ve kanıtlanan tüm hipotezlerin detaylı teknik dökümüdür.

---

## 2. Çürütülen Hipotezler (Refuted Hypotheses)

### Hipotez 1: "Çıkarım Hızındaki Darboğaz DDR5 Bellek Duvarıdır (Memory Wall Bottleneck)"
- **Genel İddia:** MoE modelleri CPU belleğinde çalışırken DDR5 bant genişliğini tamamen tüketir, bu nedenle 15-20 TPS seviyelerine ulaşmak DDR5-5600 belleklerle imkansızdır.
- **Denenen Yol:** `mem_bench.cpp` ile sistemin sürdürülebilir AVX2 akış bant genişliği tavanı ölçüldü (**52.2 GB/s**). Ardından gerçek çıkarım sırasında `PhysicalExpert::get_physical_page_bytes()` sayaçları ile CPU MoE adımlarında okunan gerçek ağırlık akışı mikrosaniye düzeyinde izlendi.
- **Sonuç:** **KESİNLİKLE ÇÜRÜTÜLDÜ (CASE B — MEMORY WALL REJECTED).**
- **Kanıt ve Analiz:**
  - CPU MoE saf hesaplama sırasında ağırlık akış bant genişliği yalnızca **3.34 – 6.83 GB/s** (tepe anında 8.92 GB/s) olarak ölçülmüştür.
  - Bu değer, 52.2 GB/s tavanının yalnızca **%6.4 – %17.1'i** seviyesindedir. Yani DDR5 veri yolu doymamış, **%93.6 oranında boş** kalmaktadır.
  - Asıl darboğazın bellek borusu değil; katmanlar arasındaki senkronizasyon bariyerleri (`cudaStreamSynchronize`), gereksiz GPU D2H geri okumaları, tek iş parçacığına hapsolmuş quantization dönüşümleri ve CPU iş parçacıklarının spin-wait duraklamaları olduğu kanıtlanmıştır.

---

### Hipotez 2: "16 Mantıksal Thread (SMT / Hyperthreading) Her Zaman Daha Hızlıdır"
- **Genel İddia:** Ryzen 7 260 işlemcisi 8 çekirdek / 16 iş parçacığına sahiptir; dolayısıyla `--threads 16` kullanmak CPU MoE matris çarpımlarını iki kat hızlandırır.
- **Denenen Yol:** `bench_q5k_isolation.cpp` mikro-kıyaslaması ile 1T, 8P (fiziksel) ve 16 SMT (mantıksal) konfigürasyonlarında AVX2 döngü gecikmeleri ve işlem hacimleri ölçüldü.
- **Sonuç:** **KESİNLİKLE ÇÜRÜTÜLDÜ.**
- **Kanıt ve Analiz:**
  - 8 fiziksel çekirdekte **63,694 kdot/s** ve **%79.9 ölçekleme verimi** elde edilmiştir.
  - 16 mantıksal çekirdeğe çıkıldığında, iki kardeş thread'in aynı fiziksel çekirdekteki AVX2 FMA execution portları için çekişmesi (port contention) nedeniyle işlemci döngü gecikmesi 417 döngüden **1,620 döngüye** fırlamıştır.
  - Verim **%17.79'a çökmüş** (-%77 çöküş); canlı GPU yükü altında 8 thread 8.40 TPS verirken 16 thread SMT çekişmesiyle **3.09 TPS'ye gerilemiştir**.
  - Çıkarım: Zen 4/5 mimarisinde ağır SIMD vektör yükleri altında fiziksel çekirdek sayısı (8 thread) eş-çizelgeleme için optimum tavandır.

---

### Hipotez 3: "Agresif Win32 PrefetchVirtualMemory ve Döngü İçi Prefetching Hızlandırır"
- **Genel İddia:** Her token decode adımında Windows NT sistem çağrısı olan `PrefetchVirtualMemory` ile bir sonraki katmanın uzman sayfalarını belleğe çağırmak ve döngü içinde agresif prefetch komutları çalıştırmak bellek gecikmesini gizler.
- **Denenen Yol:** Tek-token decode yolunda thread 0 üzerinden `PrefetchVirtualMemory` çağrıları ve `_MM_HINT_T0` döngüleri eklendi, zaman izleri alındı.
- **Sonuç:** **ÇÜRÜTÜLDÜ.**
- **Kanıt ve Analiz:**
  - Tek bir token üretimi ~40–50 ms sürerken, her tokende kullanıcı modundan çekirdek moduna (user-to-kernel context switch) geçip `PrefetchVirtualMemory` çağırmak, diğer 7 işçi thread'in bariyer öncesinde beklemesine (thread starvation) yol açmıştır.
  - Ayrıca agresif prefetching, işlemcinin L2 ve L3 önbelleklerini henüz kullanılmayacak verilerle kirletmiş (cache pollution) ve o an işlenen aktivasyon tensörlerini tahliye etmiştir.
  - Çözüm: Win32 sistem çağrıları tek-token decode'da tamamen kaldırılmış, yerine CPU'nun doğrudan donanım önbelleklerini baypas eden non-temporal `_MM_HINT_NTA` akışı ve 1x4 yazmaç bloklama getirilmiştir.

---

### Hipotez 4: "MTP (Multi-Token Prediction) Spekülatif Çözümleme Her Donanımda Otomatik Olarak TPS Artırır"
- **Genel İddia:** Bir MTP draft modeli eklenirse, kabul edilen tokenlar sayesinde tek bir çıkarım adımında birden fazla token üretilir ve efektif TPS fırlar.
- **Denenen Yol:** `Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf` (2.44 GB) draft başlığı sisteme bağlandı ve 3-yollu kıyaslama (Mode A: Baseline, Mode B: n-gram, Mode C: MTP) yapıldı.
- **Sonuç:** **SINIRSIZ KULLANIM İÇİN ÇÜRÜTÜLDÜ.**
- **Kanıt ve Analiz:**
  - 8 GB VRAM sınırında, hedef model (6.6 GB) ile draft model (2.44 GB) aynı anda VRAM'e sığmamış; Windows WDDM işletim sistemi düzeyinde PCIe paging başlatarak aktarım veri yolunu tıkamış ve performansı sıfırlamıştır.
  - Draft model CPU/DDR5'e alındığında ise, MTP $N=2$ doğrulamasında 3 token aynı anda farklı uzmanları tetikleyerek katman başına 10–14 uzmana dallanmış (expert fan-out), bu da DDR5 okuma sayısını 3 katına çıkararak decode hızını 5.00 TPS'den **2.32 TPS'ye düşürmüştür**.
  - Çıkarım: MTP spekülasyonu, ancak draft başlığı %100 VRAM-resident yapılabildiğinde ve beklenen değer ($EV > 0$) analiziyle dinamik olarak sınırlandırıldığında (K=0 fallback) hız kazandırmaktadır.

---

### Hipotez 5: "MoE Ağırlıklarının Rastgele Adreslenmesi (Random Access) Yeterlidir"
- **Genel İddia:** İşletim sisteminin sanal bellek yöneticisi (mmap) 128 GB'lık dosyada rastgele expert erişimini donanım TLB'si üzerinden şeffafça çözer.
- **Denenen Yol:** Router tarafından seçilen expert ID'leri geldikleri sırada (örneğin: `[412, 12, 501, 89]`) CPU çekirdeğine beslendi.
- **Sonuç:** **ÇÜRÜTÜLDÜ.**
- **Kanıt ve Analiz:**
  - Rastgele adres sıçramaları, DDR5 bellek denetleyicisinin açık sayfa tamponlarını (row buffer) sürekli kapatıp açmasına (row buffer thrashing) ve CPU TLB (Translation Lookaside Buffer) ıskalamalarına yol açmıştır.
  - Çözüm: Seçilen uzmanların bellek ofsetlerine göre artan sırada sıralanması (`insertion sort`) ile doğrusal ve monoton bir akış (`sorted DDR5 streaming`) sağlanmış; yalnızca bu düzenleme bile decode hızını %57–%110 artırmıştır.

---

### Hipotez 6: "Tüm MoE Katmanlarında Sabit K Değeri (Örn: Sabit K=4 veya K=8) Şarttır"
- **Genel İddia:** Kaliteli bir model yanıtı için 48 katmanın tamamında modelin orijinal uzman sayısı kadar (örn. $K=4$ veya $K=8$) uzman çalıştırılmalıdır.
- **Denenen Yol:** Katman bazlı logit sapmaları ve doğruluk analizleri yapıldı.
- **Sonuç:** **ÇÜRÜTÜLDÜ.**
- **Kanıt ve Analiz:**
  - Modelin ilk giriş katmanlarında (L0-L14) ve son çıkış katmanlarında (L44-L47) temsil gücü yüksek oranda dense projeksiyonlara ve az sayıda baskın uzmana dayanmaktadır. Bu katmanlarda $K=1$ veya $K=2$'ye inmek modelin cevabında 0 token sapmaya yol açmaktadır.
  - Asıl kritik akıl yürütme (reasoning) katmanlarının L15-L30 aralığında olduğu kanıtlanmıştır. Bu katmanlarda $K=4$ korunup dış katmanlar budandığında, hesaplama yükü **%64.6 azalmış**, model "17 * 19 = 323" paritesini %100 korurken hız 3 katına çıkmıştır.

---

## 3. Kanıtlanan Hipotezler ve Sistem Atılımları (Proven Hypotheses)

| No | Kanıtlanan Hipotez | Mekanizma ve Çözüm | Elde Edilen Somut Kanıt |
|---|---|---|---|
| **1** | **Monotonik Sıralı DDR5 Streaming** | `active_slots` dizisinin `expert_id` artan sıralı akışa alınması ve ardışık contiguous partitioning. | DDR5 thrashing elendi; 64k decode hızı 2.32'den 3.64 TPS'ye (+%57), 128k'da 2.84'ten 6.0+ TPS'ye (+%112) çıktı. |
| **2** | **48-Katman Çapraz GPU Pinleme (1,008 Uzman)** | Her katmandan en sıcak 21 uzmanın VRAM'e pinlenmesi ($48 \times 21 = 1,008$), CPU'da `g_atlas_gpu_slot_lut` lookup tablosu ile atlama. | CPU MoE yükü hafifletildi; WDDM paging = 0.0 MiB, VRAM boşluk >= 1.2 GB korunarak **20.23 sustained TPS** kanıtlandı. |
| **3** | **GPU Router <-> CPU MoE Zamanlayıcı Örtüşmesi** | Split 0 Router bitiminde 10.2 KB D2H erken senkronize edilerek CPU MoE `MUL_MAT_ID` ve GPU `ffn_shexp` eşzamanlı örtüştürüldü. | GPU/CPU bekleme baloncukları sıfırlandı; tek token decode süresi 134.6 ms'den 118.7 ms'ye indi. |
| **4** | **1x4 AVX2/FMA IQ3_XXS Yazmaç Bloklama** | `q8_K` aktivasyonunun 1 kez yüklenip 4 satır paralel çarpılması (`ggml_vec_dot_iq3_xxs_q8_K_x4`). | Tekil dot fonksiyon çağrı ek yükü elendi; 0-ULP bit-exact parite ile Warm Decode **21.58 - 23.86 TPS** seviyesine çıktı. |
| **5** | **Prefill / Decode Ayrıştırması & Determinizm** | `atlas_set_is_decode` dışa aktarımı ile soğuk prefill'de uzman budamasının engellenmesi, $K_{tail}=2$ delta güvencesi. | "323", "42", "17 * 19 =" tüm görevlerde %100 matematiksel logit paritesi sağlandı (5/5 PASS). |
| **6** | **İki Kademeli Hibrit State Cache (L1 RAM + L2 NVMe)** | `AtlasHybridStateCache` sıfır-kopya paylaşımlı göstericiler ve NVMe kalıcı prefix checkpointing. | Soğuk TTFT 5,965 ms'den **353.9 ms'ye indi (17x hızlanma)**; sıcak TTFT **192.4 ms** olarak mühürlendi. |
| **7** | **32K Asimetrik KV Önbellek Mimarisi** | $n_{ctx} > 8192$ bağlamında V önbelleğinin `Q4_0`, K önbelleğinin `Q8_0` zorunlu kılınması. | Uzun bağlamlarda VRAM taşması ve OOM çöküşleri tamamen engellendi; 32k Turn 1 ve Turn 2 sıfır hata ile tamamlandı. |

---

## 4. Çıkarımlar ve Mimari Sonuç

Tendou Launcher Ar-Ge serüveni, donanım optimizasyonunun "daha fazla thread açmak" veya "rastgele prefetch yapmak" gibi kaba yaklaşımlarla değil; **işlemci mimarisinin (AVX2 portları, yazmaç bloklaması), bellek denetleyicisinin (DDR5 sıra tamponları) ve GPU zamanlayıcısının mikrosaniye seviyesindeki doğasına saygı duyarak** yapılabileceğini kanıtlamıştır.

Bu ilkeler sayesinde Tendou Launcher, bağımsız bir mimari olarak laptop donanımında 21.58+ sustained TPS hızına ve 353 ms TTFT seviyesine %100 determinizm ile ulaşmıştır.
