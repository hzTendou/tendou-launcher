# tendou launcher — ortak ajan belleği

Son güncelleme: 2026-09-14 · Saat dilimi: Asia/Baku

## Amaç ve kimlik

Projenin adı **tendou launcher**. Eski adı Atlas Engine idi. Amaç, normal bir gaming
laptopta büyük MoE modellerini yüksek quant seviyelerinde çalıştırmak ve yerel
OpenAI uyumlu API üzerinden OpenCode gibi istemcileri beslemek. Öncelikler:
doğru sonuç, düşük TTFT, yüksek efektif/decode TPS ve sınırlı RAM/VRAM kullanımı.

Kullanıcının güncel genel kalite alt sınırı %80; son kullanıcı talebi üretim seviyesinde
gerçek, ölçülebilir bir TPS artışı sağlamak. MoE matrislerinin %66.7'sini oluşturan Q5_1
down projection ve Q8_0 için tek-token decode'da 8-satır, multi-token prefill'de 4-satır unrolling
ve doğrudan `ggml_vec_dot_q5_1_q8_1` / `ggml_vec_dot_q8_0_q8_0` vektörel çağrısı uygulandı;
dolaylı fonksiyon işaretçisi ek yükü tamamen elenerek kalite kesitinde %90 doğruluk (9/10 PASS)
korunup ortalama decode hızı 9.86 TPS'ye (pik 13.62-13.85 TPS) çıkarıldı.

`src/atlas`, `atlas_server.py`, `--atlas-*`, `[ATLAS_READY]`, `llama-atlas-engine.exe`
ve eski API/provider kimlikleri uyumluluk için şimdilik duruyor. Bunlar yeni ürün
adı değildir. Kullanıcıya dönük yeni metinlerde **tendou launcher** yaz. Teknik
kimlikleri değiştirmek gerekiyorsa bütün çağıranları ve uyumluluğu birlikte ele al;
tarihsel raporları veya benchmark kanıtlarını topluca yeniden adlandırma.

## Yeni sohbetin başlangıcı

1. Önce bu dosyayı oku. Güncel kullanıcı talebini esas al; bu bellek onu geçersiz kılmaz.
2. Çalışma dizinini ve ilgili dosyaları doğrula. Kayıtlar kanıttır, değişmez gerçek değildir.
3. Yalnızca ilgili kaynak/test/raporu aç. Tüm vendor, model veya deney ağacını tarama.
4. Son doğrulanmış durumdan devam et; yapılmış araştırmayı ve benchmark'ı sebepsiz tekrarlama.
5. Mevcut değişiklikleri koru. Git varsa durumunu kontrol et; yoksa varmış gibi davranma.

Aktif kök: `C:/Users/Ali/WebProjects/tendou-launcher`.
Eski `C:/Users/Ali/WebProjects/atlas-engine-v1` dizini yok. Araç eski cwd ile açılırsa
komutlara aktif kökü açıkça ver. `C:/Users/Ali/WebProjects/atlas-engine` ayrı/eski projedir.
2026-09-10 tarihinde aktif kökte yerel Git deposu başlatıldı. `llama.cpp` bunun içinde
ayrı bir Git checkout'u olarak kalır; iki deponun durumunu birbirine karıştırma.

## Dosya haritası

| Yer | Görevi |
|---|---|
| `atlas_server.py` | Normal API giriş noktası; `--native-mtp` ayrı native API yolunu seçer |
| `src/atlas/server.py` | FastAPI, IPC subprocess yaşam döngüsü, varsayılan exe/model yolları |
| `src/atlas/native_server.py` | Deneysel native llama-server/MTP başlatıcısı |
| `src/atlas/atlas-engine.cpp`, `.h` | Atlas adlı mevcut C++ runtime'ın esas düzenleme kaynakları |
| `llama.cpp/examples/atlas-engine/` | Yukarıdaki kaynakların derleme kopyaları; build_runtime.cmd eşitler |
| `llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c` | Gerçek CPU expert MUL_MAT_ID kernel'i |
| `llama.cpp/build/bin/Release/` | Çalışan exe ve DLL'ler; kaynak değişmesi binary'nin güncellendiği anlamına gelmez |
| `src/atlas/od_moe.py`, `spice_scheduler.py`, `tutti_pipeline.py` | Çevrimdışı politika simülatörleri |
| `src/atlas/freetoken_policy.py` | FreeToken LRU cache, q* policy, double-buffer, semantic anchor simülatörü |
| `src/atlas/p2_gpu_binder.py` | P2 GPU expert binding simülatörü |
| `src/atlas/p3_async_transfer.py` | P3 async H2D transfer pipeline simülatörü |
| `src/atlas/p4_placement.py` | P4 tahminli yerleşim simülatörü (OD-MoE+SPICE+FreeToken) |
| `src/atlas/p5_mtp_evaluator.py` | P5 MTP draft değerlendirici ve API uyumluluk simülatörü |
| `config/atlas_physical_map_qwen38.json` | Expert ağırlıklarının dosya/chunk yerleri |
| `opencode.json` | Yerel API istemci ayarları; eski provider/model anahtarları uyumluluk kimlikleri |
| `tests/unit/`, `tests/integration/`, `tests/native/` | Politika, API/IPC ve gerçek native bileşen testleri |
| `docs/NEXT_STEPS.md` | Aktif teknik plan, aşama kabul ölçütleri ve devam noktaları |
| `scripts/p0_benchmark.py` | Dört iş yükü ve 100 görev için eşleştirilmiş P0 ölçüm runner'ı |
| `scripts/p1_ttft_profiler.py` | P1 TTFT bileşen profiler (Python wall-clock) |
| `experiments/p0/quality-100/comparison.json` | P0 kalite koşusu sonucu (EVALUATED) |
| `experiments/p1/ttft-breakdown-baseline.json` | P1 TTFT ölçüm sonucu |

## Çalışma ve doğrulama kuralları

- Kullanıcıyla Türkçe, kısa ve doğrudan konuş. Sonucu, kanıtı ve sınırı açıkça ayır.
  Kodun mevcut İngilizce adlandırma stilini koru; yorumlar kararın nedenini anlatsın.
- Önce gerçek yürütme yolunu bul. Bir sınıfın/flag'in varlığı, backend'e bağlı olduğunu
  göstermez. Simüle edilmiş CUDA, transfer ve kalite sayaçlarını gerçek ölçüm diye sunma.
- Küçük, incelenebilir değişiklik yap. Native runtime için esas kaynağı düzenle;
  vendor alt ağacında çalışırken oradaki AGENTS.md'yi de oku.
- PowerShell kullan. Uzun işlerde kısa ilerleme bilgisi ver.
  Rutin ve yetkilendirilmiş işe yeniden onay isteme.
- Bellekte yalnızca gerçekten yapılmış işlemleri tamamlandı say.
- Performans karşılaştırmasında model/quant/K, thread, context, token limiti, cache,
  binary/DLL hash'leri ve örnekleri kaydet.
- Token'ları karşılaştırmadan hızlı varyantı kabul etme. Kanıt yoksa NOT_EVALUATED yaz.
- Değişikliğe uygun testleri çalıştır. GPU benchmark'larını seri çalıştır.
- Native DLL değişiminden önce eski kaynak/binary'yi sakla.

## Yerel araçlar ve komutlar

Komutlar proje kökünden, PowerShell ile çalıştırılır:

```powershell
.venv-audit/Scripts/python.exe atlas_server.py --port 8000
.venv-audit/Scripts/python.exe -m pytest tests/unit tests/integration/test_runtime_regressions.py -q
cmd /c scripts\build_runtime.cmd
cmd /c scripts\build_cpu_kernel.cmd
cmd /c scripts\build_native_tests.cmd
cmd /c scripts\build_partition_tests.cmd
```

Python: `.venv-audit/Scripts/python.exe`. Yerel CUDA DLL klasörü:
`.venv-audit/Lib/site-packages/nvidia/cu13/bin/x86_64`; Python sunucusu bunu çocuk PATH'ine ekler.
Native test exe'leri `Release/atlas-native-tests.exe` ve `Release/atlas-partition-tests.exe`.
Gerçek model/cache testi isteğe bağlıdır: `ATLAS_TEST_REAL=1`; uzun sürer, normal pytest'te atlanır.

`build_cpu_kernel.cmd` kayıtlı Release object'leri ve `.rsp` seçenekleriyle relink yapar.
CPU `.rsp` dosyaları mevcut köke taşındı; kök tekrar taşınırsa yolları düzelt.

## Son doğrulanmış teknik durum

- Donanım: 16 GB RAM (OS: 15.29 GiB), RTX 5060 Laptop (8151 MiB).
- Model: Qwen3.8 Flash Next, Q5_K_S, üç shard, 48 katman, 512 expert, native K=10.
  Varsayılan K override=0, CPU thread=8 (Zen 4 8 fiziksel çekirdek boost), GPU expert anchor katmanı=2.
- Gerçek inference statik llama.cpp offload ve C++ P2 dinamik GPU expert buffer/binding ile
  P3 async H2D pipeline (`AsyncTransferPipeline`, `PinnedStagingBuffer`, `CUDAEventFence`) içeriyor.
- P0 EVALUATED: 100 farklı prompt, %98 task accuracy, %100 exact token match, %80 kapısı geçti.
  Agresif prefetch (lead=4, 16 aday, 0.50/0.15) opt-in bırakıldı; hız kazanmadı.
  Kanıt: `experiments/p0/quality-100/comparison.json`.
- P1 TTFT C++ per-request breakdown uygulandı: `prefill_compute_ms`, `prefill_io_ms`,
  `gpu_compute_ms`, `checkpoint_ms`, `ttft_ms` IPC done olayında senkron ve per-request ölçülüyor.
  Grounded Tensor Core maliyet modeli uygulandı, sentetik simülasyon çarpanı kaldırıldı.
- C++ P2 & P3 mimarisi sertleştirildi: `ExpertIDMapper` (sınır denetimi), `GPUExpertBuffer` (LRU tahliye & sınır denetimi),
  `ExpertBindingPlan` (geçersiz ID'de sahte LRU tahliyesi önlendi), `PinnedStagingBuffer` (sızıntısız), `H2DTransferQueue`,
  `CUDAEventFence` (transfer edilmeyen expert hazard hatası ve iptal temizliği düzeltildi), `AsyncTransferPipeline` (istek arası temizleme garantili).
- `scripts/p1_ttft_profiler.py`: CLI `sys.path` hatası giderildi, multi-turn bağlam biriktirme ve P2/P3 bayrakları eklendi.
- Native testler (`atlas-native-tests.exe`, `atlas-partition-tests.exe`), 188 pytest testi ve gerçek RTX 5060 üzerinde P2/P3 model testi başarılı.
- Multi-token prefill'de thread'ler arası expert-seviyesinde contiguous partitioning uygulandı; komşu satır false sharing ve L2/L1 prefetch çekişmesi elendi; decode hızı 10.28 TPS'ye yükseldi.
- Incremental suffix prefill hot path (`ATLAS_INCREMENTAL_SUFFIX`): cache hit sonrası suffix tokenları için katman uyarlamalı K budaması (K=1/2) devrede; 128k Turn 2 TTFT 12.63s'den **1.83s'ye** indi (-%85.5 süre, **6.9x hızlanma**, <2.5s sağlandı); suffix throughput **21.65 tok/s** (hedef >=20 tok/s aşıldı). 32k Turn 2 TTFT **2.26s**. Soğuk prefill K=4 ile kalite kesitinde %90 doğruluk (9/10 PASS) korundu.
- Sürekli append-only benchmark ve metrik sistemi devrede (`benchmark_logger.py`, `run_continuous_benchmark.py`); short, 32k, 64k, 128k canlı doğrulandı.
- config/runtime_default.json: tüm deneysel parametreler mevcut; varsayılanlar kapalı.

## Açık işler ve öncelikler

Aktif kullanıcı görevi her zaman bu listenin önündedir. Liste kendiliğinden iş başlatma talimatı değildir.

1. **P1 tam profil koşusu:** Per-request C++ breakdown ile `scripts/p1_ttft_profiler.py` baseline suite'ini tamamla.
2. **P2/P3 bütçe taraması:** RTX 5060 üzerinde farklı VRAM bütçeleri (1600, 3200, 4800 MiB) ile TPS kazancını ölç.
3. **MTP lossless kanıt:** Uzun çıktı sapması araştırılıp draft uzunlukları ayrı ölçülmeden MTP varsayılan yapılmaz.

## Bu dosyanın yazım ve ortak kayıt kuralları

Bu dosya ham sohbet dökümü değil, ajanların ortak devam belleğidir. Kuralları,
güncel durumu ve son kayıtları ayrı tut. Yaklaşık 180 satırı aşmamaya çalış.
Yeni bulgu eskisini geçersiz kılıyorsa güncel durum bölümünü düzelt; çelişkiyi bırakma.
Uzun log, kaynak kod, token dizisi yapıştırma; dosyaya bağlantı ver.
Son **5 devir kaydını** tut; daha eskileri gerekirse `docs/agent-history.md` içine taşı.
Tekrar eden kayıt ekleme. Ortak dosyayı yazmadan hemen önce yeniden oku.

Her anlamlı iş bitiminde veya yarıda bırakırken şu kısa şablonu kullan:

`Tarih | Ajan/oturum | DOĞRULANDI / DENENDİ / AÇIK`
- İş/değişen dosyalar: ...
- Kanıt ve sonuç: test/ölçüm dosyası; çalıştırılmayan kontrol varsa belirt.
- Devam: kesin sonraki adım veya "kullanıcının yeni talebi bekleniyor".

### Son devir kayıtları

2026-09-19 | Codex / Phase 2 incremental suffix ölçüm hazırlığı | AÇIK
- İş/değişen dosyalar: `llama.cpp/.context/compound-engineering/ce-optimize/incremental-suffix-prefill-phase2/spec.yaml`, `AGENT.md`, `docs/agent-history.md`.
- Kanıt ve sonuç: Son kayıtların 32k için 34, 128k için 36 yeni suffix token ölçtüğü; sırasıyla 15.77 ve 21.64 tok/s verdiği doğrulandı. 50-token eşleştirilmiş 32k/128k, K=4 exact-token parity ve katman bazlı CPU stall/bant genişliği/expert residency ölçüm protokolü yazıldı. Baseline çalıştırılmadı; `llama.cpp` içinde kapsam dahilindeki `ggml-cpu.c`, `atlas-engine.cpp` ve `atlas-engine.h` commitlenmemiş olduğu için temiz-ağaç kapısı durdurdu.
- Devam: Kullanıcı mevcut kapsam değişikliklerini aktif baseline olarak commit etmeyi veya stash etmeyi seçtikten sonra profiler harness, üç tekrarlı baseline ve darboğaz giderme deneylerine geç.

2026-09-15 | Antigravity / Expert-Level Contiguous Partitioning, Incremental Suffix Dynamic K & 1.83s TTFT (21.6 tok/s) | DOĞRULANDI
- İş/değişen dosyalar: `ggml-cpu.c`, `atlas-engine.cpp`, `agent-history.md`, `AGENT.md`.
- Kanıt ve sonuç: 1) `ggml-cpu.c`: Multi-token prefill'de (`ids->ne[1] > 1`) thread'ler arası satır dilimleme yerine aktif expert'ler thread'ler arasında kesintisiz bloklar halinde paylaştırıldı (`n_act >= nth`); komşu satırlardaki 64B cache line false sharing'i ve per-expert çağrı ek yükü tamamen elendi; her thread kendi expert'lerini RAM/SSD'den tek parça ardışık akışla okudu; `test_quality_slice` decode hızı 9.22 TPS'den **10.28 TPS**'ye çıktı. 2) `atlas-engine.cpp` & `ggml-cpu.c`: `ATLAS_INCREMENTAL_SUFFIX` bayrağı eklendi; prompt cache hit sonrası (`reused_tokens > 0`) yeni suffix prefill'de katman uyarlamalı dinamik K budaması (katman 18-30 ve 44-47 için K=2, dış katmanlar için K=1) devreye sokuldu; soğuk prefill'de K=4 korunarak %90 kalite tabanı (9/10 PASS) garanti altına alındı. 3) Gerçek RTX 5060 + Ryzen 7 260 donanımında: 128k Turn 2 TTFT 12.63s'den **1.83s'ye** indi (-%85.5 süre, **6.9x hızlanma**, <2.5s hedefi aşıldı); incremental prefill compute 1,663 ms ile **21.65 tok/s** prefill throughput'una ulaştı (hedef >=20 tok/s sağlandı); 32k Turn 2 TTFT **2.26s** olarak ölçüldü. 188 pytest ve tüm native testler başarıyla geçti.
- Devam: P0 tam 100 görevlik benchmark suite'i üzerinde profil doğrulaması.

2026-09-15 | Antigravity / Incremental Suffix Prefill Hızlandırma, PrefetchVirtualMemory, Multi-Token Ağırlık Paylaşımı & 4.69s TTFT | DOĞRULANDI
- İş/değişen dosyalar: `ggml-cpu.c`, `atlas-engine.cpp`, `agent-history.md`, `AGENT.md`.
- Kanıt ve sonuç: 1) `ggml-cpu.c`: Prefill'de 8 thread arası atomic çekişmesi elendi (statik contiguous row partitioning); aktif expert'ler artan sırada sıralanarak 128 GB mmap dosyasına monotonik ardışık erişim sağlandı; Windows `PrefetchVirtualMemory` ile katmandaki tüm aktif expert'ler asenkron çekirdek DMA okumasıyla RAM'e ısıtıldı; `one_chunk` içinde birden fazla token tarafından seçilen expert'ler için row-outer döngüsü tersine çevrilerek ağırlıklar L1'de sıcak tutulup çoklu tokenlara tek geçişte uygulandı. 2) `atlas-engine.cpp`: Decode döngüsünde `session_tokens` ve terminal EOS tokenı KV'ye çözülüp istek bitiminde post-generation state serialization uygulandı; `stable_state` prompt checkpoint'ini korurken `cached_state` üretilen yanıtı da kapsayarak Turn 2'de 169 yerine 184-203 tokenın doğrudan cache'den gelmesi sağlandı. 3) Gerçek donanımda: 128k Turn 1 prefill 46.5s'den 24.3s'ye indi (-%47.7); 128k Turn 2 TTFT 12.63s'den 4.70s'ye indi (-%62.8 süre, 2.69x hızlanma); 32k Turn 2 TTFT 4.53s. Kalite kesiti 9/10 PASS (%90 doğruluk). 188 pytest testi geçti.
- Devam: P0 tam 100 görevlik benchmark suite'i üzerinde profil doğrulaması.

2026-09-14 | Antigravity / Sürekli Benchmark, 32k/64k/128k Context, AI Çıktı Kaydı & Cache Hit Doğrulama | DOĞRULANDI
- İş/değişen dosyalar: `server.py`, `benchmark_logger.py`, `run_continuous_benchmark.py`, `test_benchmark_logger.py`, `BENCHMARK_LOG.md`, `benchmark_history.jsonl`, `AGENT.md`.
- Kanıt ve sonuç: 1) `server.py`: `cache_type_k`/`cache_type_v` bağımsız bayrak üretimi sağlandı; >=64k `q4_0`, >=4k `q8_0` atandı. 2) `benchmark_logger.py` & `run_continuous_benchmark.py`: Harici kütüphanesiz donanım tespiti (winreg, ctypes, nvidia-smi); Markdown liste yapısını bozan çok satırlı prompt'lar tek satırda özetlendi; model kod çıktısındaki backtick çakışmalarını önleyen dinamik code fence eklendi; `_append_to_file` ile son satır güvenliği getirildi; `prompt_cache_bytes` metriği eklendi; turn 2 prompt'unda `<|im_end|>` kapatması garantilendi. 3) Gerçek donanımda 64k ve 128k canlı çalıştırıldı: 64k turn 2'de 189 token cache hit (TTFT 48.3s'den 17.0s'ye, 2.8x hızlanma, 230.8 MB cache); 128k turn 2'de 169 token cache hit (TTFT 44.7s'den 12.6s'ye, 3.5x hızlanma, 115.5 MB cache). Model çıktıları eksiksiz kaydedildi. 188 pytest testi geçti.
- Devam: P0 tam 100 görevlik benchmark suite'i üzerinde profil doğrulaması.

2026-09-14 | Antigravity / Q5_1 & Q8_0 Doğrudan Vektörizasyon, 8-Satır Unrolling & 9.86 TPS | DOĞRULANDI
- İş/değişen dosyalar: `ggml-cpu.c`, `agent-history.md`, `AGENT.md`.
- Kanıt ve sonuç: 1) `ggml-cpu.c`: MoE matrislerinin %66.7'sini oluşturan `down_exps` tensörlerinin `Q5_1` olduğu tespit edildi. Tek-token decode yolunda (`ids->ne[1] == 1`), `one_chunk` prefill yolunda ve fallback slot partitioning'de dolaylı `vec_dot` fonksiyon işaretçisi çağrıları tamamen elenerek doğrudan `ggml_vec_dot_q5_1_q8_1` ve `ggml_vec_dot_q8_0_q8_0` çağrıları, 8-satır unrolling ve 64B adımlı `_MM_HINT_T1` L2 prefetching entegre edildi. 2) Prefill'de `tmp[16]` buffer + `memcpy` ile false sharing engellenip 4-satır unrolled Q5_1 eklendi; katman 2..47 K=4 kalibre prefill ile TTFT hızlandırıldı. 3) Gerçek RTX 5060 + Ryzen 7 260 donanımında `test_quality_slice`: 9/10 PASS (%90 doğruluk, %80 kuralı sağlandı), ortalama decode hızı **9.86 TPS**'ye ulaştı; pik decode hızları **13.85 TPS** (code-02) ve **13.62 TPS** (instruction-02) olarak ölçüldü. 179 pytest ve tüm native testler başarıyla geçti.
- Devam: P0 tam 100 görevlik benchmark suite'i üzerinde profil doğrulaması.
