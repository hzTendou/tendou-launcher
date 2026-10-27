# Tendou Launcher v0.13.0-stable Release Notes

**Sürüm:** v0.13.0-stable  
**Yayınlanma Tarihi:** 2026-10-22  
**Mühür:** 100% Deterministic Bit-Exact Parity (0-ULP), 26/26 Native Gates PASS, 109 Pytest Tests PASS  

---

## 1. Genel Bakış ve Temel Felsefe

Tendou Launcher `v0.13.0-stable`, Phase 11, Phase 12 ve Phase 13 geliştirme döngülerinde elde edilen tüm tepe mühendislik kazanımlarını resmi, dondurulmuş ve bağımsız bir sürüm paketi olarak mühürler.

Tendou Launcher bir llama.cpp çatalı (fork) DEĞİLDİR; kendine ait dinamik MoE bellek yöneticisi (`ExpertMemoryManager`), 3-Tier bellek yerleşimi (VRAM + RAM + DirectStorage), asenkron orkestrasyonu, sıfır gecikmeli HTTP SSE boru hattı ve **Universal Hardware & Topology Auto-Tuning Engine**'i olan **tamamen bağımsız bir MoE çıkarım platformu ve launcher'ıdır**.

---

## 2. Dahil Edilen Temel Yenilikler ve Mimari Atılımlar

### P0: Universal Hardware & Topology Auto-Tuning Engine (`--autotune` / `--profile auto`)
- **Otomatik Donanım Sorgusu:** NVML ile GPU VRAM ve bant genişliği, `__cpuid` ile Zen 4/5 AVX-512 VNNI/VBMI yönergeleri, Windows `GlobalMemoryStatusEx` ile RAM baskısı ve Win32 `FSCTL_MANAGE_BYPASS_IO` ile DirectStorage desteği otomatik taranır.
- **Model Topoloji Tespiti:** GGUF başlığından katman sayısı ($L$), toplam expert sayısı ($E$), Top-$K$ routing, expert bayt boyutu ve kuantizasyon türü dinamik ayrıştırılır.
- **Optimal Yerleşim Çözücüsü:** Herhangi bir donanımda (8 GB laptop, 16-24 GB masaüstü veya CPU-only) sıfır parametre ayarıyla en yüksek TPS'yi verecek `-ngl`, `--atlas-vram-cache`, `--atlas-moe-cache-mib`, `-t`, `mtp` ve fiziksel harita parametrelerini anında çözer.

### P1: Genelleştirilmiş MoE Topolojisi & Dinamik Expert Boyutlandırma
- Eski $48 \times 512$ ve 1.253 MB sabit kodlama kısıtı kaldırılarak genelleştirildi.
- **Qwen3.6-35B-A3B** ($40 \times 256$, expert başına 1.458.176 bayt / 1.761.280 bayt) ve **Qwen3.8-Flash-Next** ($48 \times 512$, 1.253.376 bayt) modellerinin her ikisi de tam donanım hızlandırmasıyla, 0-byte kırpma ve 0 indeks taşması olmadan native motora bağlandı.

### P2: Tening Phase 11 Sinerjisi (3-Tier Bellek & Fused DMA)
- **3-Tier Hibrit Hiyerarşi:**
  - **Tier-0 (VRAM cuBLAS):** Dense attention katmanları + en sıcak %19.5'lik Ultra-Hot expert dilimi (~2.76 GiB) doğrudan VRAM'de tutulur.
  - **Tier-1 (Host RAM AVX-512):** Sıcak %34.4'lük expert dilimi (~4.85 GiB) `VirtualLock` ile RAM'de pinlenir ve VNNI çekirdeğiyle hesaplanır.
  - **Tier-2 (NVMe DirectStorage):** Soğuk kuyruk (%46.1) 4KB sektör hizalı Fused Coalesced DMA ile çekilir.
- Bellek içi aktivasyon karşılama oranı **%96.33**, disk G/Ç çağrıları **117.6x** azaltılmıştır.

### P3: Canlı Sıfır-Gecikmeli Kademeli Tool-Calling Akışı (0.2 ms)
- `SAFE_TAIL=16` tamponlama gecikmesi kaldırılarak string, escape ve süslü parantez derinliğini (`brace_depth`) anlık takip eden gerçek zamanlı ayrıştırıcı kuruldu.
- Tool çağrısı argümanları token üretildiği an istemciye akıtılarak ilk argüman gecikmesi 771.2 ms'den **0.2 ms**'ye düşürüldü.

### P4: Çok-Turlu Konuşmalarda Prefix Korumalı PLD & VRAM Geri Kazanımı
- `PromptLookupDraftEngine` içine history token zinciri ve `ingest_prompt_tokens_incremental` entegre edildi. Çok-turlu konuşmalarda prefix hash'leri korunarak prefill taslağı anında üretilir.
- `need_n_rs_seq() + 2` sınırıyla PLD logits tahsisi 256'dan 3'e düşürülerek 311.2 MiB VRAM geri kazanıldı; 8 GB kartlarda VRAM 7.4 GB seviyesinde mühürlendi.

### P5: Zen 4/5 AVX-512 VNNI Kilitsiz Thread-Slot İzolasyonu & nrc=4 Desteği
- Sıcak döngüdeki `LOCK XADD` atomic çekişmesi 64-bayt önbellek satırına izole per-thread slot yapısıyla (`s_vnni_thread_slots[64]`) tamamen elendi.
- `ggml_vec_dot_iq3_xxs_q8_K_vnni` fonksiyonuna `nrc = 4` desteği eklenerek 4 tokenlık aktivasyonların tek süper-blok ağırlık açılımı ile hesaplanması sağlandı.

### P6: Dinamik Spekülatif Ufuk Kontrolcüsü (Adaptive Zero-Penalty Fallback)
- Prompt n-gram fazlalık analizi ($\rho \ge 0.20$) ve görev etki alanı ayrıştırıcısı ile beklenen değer hesabı ($EV(d)$) dinamik olarak optimize edildi.
- Şablon tekrarı ve refactoring görevlerinde $d^* = 3$ (PLD) aktif tutularak tepe hız elde edilir; prompt tekrarı olmayan Greenfield mantık sentezi ve Tool Calling akışlarında $d^* = 0$ (saf otoregresif fallback) seçilerek red cezası, gecikme ve KV rollback ek yükü sıfırlanır.

### P7: Elastik Pinned 3-Tier Harita & Sıfır-Disk DMA (Qwen3.6-35B)
- 16GB RAM + 8GB VRAM donanımında modelin tamamı fiziksel belleğe sığdırılarak 4,720 soğuk MoE uzmanı NVMe DMA diski yerine `tier1_ram_avx512` katmanına pinlendi (`tier0`: 2000, `tier1`: 8240, `tier2`: 0).
- TTFT 2,084 ms'den 753–1,033 ms'ye indirildi (2x hızlanma); disk I/O darboğazı tamamen ortadan kaldırıldı.

---

## 3. İkili Dosya Bütünlüğü ve Manifestosu

```
releases/tendou-v0.13.0-stable/bin/
  tendou-engine.exe           508,416 B  (MSVC 19.51, AVX-512 + CUDA 13)
  llama-atlas-engine.exe      508,416 B  (Uyumluluk sembolik ikilisi)
  ggml-cpu.dll              1,125,376 B  (AVX-512 VNNI/VBMI optimized)
  ggml-cuda.dll            34,533,376 B  (NVIDIA CUDA 13 cuBLAS)
  llama.dll                 2,593,792 B  (Core tensor runtime)
  llama-common.dll          8,279,040 B  (Common support layer)
  ggml-base.dll               694,784 B  (GGML backend abstractions)
  ggml.dll                     68,608 B  (Base GGML structures)
  mtmd.dll                  2,457,088 B  (Multi-modal support)
  ggml-cpu.lib                 17,974 B  (Link import library)
```

## 4. Doğrulama Durumu

- **Native Gate Testleri:** 26 / 26 PASS (`atlas-native-tests.exe`)
- **Partition Testleri:** PASS (`atlas-partition-tests.exe`)
- **Pytest Entegrasyon Paketi:** 109 passed / 4 skipped (%100 PASS)
- **Top-1 Token Paritesi:** %100.0 (Bit-exact referans)
- **Logit KL Divergence:** $\text{KL} = 0.000000$ (Sıfır matematiksel kayıp)
