# Tendou Launcher — Yol Haritası ve Açık Araştırma Konuları (Roadmap & Open Problems)

**Yazarlar:** Tendou Launcher Core Systems Team  
**Tarih:** 2026-10-19  
**Kapsam:** Phase 9 Sonrası Gelecek Vizyonu, Phase 10 Hedefleri ve Açık Teknik Problemler

---

## 1. Giriş

Tendou Launcher, Phase 9 (`v0.9.0-stable`) sürümü ile 16 GB RAM ve 8 GB VRAM donanımlı gaming laptoplarda 21.58+ decode TPS, 353 ms soğuk TTFT ve 5/5 tam deterministik parite hedeflerini mühürlemiştir. Ancak büyük dil modellerinin bağlam gereksinimleri (context window) ve spekülatif çıkarım sınırları sürekli genişlemektedir.

Bu doküman; projenin ucu açık teknik problemlerini, araştırma aşamasındaki çalışma paketlerini ve Phase 10 yol haritasını ortaya koyar.

---

## 2. Açık Araştırma Konuları ve Çalışma Paketleri

### 2.1. Phase 10: 64k / 128k / 256k Dinamik Bağlam Ölçekleme (Context Scaling)
- **Mevcut Durum:** 32K bağlamda `cache_type_k = q8_0` ve `cache_type_v = q4_0` asimetrik KV mimarisi ve sıfır-kopya `AtlasHybridStateCache` sorunsuz çalışmaktadır.
- **Açık Problem:** Bağlam uzunluğu 64k, 128k veya 256k seviyesine çıktığında, KV önbelleğinin kapladığı bellek VRAM ve RAM bütçesini zorlamaya başlamakta; prefill süresi token sayısı arttıkça karesel (quadratic) büyüme eğilimi göstermektedir.
- **Araştırılan Çözüm Yolu:**
  1. **Kademeli Asimetrik Quantization:** Bağlam uzunluğuna göre dinamik KV sıkıştırması:
     - $N \le 8,192$: `q8_0 / q8_0` (Maksimum sayısal hassasiyet).
     - $8,192 < N \le 32,768$: `q8_0 / q4_0` (Mevcut kanıtlanmış Phase 9 tabanı).
     - $32,768 < N \le 131,072$: `q4_0 / q4_0` (11.25 KB/tok ayak izi).
     - $N > 131,072$: `q4_0 / iq2_xxs` veya katman bazlı sliding window KV eviction.
  2. **Sliding Window Chunking & Sparse Attention:** Son 8k tokenı tam hassasiyette tutup eski katmanları sıkıştırılmış duruma geçiren dinamik dikkat halkası.

---

### 2.2. Track C: VRAM-Resident MTP Spekülatif Çıkarım Amplifikasyonu
- **Mevcut Durum:** 8 GB VRAM sınırında hedef model ve draft model aynı anda VRAM'e sığmadığı için MTP varsayılan olarak `off` (kapalı) tutulmaktadır.
- **Açık Problem:** 12 GB veya 16 GB VRAM'e sahip sistemlerde (örn. RTX 4080/4090 veya RTX 5070 Ti+ Laptoplar) ek 2–3 GB boş VRAM kapasitesi mevcuttur. Bu sistemlerde MTP'nin %100 VRAM-resident olarak devreye sokulması potansiyel olarak **35–45 TPS** decode hızını mümkün kılabilir.
- **Araştırılan Çözüm Yolu:**
  1. `models/mtp/Qwen3.8-Flash-Next-MTP-Q4_0.gguf` modelinin tüm MoE katmanlarının CUDA VRAM üzerine yerleştirilmesi.
  2. Dinamik kabul ufku ($d$) için Beklenen Değer ($EV$) formülünün kalibrasyonu:
     $$EV = \sum_{i=1}^{d} P(\text{accept}_i) \cdot T_{\text{target}} - T_{\text{draft\_step}} \cdot d$$
     $EV \le 0$ olduğunda otomatik olarak $d=0$ (fallback) yapılarak kritik yol regresyonu engellenmelidir.

---

### 2.3. Zen 4 / Zen 5 AVX-512 VNNI Özelleşmiş Tensör Çekirdeği
- **Mevcut Durum:** Güncel derleme 256-bit AVX2 ve FMA komut setlerini kullanmaktadır.
- **Açık Problem:** AMD Ryzen 7 260 ve Ryzen 9 işlemcileri donanımsal olarak **AVX-512 VNNI** (Vector Neural Network Instructions) desteğine sahiptir. Bu komutlar, INT8 ve INT4 dot product işlemlerini tek döngüde çarpıp toplayabilmektedir.
- **Araştırılan Çözüm Yolu:**
  - `ggml-cpu/arch/x86/quants.c` içine `_mm512_dpbusd_epi32` veya 256-bit VNNI intrinsic'leri ile optimize edilmiş yeni bir `ggml_vec_dot_iq3_xxs_vnni` çekirdeği yazılması.
  - Hedeflenen Kazanç: CPU MoE saf hesaplama süresini %20–%30 oranında daha aşağı çekmek.

---

### 2.4. Windows DirectStorage / Bypass-IO ile Sıfır-Kopya SSD Akışı
- **Mevcut Durum:** Model ağırlıkları Windows sanal bellek yöneticisi (`mmap`) üzerinden okunmaktadır.
- **Açık Problem:** İşletim sistemi mmap sayfası yüklerken sayfa hatası (page fault) kesintisi üretir ve CPU çekirdek moduna girer.
- **Araştırılan Çözüm Yolu:**
  - PCIe Gen4/Gen5 NVMe SSD'den doğrudan GPU VRAM'ine veya pinned RAM tamponlarına aktarım sağlayan Win32 DirectStorage API'sinin değerlendirilmesi.

---

## 3. Öncelik Sıralaması

| Öncelik | Çalışma Paketi | Hedef Donanım | Beklenen Etki |
|---|---|---|---|
| **P0** | Phase 10: 64k/128k Asimetrik KV Dinamik Ölçekleme | 16 GB RAM / 8 GB VRAM | Sıfır OOM ile 128k bağlamda kesintisiz kodlama |
| **P1** | Zen 4 AVX-512 VNNI Kernel Derlemesi | Ryzen 7 / 9 CPU | CPU MoE süresinde %25 ek tasarruf |
| **P2** | Track C VRAM-Resident MTP Profilleme | 12 GB+ VRAM GPU | Yüksek kabul oranıyla 30+ Decode TPS |
| **P3** | DirectStorage NVMe DMA Araştırması | PCIe 4.0 SSD | İlk model yükleme ve sayfa getirme süresini kısaltma |
