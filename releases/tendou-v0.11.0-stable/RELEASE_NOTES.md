# Tendou Launcher v0.11.0-stable (LATEST STABLE RELEASE)

**Yayın Tarihi:** 2026-10-20  
**Durum:** En Güncel Kararlı Sürüm (Mühürlendi, Doğrulandı, Gate 1–26 PASS)

---

## Proje Kimliği ve Zihniyeti
Tendou Launcher, bir `llama.cpp` sürümü, modu veya çatalı (fork) **DEĞİLDİR**. Kendine ait bellek yöneticisi (`PhysicalExpert`, `CrossLayerGPUPinning`), asenkron orkestrasyonu, sıfır-kopya hibrit durum önbelleği (`AtlasHybridStateCache`), DirectStorage / Win32 BypassIO G/Ç motoru, donanıma özel AVX-512 VNNI/VBMI tensör çekirdekleri, derin telemetri izlemesi ve yerel OpenAI uyumlu sunucusu olan **tamamen bağımsız bir MoE çıkarım platformu ve launcher'ıdır**. `llama.cpp` ve `ggml`, yalnızca alt seviye tensör ve hesaplama arka ucu (backend engine dependency) olarak kullanılır.

---

## Sürüm Özellikleri (Phase 10 & Phase 11 Milestones)

### 1. 64k / 128k / 256k Dinamik Bağlam Ölçekleme (Phase 10 — Paket P0)
- **Kademeli Asimetrik KV Durum Makinesi:**
  - $N \le 8,192$: $K = \text{q8\_0}, V = \text{q8\_0}$ (Maksimum sayısal hassasiyet, 21.25 KB/tok)
  - $8,192 < N \le 32,768$: $K = \text{q8\_0}, V = \text{q4\_0}$ (Phase 9 tabanı, 16.25 KB/tok)
  - $32,768 < N \le 131,072$: $K = \text{q4\_0}, V = \text{q4\_0}$ (**11.25 KB/tok**, 16 GB RAM sınırında sıfır OOM garantisi)
  - $N > 131,072$: Attention-Sink korunumu (ilk 64 token) + 8k yerel halka tamponu + `iq2_xxs` hibrit sıkıştırma ($< 1.95\text{ GiB}$ toplam 256k KV ayak izi).
- **Amortize Edilmiş Çift İmleçli (Dual-Cursor) Arka Plan Yeniden Nicemleme:**
  - Eşik aşımlarında tek adımda kilitlenme yerine adım başına 128 tokenlık ($B=4$ blok) yumuşak dönüşümle gecikme sıçraması $< 0.12\text{ ms}$ seviyesine indirildi.
  - Geri sarma koruması (`rollback_context`) ile işaretsiz tamsayı taşması (unsigned underflow) tamamen engellendi.
- **Blok-Seyrek (Block-Sparse) Dikkat:**
  - FlashAttention uyumlu 64x64 karo maskelemesi ile nedensel gelecek karo sızıntıları giderildi.

### 2. Zen 4 / Zen 5 AVX-512 VNNI + VBMI Tensör Çekirdekleri (Phase 10 & 11 — Paket P1)
- **Doğrudan `__mmask64` VNNI İşaret Maskesi:**
  - Vektörel şerit karıştırma ve `vcmpltb` yükü kaldırılarak `k_even_sign_mask` tablosu üzerinden doğrudan 64-bit tamsayı operasyonlarıyla `__mmask64` üretimi sağlandı.
- **Maskeli Broadcast (`_mm512_mask_set1_epi32`):**
  - 16 argümanlı yavaş `_mm512_set_epi32` kaldırıldı; `0xFF00` maskeli `_mm512_mask_set1_epi32` ile döngü içi yığın taşması (stack spill) tamamen elendi.
  - Tekil decode (`max_t == 1`) ve artık slot satırları da 100% AVX-512 VNNI yoluna bağlandı.
- **Çoklu-Token (`nrc = 2`) Ağırlık Paylaşımı:**
  - $B \ge 2$ spekülatif doğrulama adımlarında ağırlık ve kod kitabı ızgara açılımları tekil okunup 2 ayrı token aktivasyonuyla paralel çarpıldı.
  - DDR5 ağırlık bant genişliği tüketimi %50 düşürüldü; `_InterlockedIncrement64` / `__atomic_fetch_add` ile multithreaded OpenMP ortamında yarış durumları engellendi.
  - Tekil `nrc = 1` çağrılarıyla **0-ULP tam bit-exact parite** korundu.

### 3. Track C — MTP Spekülatif Çıkarım Amplifikasyonu (Phase 10 & 11 — Paket P2)
- **MoE Duyarlı Dinamik Beklenen Değer ($EV$) Kalibrasyonu:**
  - Doğrulanan $d$ taslak tokenın CPU MoE uzman uyanış çarpanını ($\alpha(d) \ge 1$) hesaba katan genişletilmiş Beklenen Değer formülasyonu:
    $$EV(d) = \sum_{i=1}^{d} \left( \prod_{j=1}^{i} P(\text{accept}_j) \right) \cdot T_{\text{target}}(1) - \left[ d \cdot T_{\text{draft\_step}} + (\alpha(d) - 1) \cdot T_{\text{CPU\_MoE}} \right]$$
  - EMA gecikme telemetrisi ile çalışma anında $EV(d) \le 0$ olduğunda sıfır gecikmeyle $d = 0$ (standart greedy decode) fallback moduna geçilerek hız kaybı önlendi.
- **Sıfır-Kopya KV Önbellek Geri Sarma (Zero-Cost Rollback):**
  - Reddedilen taslaklar bellek kopyalaması yapılmadan yalnızca atomik imleç geri çekme ile $O(1)$ sürede temizlenir.
- **Sıfır-VRAM N-Gram Prompt Lookup (PLD) Taslak Motoru:**
  - GPU VRAM kotasını tüketmeyen (0 MB VRAM), CPU/RAM üzerinde 4096-token ring buffer ve 64-bit FNV-1a hash indeksli ultra hafif taslak motoru entegre edildi.
  - Taslak üretim gecikmesi $T_{\text{draft\_step}} \approx \mathbf{0.002\text{--}0.008\text{ ms}}$ seviyesine indirildi (sinirsel modele kıyasla >700x hızlanma).
  - Yaratıcı sohbet iş yükünde efektif decode hızı **12.21 TPS'den 20.53 TPS'e (+%68)** yükseldi ve 5/5 deterministik parite korundu.

### 4. Windows DirectStorage / BypassIO Sıfır-Kopya SSD Akışı (Phase 10 — Paket P3)
- Windows 11 NVMe yığınında `FSCTL_MANAGE_BYPASS_IO` (`0x000903E8`) ile dosya sistemi filtre sürücüleri baypas edildi.
- 4096-byte sektör hizalı `VirtualAlloc` + `cudaHostRegister` pinned staging tamponları kuruldu.
- $O(1)$ çift tamponlu (double-buffered) öngörülü MoE uzman getirme motoru ile sayfa hatası (page fault) kaynaklı mikro-duraksamalar elendi.

---

## Doğrulama ve Test Kapıları Sonuçları
- **Yerel Test Kapıları (`atlas-native-tests.exe`):** 26/26 Gate **%100 PASS**.
- **Matris Bölümleme Doğrulaması (`atlas-partition-tests.exe`):** **PASS** (tekil iş parçacığı referansıyla tam eşleşme).
- **Pytest Birim ve Aşama Testleri:** 186 passed, 4 skipped (**%100 PASS**).
- **Deterministik Çıktı Eşitliği:** Greedy decode ve MTP açık modda 5/5 tekrarda token-token tam özdeşlik.

---

## Dosyalar
- `bin/tendou-engine.exe`: Tendou Launcher ana native çıkarım motoru binary'si (Phase 11 VNNI + PLD entegre).
- `bin/llama-atlas-engine.exe`: Geriye dönük uyumluluk ve eski scriptler için ikincil isim eşlemesi.
- `bin/ggml-cpu.dll`: Tendou hızlandırılmış CPU tensör çekirdeği kütüphanesi (`__mmask64` + masked broadcast + `nrc=2`).
- `bin/ggml-cpu.lib`: GGML CPU export import kütüphanesi.
- `source/`: Bu sürüme ait dondurulmuş motor ve backend kaynak kodları (`src/atlas`, `llama.cpp`).
