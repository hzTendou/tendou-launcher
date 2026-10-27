# Tendou Launcher — Kapsamlı Kullanıcı ve Entegrasyon Kılavuzu

**Sürüm:** v0.9.0-stable  
**Hedef Platform:** Windows 11 (x86-64 / NVIDIA CUDA)  
**Doküman Kodu:** `DOC-USER-GUIDE-09`

---

## 1. Tendou Launcher Nedir?

**Tendou Launcher**, tüketici sınıfı bilgisayarlarda (özellikle standart 16 GB RAM ve 8 GB VRAM'e sahip gaming laptoplarda) 35B–48B parametreli devasa Mixture-of-Experts (MoE) dil modellerini yüksek hızda ve sıfır matematiksel kayıpla çalıştırmak için tasarlanmış **bağımsız bir MoE çıkarım platformu ve yerel API sunucusudur**.

Kendisine ait bellek yöneticisi (`PhysicalExpert`, `CrossLayerGPUPinning`), sıfır-kopya hibrit önbelleği (`AtlasHybridStateCache`), gelişmiş SIMD çekirdekleri ve yerel OpenAI uyumlu REST API arayüzü sayesinde OpenCode, Continue, Roo Code ve Cursor gibi kodlama istemcilerine kesintisiz yerel zeka sağlar.

---

## 2. Donanım ve Yazılım Gereksinimleri

| Bileşen | Minimum Gereksinim | Önerilen / Test Edilen |
|---|---|---|
| **İşletim Sistemi** | Windows 10/11 64-bit | Windows 11 (Build 22631+) |
| **İşlemci (CPU)** | 6 Çekirdek x86-64 (AVX2 destekli) | AMD Ryzen 7 260 / 7840HS / 8845HS (8C/16T Zen 4) |
| **Sistem Belleği (RAM)**| 16 GB DDR5 | 16 GB veya 32 GB DDR5-5600 |
| **Ekran Kartı (GPU)** | 6 GB NVIDIA RTX Serisi (CUDA) | NVIDIA RTX 5060 Laptop (8 GB GDDR6) veya üstü |
| **Depolama** | 100 GB Boş Alan (PCIe NVMe SSD) | PCIe Gen4 NVMe SSD |
| **Python** | Python 3.10 – 3.12 | Python 3.12 64-bit |

---

## 3. Kurulum ve Ortam Hazırlığı

### Adım 1: Depoyu Hazırlama ve Sanal Ortam Kurulumu
PowerShell konsolunu açın:

```powershell
cd C:\Users\Ali\WebProjects\tendou-launcher

# Python sanal ortamını oluşturun
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Bağımlılıkları yükleyin
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Adım 2: Kararlı Motor İkililerinin Doğrulanması
Tendou Launcher, hazır ve mühürlenmiş kararlı sürümlerle birlikte gelir:
- En güncel kararlı sürüm: [`releases/tendou-v0.9.0-stable/bin/tendou-engine.exe`](file:///c:/Users/Ali/WebProjects/tendou-launcher/releases/tendou-v0.9.0-stable/bin/tendou-engine.exe)
- Motor arama sırası otomatik olarak `tendou-engine.exe`'yi algılar; manuel bir derleme yapmanıza gerek yoktur.

---

## 4. Model Ağırlıklarının Temini

Tendou Launcher, GGUF formatındaki optimize edilmiş MoE modellerini destekler:

1. **Önerilen 1. Model (Qwen3.8 Flash Next IQ3_XXS):**
   - 48 Katman, 512 Uzman.
   - Boyut: ~70 GB (2 Shard).
   - Dizin: `models/qwen3.8-gsq-rco-iq3_xxs/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf`
2. **Önerilen 2. Model (Qwen3.6-35B-A3B-UD-Q8_K_XL):**
   - 40 Katman, 256 Uzman, Q8 hassasiyeti.
   - Boyut: ~38.5 GB tek parça.
   - Doğrudan HuggingFace önbelleğinden veya yerel diskten beslenebilir.

Ortam değişkeni ile model yolunu belirtebilirsiniz:
```powershell
$env:TENDOU_MODEL_PATH = "C:\Modeller\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
```

---

## 5. Sunucuyu Başlatma

### Yöntem A: Hazır Başlatıcı Dosyası ile (Önerilen)
Kök dizindeki `start_server.cmd` dosyasını çift tıklayarak veya komut satırından çalıştırabilirsiniz:
```cmd
start_server.cmd
```

### Yöntem B: CLI Üzerinden Detaylı Parametrelerle Başlatma
```powershell
.venv-audit\Scripts\python.exe atlas_server.py `
  --model-path "C:\Modeller\modeliniz.gguf" `
  --port 8000 `
  --threads 8 `
  --gpu-layers 2 `
  --prompt-cache-mb 256 `
  --cache-type-k q8_0 `
  --cache-type-v q4_0 `
  --boost
```

### Önemli CLI Bayrakları ve Anlamları

| Bayrak | Varsayılan | Açıklama |
|---|---|---|
| `--port` | `8000` | API sunucusunun dinleyeceği yerel port. |
| `--threads` | `8` | CPU MoE hesaplaması için iş parçacığı sayısı (SMT çekişmesini önlemek için fiziksel çekirdek sayısı önerilir). |
| `--gpu-layers` | `2` | GPU VRAM üzerine kilitlenecek MoE katmanı sayısı (VRAM taşmasını önlemek için 2 önerilir). |
| `--prompt-cache-mb` | `256` | L1 RAM durum önbelleği boyutu (MB). |
| `--cache-type-k` | `q8_0` | K önbelleği quantization formatı. |
| `--cache-type-v` | `q4_0` | V önbelleği quantization formatı (Asimetrik KV, düşük VRAM ayak izi). |
| `--boost` | Aktif | Katman uyarlamalı dinamik K budamasını devreye sokarak TPS'yi maksimize eder. |
| `--mock` | `False` | Model ağırlığı olmadan sadece API arayüzünü test etmek için sahte motor modu. |

---

## 6. API Kullanımı ve İstemci Entegrasyonu

Tendou Launcher, **OpenAI REST API v1** standardıyla %100 uyumludur.

### 6.1. Sağlık Kontrolü ve Model Listesi
```powershell
# Sağlık Durumu
Invoke-RestMethod http://127.0.0.1:8000/health

# Aktif Model Bilgisi
Invoke-RestMethod http://127.0.0.1:8000/v1/models
```

### 6.2. Python `openai` Kütüphanesi ile Kullanım
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="tendou-local"  # Yerel sunucu için rastgele bir anahtar yeterlidir
)

response = client.chat.completions.create(
    model="qwen3.8-flash-next",
    messages=[
        {"role": "system", "content": "Sen yardımcı ve uzman bir yazılım mühendisisin."},
        {"role": "user", "content": "Python ile binary search algoritması yaz."}
    ],
    temperature=0.2,
    stream=True
)

for chunk in response:
    content = chunk.choices[0].delta.content or ""
    print(content, end="", flush=True)
```

---

## 7. OpenCode Entegrasyonu

Tendou Launcher, OpenCode TUI kodlama aracına yerel olarak tam entegre çalışır.

Proje kök dizinindeki `opencode.json` konfigürasyonu hazırdır:
```json
{
  "provider": "openai",
  "baseURL": "http://127.0.0.1:8000/v1",
  "apiKey": "tendou-local",
  "models": {
    "default": "qwen3.8-flash-next"
  }
}
```

Doğrudan çalıştırmak için:
```cmd
opencode.cmd
```
OpenCode arayüzünde tool-call desteği, dosya okuma/yazma ve otomatik keepalive pingleri ile kesintisiz kodlama yapabilirsiniz.

---

## 8. Sık Karşılaşılan Sorunlar ve Çözümleri (Troubleshooting)

### Sorun 1: "Address already in use / Port 8000 meşgul"
- **Çözüm:** Arka planda kalmış eski bir sunucu işlemini sonlandırın:
  ```powershell
  taskkill /F /IM python.exe /T
  taskkill /F /IM tendou-engine.exe /T
  ```

### Sorun 2: "CUDA out of memory / VRAM yetersiz"
- **Neden:** `--gpu-layers` değeri çok yüksek seçilmiş olabilir veya arkada açık başka bir oyun/GPU uygulaması VRAM tüketiyordur.
- **Çözüm:** `--gpu-layers 2` veya `1` kullanın. Tendou Launcher'ın 48-katman çapraz pinleme mekanizması katman sayısından bağımsız olarak sıcak uzmanları VRAM'e yerleştirecektir.

### Sorun 3: "Yanıt çok yavaş başladı (Uzun TTFT)"
- **Neden:** İlk istekte model mmap sayfalarını NVMe diskten RAM'e çekiyordur (soğuk başlangıç).
- **Çözüm:** Tendou Launcher'ın `AtlasHybridStateCache` sistemi devreye girdikten sonra 2. turdan itibaren sıcak TTFT 200–350 ms aralığına inecektir.
