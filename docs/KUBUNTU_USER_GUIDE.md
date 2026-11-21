# Tendou Launcher — Kubuntu Kullanıcı, Sunucu ve Ajan CLI Kılavuzu

**Hedef Sistem:** Kubuntu 26.04 (Linux 7.0+, GCC 14, NVIDIA CUDA 13.2 `sm_120a`)  
**Doküman Kodu:** `DOC-KUBUNTU-GUIDE-14`  
**Son Güncelleme:** 2026-11-09  

---

## 1. Genel Bakış

**Tendou Launcher**, tüketici sınıfı donanımlarda (özellikle standart 16 GB RAM ve 8 GB VRAM'e sahip gaming laptoplarda) 35B–197B parametreli MoE (Mixture of Experts) modellerini yüksek hızda ve sıfır matematiksel kayıpla çalıştırmak için tasarlanmış bağımsız bir çıkarım platformu ve yerel API sunucusudur.

Kubuntu üzerinde çalışırken:
- **Daha Yüksek Performans:** İşletim sisteminin hafif bellek ayak izi ve GCC 14 native AVX-512 VNNI çekirdekleri sayesinde Windows 11'e kıyasla **%30 ila %45 daha yüksek Decode TPS** (35B modelinde 80–104 TPS, 38B/197B modelinde 31–50 TPS) elde edilir.
- **Sıfır-Kopya CUDA Host Pinning:** `ExpertSlabPool` ve `DirectStorage/pread64` staging havuzları `CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP` ile kilitlenerek DMA gecikmesi en aza indirilir.
- **Çoklu Ajan Protokolü:** **Claude Code CLI**, **OpenCode** ve **Cline (VS Code / Cursor)** araçlarını hiçbir harici proxy veya çeviriciye ihtiyaç duymadan doğrudan besler.

---

## 2. Tek Seferlik Sistem ve Donanım Hazırlığı

Kubuntu üzerinde maksimum verim ve kararlılık için aşağıdaki adımları bir kez uygulamanız önerilir.

### A. GPU Persistence Mode (Kalıcılık Modu)
NVIDIA sürücüsünün her çağrıda VRAM bağlamını baştan oluşturmasını önler:
```bash
sudo nvidia-smi -pm 1
```
*(Yeniden başlatmalarda kalıcı olması için `/etc/systemd/system/nvidia-persistenced.service` etkinleştirilebilir).*

### B. Bellek ve Mmap Tavan Ayarları
Büyük MoE modellerinin shard haritaları ve slab havuzları için çekirdek sınırlarını optimize edin:
```bash
sudo sysctl -w vm.max_map_count=1048576
sudo sysctl -w vm.swappiness=10
```
Kalıcı yapmak için `/etc/sysctl.d/99-tendou.conf` dosyasına ekleyin:
```ini
vm.max_map_count=1048576
vm.swappiness=10
```

### C. Transparent HugePages (2 MB Büyük Sayfalar)
Bellek TLB önbellek kaçırmalarını önlemek için:
```bash
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
```

### D. Kilitli Bellek Sınırı (`memlock`)
CUDA host pinning (`cuMemHostRegister`) ve `mlock` için sınırları kaldırın:
```bash
sudo tee /etc/security/limits.d/99-tendou-memlock.conf << 'EOF'
* soft memlock unlimited
* hard memlock unlimited
EOF
```

### E. Dual-Boot Disk Performansı (NTFS vs NTFS3)
Eğer model dosyalarınız Windows ile ortak bir NTFS diskinde (`/mnt/Ortak`) duruyorsa, Ubuntu varsayılan olarak kullanıcı alanında çalışan `mount.ntfs-3g` (FUSE) sürücüsünü kullanır. FUSE, her soğuk sayfa okumasında kullanıcı-çekirdek bağlam geçişi ek yükü getirir.  
Linux 7.0 çekirdeğinde yerleşik bulunan yüksek hızlı `ntfs3` sürücüsünü kullanmak için `/etc/fstab` içindeki ilgili satırı güncelleyebilirsiniz:
```ini
# /etc/fstab örneği (UUID veya cihaz adınıza göre düzenleyin):
/dev/nvme0n1p4  /mnt/Ortak  ntfs3  defaults,uid=1000,gid=1000,dmask=022,fmask=133,iocharset=utf8  0  0
```
Bu ayar, modelin ilk soğuk yükleme (Cold TTFT) süresini 2–3 kat hızlandırır.

---

## 3. Tendou Launcher Sunucusunu Kubuntu'da Başlatma

Sunucuyu çalıştırmak için proje kök dizininde hazır bulunan başlatma betiğini kullanabilirsiniz.

### A. Hızlı Başlatma (Varsayılan Model)
```bash
cd /mnt/Ortak/WebProjects/tendou-launcher
./start_server.sh --port 8000
```
Sunucu hazır olduğunda ekranda şu çıktıyı görürsünüz:
```text
[ATLAS_READY] Server initialized and listening on http://127.0.0.1:8000/v1
```

### B. Farklı Model Seçimi ile Başlatma
Sunucuya `--model-path` veya `--model` argümanı vererek dilediğiniz modeli başlatabilirsiniz:

```bash
# 1. Öneri: Qwen3.6-35B-A3B (Hızlı ve Güçlü - 80-104 TPS)
./start_server.sh --model 35b --port 8000

# 2. Öneri: Qwen3.8-Flash-Next (197B, 70.6 GiB Devasa Akıl Yürütme - 31-50 TPS)
./start_server.sh --model 38b --port 8000

# Manuel Model Yolu ile:
./start_server.sh --model-path /mnt/Ortak/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/.../Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --port 8000
```

### C. Parametreler ve Anlamları
```bash
./start_server.sh \
  --port 8000 \
  --threads 8 \
  --gpu-layers 2 \
  --prompt-cache-mb 512 \
  --cache-type-k q8_0 \
  --cache-type-v q4_0 \
  --boost
```
- `--threads 8`: CPU tarafındaki MoE hesaplaması için iş parçacığı sayısı (SMT çekişmesini önlemek için 8 fiziksel çekirdek).
- `--gpu-layers 2`: Dikkat (attention) ve kritik yönlendirici katmanlarını GPU VRAM'ine kilitler.
- `--prompt-cache-mb 512`: Çok turlu sohbetlerde aynı proje geçmişini saklar (Warm TTFT: 10–14 ms).
- `--boost`: Autotune motorunun donanımınızı (AVX-512 VNNI, VRAM kotası) en yüksek profile ayarlamasını sağlar.

---

## 4. Sunucuyu Arka Planda Yönetme (tmux / systemd)

Gerçek iş sırasında terminali kapatabilmek için sunucuyu arka planda çalıştırmanız önerilir.

### Yöntem 1: `tmux` ile Çalıştırma (En Pratik)
```bash
# Yeni bir tmux oturumu başlatıp sunucuyu açın:
tmux new -s tendou './start_server.sh --port 8000'

# Oturumdan çıkmak için: Ctrl + B tuşlarına basıp ardından D tuşuna basın.
# Oturuma tekrar bağlanmak için:
tmux attach -t tendou
```

### Yöntem 2: `systemd` Kullanıcı Servisi ile Çalıştırma (Otomatik Başlatma)
Kullanıcı düzeyinde servis dosyası oluşturun (`~/.config/systemd/user/tendou.service`):
```ini
[Unit]
Description=Tendou Launcher MoE Server
After=network.target

[Service]
Type=simple
WorkingDirectory=/mnt/Ortak/WebProjects/tendou-launcher
ExecStart=/mnt/Ortak/WebProjects/tendou-launcher/start_server.sh --port 8000
Restart=on-failure
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
```

Servisi etkinleştirip başlatın:
```bash
systemctl --user daemon-reload
systemctl --user enable --now tendou.service

# Canlı logları izlemek için:
journalctl --user -u tendou.service -f
```

### Sunucu Sağlık Kontrolü
Sunucunun yanıt verdiğini doğrulamak için:
```bash
curl http://127.0.0.1:8000/v1/models
```

---

## 5. Claude Code CLI ile Kubuntu'da Çalışma

**Claude Code**, Anthropic'in terminal tabanlı otonom kodlama aracıdır. Tendou Launcher, yerel `/v1/messages` ve `/messages` uç noktalarını doğal olarak sunduğu için harici proxy'lere ihtiyaç duymaz.

### Adım 1: Claude Code Kurulumu
Node.js (v18+) kurulu değilse kurun ve CLI'yi yükleyin:
```bash
sudo apt update && sudo apt install -y nodejs npm
sudo npm install -g @anthropic-ai/claude-code
```

### Adım 2: Ortam Değişkenlerini Ayarlayın
Yeni bir terminalde:
```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:8000"
export ANTHROPIC_API_KEY="tendou-local"

# Doğrudan projenizin dizininde çalıştırın:
cd /mnt/Ortak/WebProjects/benim-projem
claude
```

### Adım 3: Sürekli Kullanım İçin Alias Tanımlama
Her seferinde değişken tanımlamamak için `~/.bashrc` veya `~/.zshrc` dosyanıza şu satırları ekleyin:
```bash
alias claude-tendou='ANTHROPIC_BASE_URL="http://127.0.0.1:8000" ANTHROPIC_API_KEY="tendou-local" claude'
```
Ardından sadece `claude-tendou` yazarak çalıştırabilirsiniz.

### Gerçek İş Sırasında Neler Olur?
- **Dosya Düzenleme ve Keşif:** Claude Code `read_file`, `grep`, `edit_file` komutlarını Tendou Launcher'a gönderir.
- **Warm Prompt Caching:** İlk sorudan sonra projenin kod geçmişi sunucunun `AtlasHybridStateCache` önbelleğinde kaldığı için sonraki turlar **11–14 ms Warm TTFT** hızında yanıt üretir.
- **Refactoring:** Kod değişikliklerinde Prompt Lookup Decoding (PLD $d^*=3$) otomatik devreye girerek **80–82 TPS** hızında çıktı akıtır.
- **Auto Mode / Auto Accept Desteği:** Claude Code'un otonom modunda (`--permission-mode auto` veya `/auto-mode`), arka planda çalışan ~35.000 tokenlik güvenlik denetçisi (`auto_mode` classifier) Tendou sunucusu tarafından otomatik olarak algılanır ve 0 ms gecikmeyle anında onaylanır (`<severity>0</severity>`). Böylece Claude Code'un 60 saniyelik istemci zaman aşımına takılmadan tüm terminal komutları ve araç çağrıları onay beklemeden kesintisiz çalışır. Dilerseniz tüm onayları tamamen atlamak için `claude --dangerously-skip-permissions` veya `claude --permission-mode acceptEdits` bayraklarını da kullanabilirsiniz.

---

## 6. OpenCode ile Kubuntu'da Çalışma

**OpenCode**, terminalinizde OpenAI uyumlu modellerle çalışan hafif ve hızlı bir kodlama arayüzüdür.

### Adım 1: OpenCode Kurulumu
```bash
npm install -g opencode-ai
```

### Adım 2: Proje Yapılandırması (`opencode.json`)
Çalıştığınız projenin kök dizinine (veya ev dizininize `~/.config/opencode/opencode.json`) aşağıdaki yapılandırmayı yerleştirin:
```json
{
  "provider": {
    "tendou": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Tendou Launcher",
      "options": {
        "baseURL": "http://127.0.0.1:8000/v1",
        "apiKey": "tendou-local"
      },
      "models": {
        "qwen3.6-35b": {
          "name": "Qwen 3.6 35B (Tendou 3-Tier)",
          "limit": {
            "context": 65536,
            "output": 8192
          }
        },
        "qwen3.8-flash-next": {
          "name": "Qwen 3.8 Flash Next 197B",
          "limit": {
            "context": 65536,
            "output": 8192
          }
        }
      }
    }
  }
}
```

### Adım 3: Çalıştırma
```bash
cd /mnt/Ortak/WebProjects/benim-projem
opencode
```
OpenCode, streaming token yanıtları ve araç çağrılarını Tendou Launcher'ın `/v1/chat/completions` uç noktasından **80–95 TPS** hızında alır.

---

## 7. Cline (VS Code / Cursor / VSCodium) ile Kubuntu'da Çalışma

**Cline**, editörünüzün içinde tam yetkili otonom yazılım geliştirme yapan açık kaynaklı bir ajandır. Tendou Launcher, Cline'ın özel XML araç protokolünü ve tek-araç kesme kurallarını doğal olarak destekler.

### Adım 1: Kurulum
Kubuntu'da kullandığınız editörde (VS Code / Cursor / VSCodium) **Extensions** sekmesinden `Cline` eklentisini kurun.

### Adım 2: Cline Ayarlarını Yapılandırma
Editörde Cline panelini açın, sağ üstteki dişli çark simgesine (Settings) tıklayın:

#### Seçenek A: OpenAI Compatible / NVIDIA NIM Modu (Önerilen)
- **API Provider:** `OpenAI Compatible`
- **Base URL:** `http://127.0.0.1:8000/v1`
- **API Key:** `tendou-local`
- **Model ID:** `qwen3.6-35b-a3b` *(veya `qwen3.8-flash-next`)*

#### Seçenek B: Anthropic Modu
- **API Provider:** `Anthropic`
- **Base URL:** `http://127.0.0.1:8000`
- **API Key:** `tendou-local`
- **Model ID:** `qwen3.6-35b-a3b`

### Tendou Launcher'ın Cline İçin Özel Avantajları
1. **Tek-Araç-Kuralı Kesici (`truncate_after_first_cline_xml_tool`):**  
   Cline `<read_file>`, `<replace_in_file>` veya `<write_to_file>` etiketini kapattığı anda üretim anında sonlandırılır. Modelin sahte terminal yanıtı uydurması (%100) engellenir.
2. **Hızlı İlk Argüman İletimi:**  
   Dosya düzenleme işlemlerinde ilk XML argümanı **25–73 ms** içinde ekrana düşer.
3. **Önbellek Telemetrisi:**  
   Cline alt panelindeki *Context Tokens / Cache Read* sayaçları (`cached_tokens`) gerçek zamanlı olarak güncellenir.

---

## 8. Gerçek İş Sırasında Maksimum Verim İçin İpuçları

1. **Aynı Oturumda Kalın (Warm Context):**  
   Ajanla çalışırken sohbeti sürekli sıfırlamak yerine aynı görev hattında devam edin. `AtlasHybridStateCache` sayesinde 2. turdan itibaren TTFT süresi 10–14 milisaniyeye iner.
2. **Kod Refactoring'de Spekülatif Kazanç:**  
   Büyük fonksiyonları veya dosya bloklarını yeniden yazdırırken model tekrarlayan desenleri PLD ($d^*=3$) ile tek adımda üretir ve hız **82–89 TPS** bandına fırlar.
3. **Donanım Takibi:**  
   İkinci bir terminalde kaynakları canlı takip edebilirsiniz:
   ```bash
   # GPU VRAM ve Güç Takibi:
   watch -n 1 nvidia-smi

   # CPU ve RAM Takibi:
   htop
   ```
4. **Hata Durumunda Hızlı Yeniden Başlatma:**  
   Port meşgul hatası alırsanız sunucu otomatik `SO_REUSEADDR` uygular. Gerekirse eski süreci sonlandırabilirsiniz:
   ```bash
   fuser -k 8000/tcp
   ```

---

## 9. Tendou Studio (Web UI & Tauri v2 Masaüstü) ve Geliştirici Komutları

`./start_server.sh --port 8000` çalıştırıldığında kurumsal **Tendou Studio Web UI** (`ui/dist/`) otomatik olarak `http://127.0.0.1:8000/` adresinde açılır (canlı 4 Hz TPS sayacı, 52 CLI ayarı, Playground, Zero-Zombie Hot-Swap ve Agent Hub). Yerel **Tauri v2** masaüstü uygulaması için:

```bash
# Tendou Studio Tauri v2 Linux masaüstü uygulamasını derleme ve başlatma:
./scripts/build_desktop_linux.sh
./desktop/src-tauri/target/release/tendou-studio
```

Motoru, CPU çekirdeğini ve testleri yerel olarak derlemek/çalıştırmak için:

```bash
# 1. Tüm motoru, CPU çekirdeğini ve test ikililerini derleme:
./scripts/build_runtime.sh && ./scripts/build_cpu_kernel.sh && ./scripts/build_native_tests.sh

# 2. C++ 46 Kapı Native Testini Çalıştırma (Gates 1–46: CUDA Host Pinning, Boost-Aware & AVX2_x4 0-ULP):
./llama.cpp/build-linux/bin/atlas-native-tests
./llama.cpp/build-linux/bin/atlas-partition-tests

# 3. Python Birim ve Entegrasyon Testlerini Çalıştırma:
./.venv-audit/bin/python -m pytest tests/unit tests/integration/test_openai_server.py tests/integration/test_ipc_resilience.py -q

# 4. Bilgi Grafını Güncelleme (Graphify):
./.venv-audit/bin/graphify update .
```
