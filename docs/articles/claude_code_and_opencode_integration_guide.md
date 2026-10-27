# Tendou Launcher — Claude Code, OpenCode & Cline (NVIDIA NIM) Entegrasyon Kılavuzu

**Sürüm:** v0.14.0-stable  
**Doküman Kodu:** `DOC-INTEG-CLAUDE-OPENCODE-CLINE-14`  
**Tarih:** 2026-10-24  

---

## 1. Genel Bakış ve Üçlü Protokol Mimarisi

**tendou launcher**, tüketici sınıfı donanımlarda büyük MoE (Mixture of Experts) modellerini yüksek hızda çalıştırmak için **OpenCode**, **Claude Code CLI** ve **Cline (NVIDIA NIM / VS Code Agentic Coding)** protokollerini yerel ve tam optimize olarak sunar:

1. **OpenCode & Standart Kodlama Ajanları (`/v1/chat/completions`):**
   - OpenAI uyumlu Chat Completions protokolü.
   - Server-Sent Events (SSE) ile token-token sıfır gecikmeli akış.
   - Araç çağrılarında artırımlı (incremental) JSON argüman yayını.

2. **Claude Code CLI & Anthropic SDK (`/v1/messages` ve `/messages`):**
   - Claude Code'un doğrudan bağlandığı yerel Anthropic Messages API.
   - Anthropic formatındaki `system`, `messages` (metin blokları, `tool_use`, `tool_result`) ve `tools` (`input_schema`) yapılarını Qwen ChatML XML formatına tam çift yönlü dönüştürme.
   - Claude Code SSE akış olayları (`message_start`, `content_block_start`, `content_block_delta` [text_delta & input_json_delta], `content_block_stop`, `message_delta`, `message_stop`).

3. **Cline (NVIDIA NIM / OpenAI Compatible & Anthropic Native XML Modu):**
   - Çok parçalı içerik dizilerinin (`[{"type": "text", "text": "..."}]`) JSON dizi parantezleri sızmadan saf metne düzleştirilmesi (`extract_message_content_text`) ve `<environment_details>` bloklarında **%100 deterministik KV Prefix Cache isabeti**.
   - Cline yerel XML araç protokolü (`<read_file>`, `<replace_in_file>`, `<write_to_file>`, `<execute_command>`, `<search_files>`, `<list_files>`, `<attempt_completion>`) algılaması (`is_cline_xml_prompt`).
   - **Tek-Araç-Kuralı Kesici (`truncate_after_first_cline_xml_tool`)**: İlk XML araç etiketi kapandığı anda üretimi sıfır gecikmeyle durdurarak sahte `<environment_details>` halüsinasyonlarını önler ve KV önbellek telemetrisini korur.
   - **Qwen `<tool_call>` -> Cline XML Köprüsü (`convert_qwen_tool_call_to_cline_xml`)**: Model alışkanlıkla JSON `<tool_call>` üretirse anında Cline XML formatına çevirir.
   - **İş Yükü Duyarlı Spekülatif Ufuk ($d^*$)**: Dosya okuma/keşif turlarında `d*=0` (sıfır red cezası), `replace_in_file` (`<<<<<<< SEARCH` / `>>>>>>> REPLACE`) ve kod refaktör turlarında otomatik `PLD d*=3` aktivasyonu.

---

## 2. Claude Code CLI Entegrasyonu (Adım Adım)

Claude Code terminal aracını Tendou Launcher'a bağlamak için ekstra bir proxy veya aracı yazılıma gerek yoktur.

### Adım 1: Tendou Launcher Sunucusunu Başlatın
```powershell
# PowerShell üzerinden sunucuyu başlatın
.venv-audit\Scripts\python.exe src\atlas\server.py --port 8000
```
Sunucu hazır olduğunda `http://127.0.0.1:8000/v1` adresinde dinlemeye başlar.

### Adım 2: Claude Code Ortam Değişkenlerini Ayarlayın
Yeni bir terminalde ortam değişkenlerini tanımlayın:

**Windows PowerShell:**
```powershell
$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:8000"
$env:ANTHROPIC_API_KEY = "tendou-local"

# Claude Code CLI'yi başlatın
claude
```

**Bash / WSL / macOS:**
```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:8000"
export ANTHROPIC_API_KEY="tendou-local"

claude
```

Claude Code CLI, Tendou Launcher'ın `/v1/messages` ve `/messages` uç noktalarını doğal bir Anthropic sunucusu gibi algılar. Dosya okuma (`read_file`), düzenleme (`edit_file`), komut çalıştırma (`bash`) ve arama (`grep`, `glob`) araçlarını yerel MoE modeli üzerinden sıfır gecikmeyle yürütür.

---

## 3. OpenCode Entegrasyonu

Tendou Launcher kök dizininde hazır bulunan [`opencode.json`](file:///c:/Users/Ali/WebProjects/tendou-launcher/opencode.json) yapılandırması üzerinden OpenCode doğrudan çalışır:

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
        }
      }
    }
  }
}
```

---

## 4. Cline (NVIDIA NIM / VS Code Agentic Coding) Entegrasyonu

Cline eklentisini veya NVIDIA NIM tabanlı Cline ajan istemcisini **tendou launcher**'a iki farklı modda bağlayabilirsiniz:

### Seçenek A: OpenAI Compatible / NVIDIA NIM Modu (Önerilen)
- **API Provider:** `OpenAI Compatible` (veya `NVIDIA NIM`)
- **Base URL:** `http://127.0.0.1:8000/v1`
- **API Key:** `tendou-local`
- **Model ID:** `qwen3.6-35b-a3b` veya `qwen3.8-flash-next`

### Seçenek B: Anthropic Protokol Modu
- **API Provider:** `Anthropic`
- **Base URL:** `http://127.0.0.1:8000`
- **API Key:** `tendou-local`
- **Model ID:** `qwen3.6-35b-a3b`

Tendou Launcher, Cline'ın `stream_options: {"include_usage": true}` parametresini ve çok parçalı `content` dizilerini yerel olarak destekler; Cline arayüzündeki Context Window / Cache Read (`cache_read_input_tokens` & `cached_tokens`) sayaçları gerçek zamanlı güncellenir.

---

## 5. Donanım ve Üçlü-Protokol Benchmark Sonuçları (`v0.14.0-stable`)

Yapılan resmi donanım testlerinde (AMD Ryzen 7 260 + NVIDIA RTX 5060 Laptop 8GB + 16GB RAM) her iki model de **OpenCode**, **Claude Code** ve **Cline** protokollerinde gerçekçi kodlama senaryolarında doğrulanmıştır:

| Model | İş Yükü / Senaryo | OpenCode Decode / Efektif TPS (Warm TTFT) | Claude Code Decode / Efektif TPS (Warm TTFT) | Cline Decode / Efektif TPS (Warm TTFT) | VRAM Kullanımı |
|---|---|---|---|---|---|
| **Qwen3.6-35B-A3B** (35B UD-IQ4_XS) | 1. Code Refactoring (`PLD d*=3`) | **60.94 / 50.63 tok/s** (39.3 ms) | **61.88 / 56.34 tok/s** (14.7 ms) | **59.26 / 54.79 tok/s** (15.4 ms) | 4192 MiB |
| **Qwen3.6-35B-A3B** (35B UD-IQ4_XS) | 2. Greenfield Synthesis (`d*=0`) | **52.49 / 52.63 tok/s** (16.8 ms) | **54.09 / 53.76 tok/s** (17.5 ms) | **54.60 / 54.79 tok/s** (15.0 ms) | 4192 MiB |
| **Qwen3.6-35B-A3B** (35B UD-IQ4_XS) | 3. Agentic Tool Stream (`d*=0`) | **70.18 / 53.16 tok/s** (232.2 ms) | **72.12 / 55.05 tok/s** (218.8 ms) | **56.60 / 55.30 tok/s** (37.4 ms) | 4192 MiB |
| **Qwen3.8-Flash-Next** (IQ3_XXS) | 1. Code Refactoring (`PLD d*=3`) | **26.16 / 23.76 tok/s** (43.3 ms) | **26.26 / 24.90 tok/s** (21.3 ms) | **27.85 / 26.37 tok/s** (20.6 ms) | 6900 MiB |
| **Qwen3.8-Flash-Next** (IQ3_XXS) | 2. Greenfield Synthesis (`d*=0`) | **25.21 / 25.17 tok/s** (41.4 ms) | **22.91 / 23.17 tok/s** (20.5 ms) | **24.67 / 24.97 tok/s** (20.1 ms) | 6900 MiB |
| **Qwen3.8-Flash-Next** (IQ3_XXS) | 3. Agentic Tool Stream (`d*=0`) | **25.73 / 25.68 tok/s** (41.8 ms) | **25.86 / 26.07 tok/s** (22.6 ms) | **25.91 / 25.89 tok/s** (39.6 ms) | 6900 MiB |

> [!NOTE]
> Her iki modelde ve her üç protokolde (**OpenCode**, **Claude Code**, **Cline**) **stabil 20+ Efektif TPS** hedefi (35B modelinde **50.63 – 56.34 Efektif TPS / 72.12 Peak Decode TPS**, Qwen3.8-Flash-Next modelinde **23.17 – 26.37 Efektif TPS / 27.85 Peak Decode TPS**) ve **14.7 – 43.3 ms** sıfır-prefill Warm TTFT önbellek hızı (**422.5x'e varan TTFT hızlanması**) eksiksiz olarak mühürlenmiştir.
