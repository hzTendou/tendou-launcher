new_entry = """2026-10-22 | Antigravity / Phase 13 to v0.13.0-stable & Tening Phase 11 Synergy: Dinamik MoE Topolojisi, Universal Auto-Tuner & Çift-Model Gerçek Donanım Benchmarkı | DOĞRULANDI
- İş/değişen dosyalar: `src/atlas/atlas-engine.h`, `src/atlas/atlas-engine.cpp`, `llama.cpp/examples/atlas-engine/atlas-engine.h`, `llama.cpp/examples/atlas-engine/atlas-engine.cpp`, `src/atlas/autotune.py`, `src/atlas/server.py`, `tests/unit/test_autotune.py`, `releases/tendou-v0.13.0-stable/` (`bin/tendou-engine.exe`, DLLs, `source/`, `RELEASE_NOTES.md`), `config/atlas_physical_map_35b_3tier.json`, `scripts/run_v13_stable_synergy_benchmark.py`, `docs/json/benchmark_v13_stable_synergy_report.json`, `VERSION.txt`, `AGENT.md`, `docs/agent-history.md`.
- Kanıt ve sonuç: 1) **C++ Dinamik MoE Topolojisi ve 3-Tier Fiziksel Harita Genelleştirmesi**: `atlas-engine.cpp` içindeki sabit 48x512 katman/uzman ve 1.25 MB kırpma kısıtları kaldırıldı; `total_bytes` parsing desteği ve dinamik expert boyutu getirilerek hem Qwen3.6-35B-A3B (40x256, 1.458 MB / 1.761 MB) hem de Qwen3.8 Flash Next (48x512, 1.253 MB) sıfır-byte kırpma ve sıfır indeks taşmasıyla native motora bağlandı. Tening Phase 11'in 4KB sektör hizalı 3-Tier haritası (`config/atlas_physical_map_35b_3tier.json`) entegre edildi. 2) **Universal Hardware & Topology Auto-Tuning Engine (`src/atlas/autotune.py`)**: NVML GPU sorgusu (VRAM, compute cap), Win32 CPUID SIMD/çekirdek sorgusu (AVX-512 VNNI/VBMI), Windows `GlobalMemoryStatusEx` RAM ve Win32 BypassIO / DirectStorage tespiti ile GGUF başlığı hafif metadata ayrıştırıcısı eklendi. `server.py`'ye `--autotune` ve `--profile auto` entegre edildi (`test_autotune.py` 3 test PASS). 3) **Bağımsız Kararlı Sürüm (`releases/tendou-v0.13.0-stable`)**: `bin/` altında `tendou-engine.exe` (508,416 B), `llama-atlas-engine.exe` ve tüm çalışma zamanı bağımlılıkları (`ggml-cpu.dll`, `ggml-cuda.dll`, `llama.dll`, `llama-common.dll`, `ggml-base.dll`, `ggml.dll`, `mtmd.dll`) self-contained olarak paketlendi. Sürüm `0.13.0-stable` olarak mühürlendi. Tening `tendou_runner.py` v0.13.0-stable'a bağlandı (Tening 29/29 test PASS). 4) **Çift-Model Gerçek Donanım Benchmarkı (`benchmark_v13_stable_synergy_report.json`)**: RTX 5060 + Ryzen 7 donanımında 3 gerçekçi senaryoda (Code Refactoring, Greenfield Rust Ring Buffer, Multi-Turn Tool Stream) uçtan uca ölçüldü: Qwen3.6-35B-A3B: Code Refactor 40.78 tok/s (Config-A) vs 19.82 tok/s (Config-D); Greenfield 44.33 tok/s vs 37.47 tok/s; Multi-Turn Tool 35.70 tok/s vs 21.33 tok/s (VRAM: 4,098.2 MiB / 4,382.4 MiB). Qwen3.8-Flash-Next: Code Refactor 11.46 tok/s (A, TTFT 364 ms) vs 6.48 tok/s (D, TTFT 351 ms); Greenfield 16.73 tok/s vs 12.43 tok/s; Multi-Turn Tool 18.34 tok/s vs 8.96 tok/s (VRAM: 6,878.2 MiB / 7,285.8 MiB < 7600 MiB). 5) **Test ve Kapı Güvencesi**: `atlas-native-tests.exe` (26/26 Gate PASS), `atlas-partition-tests.exe` PASS, pytest 109 passed / 4 skipped (%100 PASS).
- Devam: Phase 14 çoklu-model yerel ajan orkestrasyonu (multi-agent swarm), paylaşımlı VRAM prefix bağlam yönetimi ve cross-model speculative pipeline."""

with open('AGENT.md', 'r', encoding='utf-8') as f:
    text = f.read()

# Update date
text = text.replace('Son güncelleme: 2026-10-20', 'Son güncelleme: 2026-10-22')

# Find the section "### Son devir kayıtları"
marker = '### Son devir kayıtları\n\n'
idx = text.find(marker)
if idx != -1:
    top_part = text[:idx + len(marker)]
    records_part = text[idx + len(marker):]
    
    # Split records by double newline before date
    records = []
    current = []
    for line in records_part.splitlines():
        if line.startswith('2026-10-'):
            if current:
                records.append("\n".join(current))
                current = []
        current.append(line)
    if current:
        records.append("\n".join(current))
    
    print(f"Current records count: {len(records)}")
    # Drop the oldest (last) record (Phase 10)
    kept_records = records[:4]
    
    # Combine: new_entry + kept 4 records (total 5)
    all_5_records = [new_entry] + kept_records
    new_records_text = "\n\n".join(all_5_records).strip() + "\n"
    
    updated_text = top_part + new_records_text
    
    with open('AGENT.md', 'w', encoding='utf-8') as f:
        f.write(updated_text)
        
    line_count = len(updated_text.splitlines())
    print(f"AGENT.md updated successfully. Total lines: {line_count}")
    if line_count <= 180:
        print("Line count <= 180 verified!")
    else:
        print("WARNING: Line count exceeds 180!")
