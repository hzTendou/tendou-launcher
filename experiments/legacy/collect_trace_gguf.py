"""
Atlas Engine — GGUF Router Trace Collector (llama.cpp/atlas-trace binary üçün)

collect_trace.py-nin GGUF versiyası. transformers/safetensors əvəzinə,
`llama-eval-callback` (atlas-trace.cpp-dən compile olunmuş) binary-sini
hər prompt üçün subprocess kimi çağırır, stdout-dan ATLAS_MOE sətirlərini
parse edir və nəticəni analyze_trace.py-nin gözlədiyi trace.jsonl formatına
çevirir.

TƏLƏB OLUNAN QURAŞDIRMA (bir dəfəlik):
  1. git clone https://github.com/ggml-org/llama.cpp
  2. examples/eval-callback/eval-callback.cpp faylını atlas-trace.cpp
     məzmunu ilə əvəz et
  3. cmake -B build -DCMAKE_BUILD_TYPE=Release   (GPU üçün -DGGML_CUDA=ON əlavə et)
  4. cmake --build build --config Release -j --target llama-eval-callback
  5. Nəticə: build/bin/llama-eval-callback(.exe)

İSTİFADƏ:
    python collect_trace_gguf.py \
        --binary ./llama.cpp/build/bin/llama-eval-callback.exe \
        --model ./models/Huihui-Qwen3-30B-A3B-Instruct-2507-abliterated.i1-IQ1_S.gguf \
        --out trace_gguf.jsonl \
        --max-new-tokens 64 \
        --ngl 99
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_PROMPTS = [
    # Ümumi (baseline, model əsasən bu tip data ilə öyrədilib)
    "Explain how photosynthesis works in simple terms.",
    "Write a Python function that reverses a linked list.",
    "What were the main causes of World War I?",
    "Solve for x: 2x^2 - 5x + 3 = 0, showing your steps.",
    "How does a transformer neural network attention mechanism work?",
    "Explain Newton's three laws of motion with examples.",
    # Azərbaycanca — Atlas Academy-nin real domenini əks etdirir.
    # Router-in expert-locality-si dil/domendən asılı ola bilər, ona görə
    # go/no-go qərarı üçün MƏHZ bu tip promptlar üzərində ölçmək vacibdir.
    "Kvadrat tənliyi izah et: ax^2 + bx + c = 0 necə həll olunur, addım-addım göstər.",
    "Nyutonun hərəkət qanunlarını sadə dildə izah et və gündəlik həyatdan nümunə göstər.",
    "Bir cismin sürətlənməsi 5 m/s² və başlanğıc sürəti 0 olarsa, 10 saniyədən sonra sürəti nə qədər olar?",
    "Fotosintez prosesini orta məktəb şagirdi üçün izah et.",
    "Üçbucağın daxili bucaqlarının cəmi niyə 180 dərəcədir, sübut et.",
    "Ohm qanununu izah et və bir dövrədə cərəyanı necə hesabladığını göstər.",
]


ATLAS_LINE_RE = re.compile(
    r"^ATLAS_MOE phase=(prompt|decode) call=(\d+) layer=(-?\d+) n_used=(\d+) n_tok=(\d+) vals=(.*)$"
)
ATLAS_DONE_RE = re.compile(
    r"^ATLAS_DONE prompt_tokens=(\d+) n_decoded=(\d+) seconds=([\d.]+) tok_per_sec=([\d.]+)$"
)


def run_binary(binary: str, model: str, prompt: str, n_predict: int, ngl: int,
                ctx_size: int, extra_args: list[str]) -> str:
    cmd = [
        binary,
        "-m", model,
        "-p", prompt,
        "-n", str(n_predict),
        "-ngl", str(ngl),
        "--ctx-size", str(ctx_size),
    ] + extra_args

    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        sys.stderr.write(f"[XƏBƏRDARLIQ] binary qeyri-sıfır kodla çıxdı ({result.returncode})\n")
        sys.stderr.write("--- stderr (son 40 sətir) ---\n")
        sys.stderr.write("\n".join(result.stderr.splitlines()[-40:]) + "\n")
    return result.stdout


def parse_stdout(stdout: str):
    """
    ATLAS_MOE sətirlərini parse edib, call_idx -> layer_idx -> [[expert...], [expert...], ...]
    (hər call üçün n_tok qədər token sütunu) strukturuna toplayır.
    """
    calls = {}  # call_idx -> {layer_idx: [[e0..eK-1], [e0..eK-1], ...]}  (uzunluq = n_tok)
    done_info = None

    for line in stdout.splitlines():
        m = ATLAS_LINE_RE.match(line)
        if m:
            phase = m.group(1)
            call_idx = int(m.group(2))
            layer_idx = int(m.group(3))
            n_used = int(m.group(4))
            n_tok = int(m.group(5))
            vals_str = m.group(6)
            vals = [int(x) for x in vals_str.split(",")] if vals_str else []
            if len(vals) != n_used * n_tok:
                sys.stderr.write(
                    f"[XƏTA] call={call_idx} layer={layer_idx}: gözlənilən {n_used*n_tok} "
                    f"dəyər, tapılan {len(vals)} — bu sətir atlanır\n"
                )
                continue
            token_groups = [vals[t * n_used:(t + 1) * n_used] for t in range(n_tok)]
            calls.setdefault(call_idx, {})[layer_idx] = {"phase": phase, "tokens": token_groups}
            continue

        m2 = ATLAS_DONE_RE.match(line)
        if m2:
            done_info = {
                "prompt_tokens": int(m2.group(1)),
                "n_decoded": int(m2.group(2)),
                "seconds": float(m2.group(3)),
                "tok_per_sec": float(m2.group(4)),
            }

    return calls, done_info


def build_records(calls: dict, done_info: dict):
    """Convert callback calls into explicit prompt/decode records."""
    if not calls:
        return [], 0

    records = []
    n_layers = len(calls.get(0, {}))
    prompt_token_count = 0

    # call=0 is the full prompt decode call.
    if 0 in calls:
        layer0 = calls[0]
        lengths = {layer: len(info["tokens"]) for layer, info in layer0.items()}
        prompt_token_count = min(lengths.values()) if lengths else 0
        if lengths and len(set(lengths.values())) != 1:
            print(
                f"    -> XƏBƏRDARLIQ: prompt call=0 layer-ləri fərqli n_tok verir: "
                f"min={prompt_token_count}, max={max(lengths.values())}; ortaq hissə istifadə ediləcək."
            )
        for t in range(prompt_token_count):
            experts_by_layer = {
                str(layer): info["tokens"][t]
                for layer, info in layer0.items()
                if t < len(info["tokens"])
            }
            records.append({
                "step": len(records),
                "phase": "prompt",
                "token_index": t,
                "experts_by_layer": experts_by_layer,
            })

    # call=1,2,... are one-token generation decode calls.
    decode_index = 0
    for call_idx in sorted(k for k in calls if k != 0):
        layer_map = calls[call_idx]
        experts_by_layer = {
            str(layer): info["tokens"][0]
            for layer, info in layer_map.items()
            if info["tokens"]
        }
        if not experts_by_layer:
            continue
        records.append({
            "step": len(records),
            "phase": "decode",
            "token_index": decode_index,
            "experts_by_layer": experts_by_layer,
        })
        decode_index += 1

    return records, n_layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True, help="llama-eval-callback(.exe) yolu")
    ap.add_argument("--model", required=True, help="GGUF fayl yolu (lokal disk)")
    ap.add_argument("--out", default="trace_gguf.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--ngl", type=int, default=99, help="GPU-ya yüklənəcək layer sayı (99=hamısı)")
    ap.add_argument("--ctx-size", type=int, default=2048)
    ap.add_argument("--prompts-file", default=None, help="Hər sətirdə bir prompt (default: daxili siyahı)")
    ap.add_argument("--limit-prompts", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=None,
                     help="Yalnız README-də qeyd üçün; real top-k modeldən avtomatik gəlir")
    args = ap.parse_args()

    if not Path(args.binary).exists():
        sys.exit(f"[XƏTA] binary tapılmadı: {args.binary}\nƏvvəlcə README-dəki addımlarla compile et.")
    if not Path(args.model).exists():
        sys.exit(f"[XƏTA] model faylı tapılmadı: {args.model}")

    prompts = DEFAULT_PROMPTS
    if args.prompts_file:
        prompts = Path(args.prompts_file).read_text(encoding="utf-8").strip().splitlines()
    if args.limit_prompts:
        prompts = prompts[: args.limit_prompts]

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts):
            print(f"[{i+1}/{len(prompts)}] Tracing: {prompt[:60]!r}")
            t0 = time.time()
            stdout = run_binary(
                args.binary, args.model, prompt,
                args.max_new_tokens, args.ngl, args.ctx_size, [],
            )
            elapsed = time.time() - t0

            calls, done_info = parse_stdout(stdout)
            if not calls:
                print(f"    -> XƏBƏRDARLIQ: heç bir ATLAS_MOE sətri tapılmadı, bu prompt atlanır "
                      f"(model MoE arxitekturalı deyilmi? binary səhv çıxıbmı? stderr-ə bax)")
                continue

            records, n_layers = build_records(calls, done_info)
            tps = done_info["tok_per_sec"] if done_info else (
                done_info["n_decoded"] / elapsed if done_info else 0.0
            )

            print(f"    -> {len(records)} token trace olundu ({n_layers} layer), "
                  f"{elapsed:.1f}s real vaxt"
                  + (f", {done_info['tok_per_sec']:.2f} tok/s (generasiya)" if done_info else ""))

            f.write(
                json.dumps(
                    {
                        "prompt_idx": i,
                        "prompt": prompt,
                        "generated_text": "(qeyd olunmadı — yalnız router trace toplanıb)",
                        "num_layers": n_layers,
                        "top_k": len(next(iter(calls[0].values()))["tokens"][0]) if 0 in calls and calls[0] else None,
                        "tokens_per_second": tps,
                        "prompt_token_count": sum(r.get("phase") == "prompt" for r in records),
                        "decode_token_count": sum(r.get("phase") == "decode" for r in records),
                        "trace_format_version": 2,
                        "records": records,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"[+] Trace saved to {out_path}")


if __name__ == "__main__":
    main()
