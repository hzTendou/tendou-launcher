"""
Atlas Engine — Router Trace Collector

Bu script bir MoE modelinə (Qwen1.5-MoE, OLMoE, Mixtral, Qwen3-MoE və s.)
real promptlar verir və hər layer/token üçün router-in seçdiyi expert-ləri
qeyd edir. Nəticə bir JSONL trace faylıdır ki, sonra analyze_trace.py ilə
cache hit rate / prefetch potential ölçülə bilsin.

İSTİFADƏ:
    python collect_trace.py --model Qwen/Qwen1.5-MoE-A2.7B-Chat --out trace.jsonl

QEYD: Model `output_router_logits=True` dəstəkləməlidir (əksər HF MoE modelləri
dəstəkləyir: Mixtral, Qwen2MoE/Qwen1.5-MoE, OLMoE, DeepSeek-MoE və s.)
Qwen3-MoE buraxılanda da eyni interfeysi dəstəkləyəcək (arxitektura oxşardır).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Kiçik VRAM (8GB) / internet (20GB hotspot) üçün test edilmiş, təhlükəsiz
# safetensors MoE modelləri. GGUF BURADA İŞLƏMİR — bu skript transformers
# üzərindən output_router_logits istifadə edir, GGUF/llama.cpp bunu vermir.
RECOMMENDED_MODELS = {
    "olmoe": "allenai/OLMoE-1B-7B-0924",       # ~14GB fp16 / ~4GB VRAM 4-bit-də — ən təhlükəsiz seçim
    "qwen15moe": "Qwen/Qwen1.5-MoE-A2.7B-Chat",  # ~28GB fp16 — 20GB hotspot limitinə görə RİSKLİDİR
}


DEFAULT_PROMPTS = [
    "Explain how photosynthesis works in simple terms.",
    "Write a Python function that reverses a linked list.",
    "What were the main causes of World War I?",
    "Solve for x: 2x^2 - 5x + 3 = 0, showing your steps.",
    "Write a short poem about the ocean at night.",
    "Explain the difference between TCP and UDP.",
    "Summarize the plot of Hamlet in three sentences.",
    "How does a transformer neural network attention mechanism work?",
    "Give me a recipe for a simple tomato pasta.",
    "What is the capital of Azerbaijan and what is it known for?",
    "Write a SQL query to find the second highest salary in a table.",
    "Explain Newton's three laws of motion with examples.",
    "What are the pros and cons of remote work?",
    "Write a persuasive paragraph about renewable energy.",
    "Debug this code: def add(a, b): return a - b",
]


def load_model(model_name: str, load_in_4bit: bool):
    if model_name.upper().endswith("GGUF") or "gguf" in model_name.lower() or model_name.endswith("-i1"):
        sys.exit(
            f"[XƏTA] '{model_name}' bir GGUF/quantized repo görünür.\n"
            "Bu skript yalnız HF safetensors repo-ları ilə işləyir (transformers +\n"
            "output_router_logits istifadə edir, GGUF bunu dəstəkləmir).\n\n"
            "Bunlardan birini istifadə et:\n"
            f"  --model {RECOMMENDED_MODELS['olmoe']}   (tövsiyə olunan, kiçik və təhlükəsiz)\n"
            f"  --model {RECOMMENDED_MODELS['qwen15moe']}   (daha böyük, ~28GB yükləmə)"
        )

    print(f"[+] Loading tokenizer: {model_name}")
    try:
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        sys.exit(
            f"[XƏTA] Tokenizer yüklənmədi: {model_name}\n"
            f"Səbəb: {e}\n\n"
            "Yoxla: repo adı düzgündür? (huggingface.co/{model} səhifəsi mövcuddurmu?)\n"
            f"Sınamaq üçün: --model {RECOMMENDED_MODELS['olmoe']}"
        )

    kwargs = dict(
        trust_remote_code=True,
        output_router_logits=True,
        device_map="auto",
        low_cpu_mem_usage=True,  # 16GB RAM üçün vacib — ikiqat kopya yaratmasın
    )
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    print(f"[+] Loading model: {model_name} (4bit={load_in_4bit})")
    print("[+] Bu addım repo-nu HF cache-ə (~/.cache/huggingface) endirir — "
          "internet limitindən əvvəl model kartındakı ölçüyə bax.")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except torch.cuda.OutOfMemoryError:
        sys.exit(
            "[XƏTA] VRAM bitdi (8GB kartında).\n"
            "Sına: --max-new-tokens dəyərini azalt, --limit-prompts 5 istifadə et,\n"
            f"ya da kiçik modelə keç: --model {RECOMMENDED_MODELS['olmoe']}"
        )
    except Exception as e:
        sys.exit(f"[XƏTA] Model yüklənmədi: {e}")
    model.eval()
    return tok, model


def top_k_experts(router_logits_layer: torch.Tensor, k: int):
    """router_logits_layer: [num_tokens, num_experts] -> per-token top-k expert ids."""
    topk = torch.topk(router_logits_layer, k=k, dim=-1)
    return topk.indices.tolist()  # [num_tokens][k]


@torch.no_grad()
def trace_prompt(model, tok, prompt: str, max_new_tokens: int, top_k: int):
    """Trace prompt and decode routing with explicit causal phase boundaries.

    The prompt is forwarded once with cache enabled. Decode then feeds one newly
    generated token at a time through the same KV cache, so this does not turn
    trace collection into an O(sequence^2) recomputation.
    """
    input_ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    if input_ids.shape[1] == 0:
        return [], ""

    records = []
    out = model(input_ids, output_router_logits=True, use_cache=True)
    router_logits = out.router_logits
    if router_logits is None:
        raise RuntimeError("Model router_logits qaytarmadı — output_router_logits dəstəklənmir.")

    seq_len = input_ids.shape[1]
    for pos in range(seq_len):
        step_experts = {}
        for layer_idx, layer_logits in enumerate(router_logits):
            if layer_logits is None:
                continue
            row = layer_logits[pos:pos + 1, :]
            step_experts[layer_idx] = top_k_experts(row, top_k)[0]
        records.append({
            "step": pos,
            "phase": "prompt",
            "token_index": pos,
            "token_id": int(input_ids[0, pos].item()),
            "experts_by_layer": step_experts,
        })

    generated = input_ids
    past = out.past_key_values
    decode_index = 0

    # The prompt's final-token logits predict the first generated token.
    next_token = torch.argmax(out.logits[0, -1, :]).reshape(1, 1)
    for _ in range(max_new_tokens):
        if next_token.item() == tok.eos_token_id:
            break

        generated = torch.cat([generated, next_token], dim=1)

        # Run the newly generated token through the router. This is the routing
        # event that will be used to predict the following token.
        out = model(
            next_token,
            output_router_logits=True,
            use_cache=True,
            past_key_values=past,
        )
        past = out.past_key_values
        router_logits = out.router_logits
        if router_logits is None:
            raise RuntimeError("Decode zamanı router_logits qaytarılmadı.")

        step_experts = {}
        for layer_idx, layer_logits in enumerate(router_logits):
            if layer_logits is None:
                continue
            # With KV cache, router logits should contain exactly the new token.
            row = layer_logits[-1:, :]
            step_experts[layer_idx] = top_k_experts(row, top_k)[0]

        records.append({
            "step": len(records),
            "phase": "decode",
            "token_index": decode_index,
            "token_id": int(next_token.item()),
            "experts_by_layer": step_experts,
        })
        decode_index += 1

        next_token = torch.argmax(out.logits[0, -1, :]).reshape(1, 1)

    generated_text = tok.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
    return records, generated_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id, e.g. Qwen/Qwen1.5-MoE-A2.7B-Chat")
    ap.add_argument("--out", default="trace.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=None, help="Override top-k experts (default: model config)")
    ap.add_argument("--no-4bit", action="store_true", help="4-bit quantization istifadə etmə")
    ap.add_argument("--prompts-file", default=None, help="Hər sətirdə bir prompt olan fayl (default: daxili siyahı)")
    ap.add_argument("--limit-prompts", type=int, default=None)
    args = ap.parse_args()

    prompts = DEFAULT_PROMPTS
    if args.prompts_file:
        prompts = Path(args.prompts_file).read_text(encoding="utf-8").strip().splitlines()
    if args.limit_prompts:
        prompts = prompts[: args.limit_prompts]

    tok, model = load_model(args.model, load_in_4bit=not args.no_4bit)

    top_k = args.top_k or getattr(model.config, "num_experts_per_tok", None) or getattr(
        model.config, "num_experts_per_token", None
    )
    if top_k is None:
        raise SystemExit(
            "top-k expert sayını modeldən avtomatik tapa bilmədim — --top-k ilə əl ilə ver."
        )
    print(f"[+] top_k experts per token = {top_k}")

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts):
            t0 = time.time()
            print(f"[{i+1}/{len(prompts)}] Tracing: {prompt[:60]!r}")
            records, generated_text = trace_prompt(
                model, tok, prompt, args.max_new_tokens, top_k
            )
            elapsed = time.time() - t0
            tps = len(records) / elapsed if elapsed > 0 else 0.0
            print(f"    -> {len(records)} tokens in {elapsed:.1f}s ({tps:.2f} tok/s)")

            f.write(
                json.dumps(
                    {
                        "prompt_idx": i,
                        "prompt": prompt,
                        "generated_text": generated_text,
                        "num_layers": len(records[0]["experts_by_layer"]) if records else 0,
                        "top_k": top_k,
                        "tokens_per_second": tps,
                        "records": records,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"[+] Trace saved to {out_path}")


if __name__ == "__main__":
    main()
