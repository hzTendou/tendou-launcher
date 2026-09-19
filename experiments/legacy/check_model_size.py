"""
Atlas Engine — Model Ölçüsü Yoxlayıcı

Bir HF repo-nu ENDIRMƏDƏN ƏVVƏL, safetensors fayllarının cəm ölçüsünü göstərir.
20GB hotspot limiti olan mühitdə boş yerə yarımçıq download etməmək üçün.

İSTİFADƏ:
    python check_model_size.py --model allenai/OLMoE-1B-7B-0924
"""

import argparse
import sys

from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit-gb", type=float, default=20.0,
                     help="Xəbərdarlıq üçün limit (default: 20GB hotspot)")
    args = ap.parse_args()

    api = HfApi()
    try:
        info = api.model_info(args.model, files_metadata=True)
    except Exception as e:
        sys.exit(f"[XƏTA] Repo tapılmadı ya da əlçatan deyil: {args.model}\n{e}")

    total_bytes = 0
    relevant_exts = (".safetensors", ".bin", ".json", ".model", ".txt")
    print(f"[+] {args.model} fayl siyahısı:\n")
    for sibling in info.siblings:
        size = sibling.size or 0
        total_bytes += size
        if sibling.rfilename.endswith((".safetensors", ".bin")):
            print(f"    {sibling.rfilename:<50} {size / 1e9:>7.2f} GB")

    total_gb = total_bytes / 1e9
    print(f"\n[+] Təxmini cəm ölçü: {total_gb:.2f} GB")

    if total_gb > args.limit_gb:
        print(f"[!] XƏBƏRDARLIQ: {total_gb:.1f}GB > {args.limit_gb}GB limitindən böyükdür.")
        print("    Bu repo hotspot limitinə sığmaya bilər.")
    else:
        print(f"[OK] {args.limit_gb}GB limitinə sığır.")


if __name__ == "__main__":
    main()
