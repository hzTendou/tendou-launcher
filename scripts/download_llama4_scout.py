"""
Script to download Llama-4-Scout-17B-16E.i1-Q6_K GGUF split parts into models/ directory.
Supports resume, retry, and progress logging.
"""
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

PARTS = [
    {
        "name": "Llama-4-Scout-17B-16E.i1-Q6_K.gguf.part1of2",
        "url": "https://huggingface.co/mradermacher/Llama-4-Scout-17B-16E-i1-GGUF/resolve/main/Llama-4-Scout-17B-16E.i1-Q6_K.gguf.part1of2",
        "size": 45097156608,
    },
    {
        "name": "Llama-4-Scout-17B-16E.i1-Q6_K.gguf.part2of2",
        "url": "https://huggingface.co/mradermacher/Llama-4-Scout-17B-16E-i1-GGUF/resolve/main/Llama-4-Scout-17B-16E.i1-Q6_K.gguf.part2of2",
        "size": 43334989408,
    },
]


def download_part(url: str, dest_path: Path, expected_size: int, chunk_size: int = 16 * 1024 * 1024):
    current_size = dest_path.stat().st_size if dest_path.exists() else 0
    if current_size == expected_size:
        print(f"[DOWNLOAD] {dest_path.name} already complete ({current_size:,} bytes). Skipping.")
        return

    if current_size > expected_size:
        print(f"[DOWNLOAD] {dest_path.name} is larger than expected ({current_size} > {expected_size}). Truncating/restarting.")
        dest_path.unlink()
        current_size = 0

    mode = "ab" if current_size > 0 else "wb"
    headers = {}
    if current_size > 0:
        headers["Range"] = f"bytes={current_size}-"
        print(f"[DOWNLOAD] Resuming {dest_path.name} from {current_size:,} bytes ({current_size / 1024**3:.2f} GiB) ...")
    else:
        print(f"[DOWNLOAD] Starting fresh download of {dest_path.name} ({expected_size / 1024**3:.2f} GiB) ...")

    retries = 10
    while current_size < expected_size and retries > 0:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp, open(dest_path, mode) as f:
                t0 = time.time()
                bytes_downloaded_this_session = 0
                last_log = t0
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    f.flush()
                    current_size += len(chunk)
                    bytes_downloaded_this_session += len(chunk)
                    now = time.time()
                    if now - last_log >= 10:
                        speed = bytes_downloaded_this_session / (now - t0) / (1024 * 1024)
                        pct = (current_size / expected_size) * 100
                        remaining_bytes = expected_size - current_size
                        eta_sec = remaining_bytes / (speed * 1024 * 1024) if speed > 0 else 0
                        print(f"[{dest_path.name}] {pct:.2f}% ({current_size / 1024**3:.2f} / {expected_size / 1024**3:.2f} GiB) - Speed: {speed:.2f} MB/s - ETA: {eta_sec / 60:.1f} min")
                        last_log = now
        except Exception as e:
            retries -= 1
            print(f"[DOWNLOAD-ERROR] {e}. Retrying in 5 seconds... ({retries} retries left)")
            time.sleep(5)
            current_size = dest_path.stat().st_size if dest_path.exists() else 0
            headers["Range"] = f"bytes={current_size}-"
            mode = "ab"

    if current_size == expected_size:
        print(f"[DOWNLOAD-SUCCESS] {dest_path.name} downloaded successfully ({current_size:,} bytes).")
    else:
        raise RuntimeError(f"Failed to complete download of {dest_path.name}: {current_size} / {expected_size}")


def main():
    print("=== TENDOU LAUNCHER: Llama-4 Scout Model Downloader ===")
    for part in PARTS:
        dest = MODELS_DIR / part["name"]
        download_part(part["url"], dest, part["size"])
    print("=== All parts downloaded successfully! ===")


if __name__ == "__main__":
    main()
