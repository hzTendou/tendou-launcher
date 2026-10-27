import os
import sys
import shutil
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKUP_TIMESTAMP = "20261020_123910"
BACKUP_DIR = ROOT / "backups" / f"phase10_stable_{BACKUP_TIMESTAMP}"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

FILES_TO_BACKUP = [
    "src/atlas/atlas-engine.h",
    "llama.cpp/examples/atlas-engine/atlas-engine.h",
    "llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c",
    "llama.cpp/ggml/include/ggml-cpu.h",
    "llama.cpp/ggml/src/ggml-cpu/quants.h",
    "tests/native/test_async_prefetch.cpp",
    "tests/phases/phase10_context_vnni_mtp_directstorage/test_phase10_architecture.py",
]

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

manifest_lines = []
print(f"Creating Phase 10 Backup at: {BACKUP_DIR}")

for rel_path in FILES_TO_BACKUP:
    src_file = ROOT / rel_path
    if not src_file.exists():
        raise FileNotFoundError(f"Source file not found: {src_file}")
    
    dst_file = BACKUP_DIR / rel_path
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_file, dst_file)
    
    src_hash = sha256_file(src_file)
    dst_hash = sha256_file(dst_file)
    
    if src_hash != dst_hash:
        raise ValueError(f"Hash mismatch for {rel_path}: {src_hash} != {dst_hash}")
        
    print(f"  [OK] {rel_path} -> SHA256: {src_hash}")
    manifest_lines.append(f"{src_hash}  {rel_path}")

manifest_path = BACKUP_DIR / "MANIFEST.txt"
with open(manifest_path, "w", encoding="utf-8") as f:
    f.write("\n".join(manifest_lines) + "\n")

print(f"Manifest written to: {manifest_path}")
