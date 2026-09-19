"""Real native-router MTP sweep; failed/mismatched outputs never become speed wins."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import sysconfig
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.atlas.server import DEFAULT_EXE_PATH, DEFAULT_MODEL_PATH


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="experiments/runtime_audit/ram16/mtp")
    p.add_argument("--tokens", type=int, default=48)
    p.add_argument("--prompt", default="Newton's first law states that")
    p.add_argument("--modes", nargs="+", choices=["off", "ram1", "ram2", "ram4", "vram1", "vram2", "vram4"], default=["off", "ram1", "ram2", "ram4", "vram2"])
    p.add_argument("--mtp-path")
    p.add_argument("engine_args", nargs=argparse.REMAINDER)
    a = p.parse_args()
    if a.engine_args[:1] == ["--"]:
        a.engine_args = a.engine_args[1:]
    if a.tokens <= 0:
        p.error("--tokens must be positive")
    if a.modes[0] != "off":
        a.modes.insert(0, "off")
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    cuda_bin = Path(sysconfig.get_path("purelib"))/"nvidia/cu13/bin/x86_64"
    env["PATH"] = str(cuda_bin) + os.pathsep + env.get("PATH", "")
    rows = []
    reference = None
    for mode in a.modes:
        extra = ["--atlas-mtp", "off"] if mode == "off" else [
            "--atlas-mtp", "vram" if mode.startswith("vram") else "ram",
            "--atlas-mtp-draft-n", mode[-1], "--atlas-adaptive", "0"]
        if a.mtp_path:
            extra += ["--atlas-mtp-path", a.mtp_path]
        cmd = [DEFAULT_EXE_PATH, "-m", DEFAULT_MODEL_PATH, "-p", a.prompt,
               "-n", str(a.tokens), "--temp", "0", "--seed", "42",
               "-t", "14", "-c", "2048", "--atlas-k", "0"] + extra + a.engine_args
        started = time.perf_counter()
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=600)
        (out/f"{mode}.log").write_text(proc.stdout + "\nSTDERR:\n" + proc.stderr, encoding="utf-8")
        match = re.search(r"^\[ATLAS_RESULT\] (.+)$", proc.stdout, re.M)
        row = dict(mode=mode, command=cmd, wall_s=time.perf_counter()-started,
                   returncode=proc.returncode, prompt=a.prompt,
                   exe_sha256=hashlib.sha256(Path(DEFAULT_EXE_PATH).read_bytes()).hexdigest(),
                    runtime_dll_sha256={n:hashlib.sha256((Path(DEFAULT_EXE_PATH).parent/n).read_bytes()).hexdigest()
                                        for n in ("llama.dll", "llama-common.dll")})
        if match:
            row.update(json.loads(match.group(1)))
        row["valid"] = (proc.returncode == 0 and bool(row.get("token_ids"))
                        and not row.get("failed", True) and not row.get("diagnostic_run", False))
        if mode == "off" and row["valid"]:
            reference = row["token_ids"]
        row["general_quality_status"] = "NOT_EVALUATED"
        row["exact_reference_match"] = row["valid"] and reference is not None and row["token_ids"] == reference
        if row["valid"]:
            row["effective_tps"] = len(row["token_ids"]) * 1000 / row["request_ms"]
            row["decode_tps"] = ((len(row["token_ids"])-1) * 1000 / row["decode_span_ms"]
                                  if len(row["token_ids"]) > 1 else None)
        rows.append(row)
        (out/"summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(json.dumps({k:v for k,v in row.items() if k not in ("command", "token_ids")}), flush=True)
    if any(not r["valid"] or not r["exact_reference_match"] for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
