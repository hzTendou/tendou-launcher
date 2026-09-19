"""Single-slot OpenAI serving through the patched native speculative runtime."""
import argparse
import os
from pathlib import Path
import subprocess
import sysconfig

from .server import DEFAULT_EXE_PATH, DEFAULT_MODEL_ID, DEFAULT_MODEL_PATH


def make_launch(args):
    executable = Path(args.exe_path)
    if not executable.is_file() or not Path(args.model_path).is_file():
        raise ValueError("Native server executable or model GGUF is missing")
    env = os.environ.copy()
    env["ATLAS_HTTP_CLOSE"] = "1"
    cuda = Path(sysconfig.get_path("purelib")) / "nvidia/cu13/bin/x86_64"
    env["PATH"] = str(cuda) + os.pathsep + env.get("PATH", "")
    layers = "|".join(str(i) for i in range(2, 48))
    placement = rf"blk\.({layers})\.ffn_(up|down|gate_up|gate)_exps=CPU"
    command = [str(executable), "-m", args.model_path, "--alias", DEFAULT_MODEL_ID,
               "--host", args.host, "--port", str(args.port), "-np", "1",
               "-c", str(args.ctx_size), "-b", "256", "-ub", "64", "-ngl", "99",
               "-t", str(args.threads), "-tb", str(args.threads), "-fa", "on",
               "--fit", "off", "--load-mode", "mmap", "-ot", placement]
    if args.mtp != "off":
        head = Path(args.mtp_path) if args.mtp_path else next(iter(sorted(
            (Path.home()/".cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots").glob(
                "*/MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf"))), None)
        if head is None or not head.is_file():
            raise ValueError("Pass --mtp-path with a compatible Qwen3.8 MTP GGUF")
        env["ATLAS_MTP_PATH"] = str(head)
        head_placement = r"blk\.48\..*=CPU" if args.mtp == "ram" else r"blk\.48\.ffn_.*_exps=CPU"
        command += ["-ot", head_placement, "--spec-type", "draft-mtp",
                    "--spec-draft-n-max", str(args.draft_n), "--spec-draft-p-min", "0.28",
                    "-td", str(args.threads), "-tbd", str(args.threads)]
    else:
        command += ["--spec-type", "none"]
    return command, env


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exe-path", default=str(Path(DEFAULT_EXE_PATH).with_name("llama-server.exe")))
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--mtp-path")
    p.add_argument("--mtp", choices=["off", "ram", "dense-gpu"], default="ram")
    p.add_argument("--draft-n", type=int, choices=range(1, 9), default=1)
    p.add_argument("--ctx-size", type=int, default=2048)
    p.add_argument("--threads", type=int, default=14)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.ctx_size < 128 or args.ctx_size > 2048 or args.threads < 1:
        p.error("This experimental preset supports context 128..2048 and positive threads")
    try:
        command, env = make_launch(args)
    except ValueError as exc:
        p.error(str(exc))
    print(f"tendou launcher native API: http://{args.host}:{args.port}/v1 (one slot, MTP={args.mtp})", flush=True)
    raise SystemExit(subprocess.call(command, env=env))
