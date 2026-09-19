"""
Atlas Engine Qwen3.8 Flash-Next — strict one-shot GGUF router trace collector.

This tool does NOT silently accept malformed routing output. It validates the trace
against the model's GGUF metadata:
  - architecture must be qwen4exp unless --allow-unknown-architecture
  - exactly block_count MoE layer records are expected for every traced token
  - n_used must equal expert_used_count
  - every expert id must be in [0, expert_count)
  - prompt/decode phase and call structure must be internally consistent

Output is written atomically: a failed prompt or validation failure never leaves a
partial trace at --out. A sidecar manifest records model fingerprint and validation.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, subprocess, sys, tempfile, time
from pathlib import Path

from gguf_utils import parse_gguf

DEFAULT_PROMPTS = [
    "Explain how photosynthesis works in simple terms.",
    "Write a Python function that reverses a linked list.",
    "Solve for x: 2x^2 - 5x + 3 = 0, showing your steps.",
    "Explain Newton's three laws of motion with examples.",
    "How does a transformer neural network attention mechanism work?",
    "Kvadrat tənliyi izah et: ax^2 + bx + c = 0 necə həll olunur, addım-addım göstər.",
    "Nyutonun hərəkət qanunlarını sadə dildə izah et və gündəlik həyatdan nümunə göstər.",
    "Bir cismin sürətlənməsi 5 m/s² və başlanğıc sürəti 0 olarsa, 10 saniyədən sonra sürəti nə qədər olar?",
    "Fotosintez prosesini orta məktəb şagirdi üçün izah et.",
    "Ohm qanununu izah et və bir dövrədə cərəyanı necə hesabladığını göstər.",
]

LINE = re.compile(r"^ATLAS_MOE phase=(prompt|decode) call=(\d+) layer=(-?\d+) n_used=(\d+) n_tok=(\d+) vals=(.*)$")
DONE = re.compile(r"^ATLAS_DONE prompt_tokens=(\d+) n_decoded=(\d+) seconds=([\d.]+) tok_per_sec=([\d.]+)$")

class TraceError(RuntimeError):
    pass

def pick(meta, suffix):
    for k, v in meta.items():
        if k.endswith(suffix):
            return int(v) if isinstance(v, (int, float)) else v
    return None

def fingerprint(path: Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(8*1024*1024))
        size=path.stat().st_size
        if size>8*1024*1024:
            f.seek(max(0,size-8*1024*1024))
            h.update(f.read(8*1024*1024))
    return {"size_bytes": path.stat().st_size, "edge_sha256": h.hexdigest()}

def parse_profile(model):
    p=parse_gguf(model); m=p["metadata"]
    arch=m.get("general.architecture")
    profile={
        "architecture":arch,
        "name":m.get("general.name"),
        "quantization":m.get("general.file_type"),
        "block_count":pick(m,".block_count"),
        "expert_count":pick(m,".expert_count"),
        "expert_used_count":pick(m,".expert_used_count"),
        "context_length":pick(m,".context_length"),
        "gguf_version":p["version"],
        "tensor_count":p["tensor_count"],
    }
    missing=[k for k in ("block_count","expert_count","expert_used_count") if not profile[k]]
    if missing: raise TraceError(f"GGUF metadata missing required fields: {missing}")
    if profile["expert_used_count"]>profile["expert_count"]:
        raise TraceError("expert_used_count > expert_count")
    return profile

def run(binary, model, prompt, n_predict, ngl, ctx, extra):
    cmd=[str(binary),"-m",str(model),"-p",prompt,"-n",str(n_predict),"-ngl",str(ngl),"--ctx-size",str(ctx),*extra]
    r=subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise TraceError(f"trace binary failed with exit code {r.returncode}\nSTDERR:\n{r.stderr[-8000:]}")
    return r.stdout, r.stderr

def parse(stdout, profile):
    calls={}; done=None
    for raw in stdout.splitlines():
        m=LINE.match(raw)
        if m:
            phase,call,layer,k,ntok,vals=m.groups()
            call=int(call); layer=int(layer); k=int(k); ntok=int(ntok)
            vals=[] if not vals else [int(x) for x in vals.split(",")]
            if k != profile["expert_used_count"]:
                raise TraceError(f"call={call} layer={layer}: n_used={k}, expected={profile['expert_used_count']}")
            if ntok < 1 or len(vals)!=k*ntok:
                raise TraceError(f"call={call} layer={layer}: malformed vals count {len(vals)} != {k}*{ntok}")
            if layer<0 or layer>=profile["block_count"]:
                raise TraceError(f"invalid layer id {layer}")
            if any(e<0 or e>=profile["expert_count"] for e in vals):
                raise TraceError(f"call={call} layer={layer}: expert id out of range")
            if any(len(set(vals[i:i+k])) != k for i in range(0,len(vals),k)):
                raise TraceError(f"call={call} layer={layer}: duplicate expert within top-k")
            if call in calls and layer in calls[call]:
                raise TraceError(f"duplicate routing record call={call} layer={layer}")
            calls.setdefault(call,{})[layer]=(phase,[vals[i:i+k] for i in range(0,len(vals),k)])
            continue
        d=DONE.match(raw)
        if d:
            if done is not None: raise TraceError("duplicate ATLAS_DONE")
            done={"prompt_tokens":int(d.group(1)),"n_decoded":int(d.group(2)),
                  "seconds":float(d.group(3)),"tok_per_sec":float(d.group(4))}
    if not calls: raise TraceError("no ATLAS_MOE records found")
    if done is None: raise TraceError("ATLAS_DONE missing; refusing incomplete trace")
    return calls,done

def validate_and_build(calls, done, profile):
    if 0 not in calls: raise TraceError("prompt call=0 missing")
    expected=set(range(profile["block_count"]))
    records=[]
    prompt_layers=calls[0]
    if set(prompt_layers)!=expected:
        missing=sorted(expected-set(prompt_layers)); extra=sorted(set(prompt_layers)-expected)
        raise TraceError(f"prompt layer coverage invalid: missing={missing[:10]} extra={extra[:10]}")
    prompt_phase={x[0] for x in prompt_layers.values()}
    if prompt_phase != {"prompt"}: raise TraceError(f"call=0 phases invalid: {prompt_phase}")
    lens={len(x[1]) for x in prompt_layers.values()}
    if len(lens)!=1: raise TraceError(f"prompt layers have inconsistent token counts: {sorted(lens)}")
    n_prompt=lens.pop()
    if n_prompt != done["prompt_tokens"]:
        raise TraceError(f"prompt token mismatch: routing={n_prompt}, done={done['prompt_tokens']}")
    for t in range(n_prompt):
        records.append({"step":len(records),"phase":"prompt","token_index":t,
                        "experts_by_layer":{str(l):prompt_layers[l][1][t] for l in range(profile["block_count"])}})
    decode_calls=sorted(c for c in calls if c!=0)
    if decode_calls != list(range(1, done["n_decoded"]+1)):
        raise TraceError(f"decode calls not contiguous or count mismatch: got {decode_calls[:5]}... expected 1..{done['n_decoded']}")
    for idx,c in enumerate(decode_calls):
        lm=calls[c]
        if set(lm)!=expected: raise TraceError(f"decode call={c}: incomplete layer coverage")
        if {x[0] for x in lm.values()} != {"decode"}: raise TraceError(f"decode call={c}: invalid phase")
        if any(len(x[1])!=1 for x in lm.values()): raise TraceError(f"decode call={c}: expected exactly one routed token per layer")
        records.append({"step":len(records),"phase":"decode","token_index":idx,
                        "experts_by_layer":{str(l):lm[l][1][0] for l in range(profile["block_count"])}})
    return records

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--binary",required=True); ap.add_argument("--model",required=True)
    ap.add_argument("--out",default="trace_qwen38.jsonl")
    ap.add_argument("--manifest",default=None)
    ap.add_argument("--max-new-tokens",type=int,default=64)
    ap.add_argument("--ngl",type=int,default=99); ap.add_argument("--ctx-size",type=int,default=4096)
    ap.add_argument("--prompts-file"); ap.add_argument("--limit-prompts",type=int)
    ap.add_argument("--allow-unknown-architecture",action="store_true")
    ap.add_argument("--extra-arg",action="append",default=[])
    args=ap.parse_args()
    binary=Path(args.binary); model=Path(args.model); out=Path(args.out)
    if not binary.exists(): raise SystemExit(f"binary not found: {binary}")
    if not model.exists(): raise SystemExit(f"model not found: {model}")
    profile=parse_profile(model)
    if profile["architecture"]!="qwen4exp" and not args.allow_unknown_architecture:
        raise SystemExit(f"unexpected architecture {profile['architecture']!r}; expected 'qwen4exp'")
    prompts=DEFAULT_PROMPTS if not args.prompts_file else [x for x in Path(args.prompts_file).read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.limit_prompts: prompts=prompts[:args.limit_prompts]
    if not prompts: raise SystemExit("no prompts")
    if args.max_new_tokens<1: raise SystemExit("--max-new-tokens must be >= 1")
    manifest={"format":"atlas-qwen38-trace-manifest-v1","profile":profile,"model":str(model.resolve()),
              "model_fingerprint":fingerprint(model),"binary":str(binary.resolve()),"requested_prompts":len(prompts),
              "max_new_tokens":args.max_new_tokens,"ngl":args.ngl,"ctx_size":args.ctx_size,"sessions":[]}
    tmp=out.with_suffix(out.suffix+".tmp")
    if tmp.exists(): tmp.unlink()
    try:
        with tmp.open("w",encoding="utf-8") as f:
            for i,prompt in enumerate(prompts):
                print(f"[{i+1}/{len(prompts)}] tracing {prompt[:72]!r}",flush=True)
                t=time.time()
                stdout,stderr=run(binary,model,prompt,args.max_new_tokens,args.ngl,args.ctx_size,args.extra_arg)
                calls,done=parse(stdout,profile)
                records=validate_and_build(calls,done,profile)
                entry={"prompt_idx":i,"prompt":prompt,"num_layers":profile["block_count"],
                       "top_k":profile["expert_used_count"],"tokens_per_second":done["tok_per_sec"],
                       "prompt_token_count":done["prompt_tokens"],"decode_token_count":done["n_decoded"],
                       "trace_format_version":3,"model_profile":profile,"records":records}
                f.write(json.dumps(entry,ensure_ascii=False,separators=(",",":"))+"\n"); f.flush(); os.fsync(f.fileno())
                manifest["sessions"].append({"prompt_idx":i,"records":len(records),**done,"wall_seconds":time.time()-t,
                                             "stderr_tail":stderr.splitlines()[-5:]})
                print(f"    OK: {len(records)} records, {done['prompt_tokens']} prompt + {done['n_decoded']} decode")
        # Final whole-file sanity before publish.
        lines=tmp.read_text(encoding="utf-8").splitlines()
        if len(lines)!=len(prompts): raise TraceError("final file session count mismatch")
        os.replace(tmp,out)
    except Exception:
        if tmp.exists(): tmp.unlink()
        raise
    manifest["validation"]="PASS"
    manifest["output"]=str(out.resolve())
    mp=Path(args.manifest) if args.manifest else out.with_suffix(out.suffix+".manifest.json")
    mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"\nPASS — atomic trace written: {out}")
    print(f"Manifest: {mp}")
if __name__=="__main__":
    main()
