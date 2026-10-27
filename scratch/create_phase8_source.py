import json
import os

p_full = r"C:\Users\Ali\.gemini\antigravity\brain\2c189719-2ec4-4fe2-8e96-724a59a4fa40\.system_generated\logs\transcript_full.jsonl"
edits = []

def parse_val(v):
    if isinstance(v, str):
        if v.startswith('"') and v.endswith('"'):
            try:
                return json.loads(v, strict=False)
            except Exception:
                pass
    return v

with open(p_full, "r", encoding="utf-8") as f:
    for idx, l in enumerate(f):
        if idx < 150:
            continue
        d = json.loads(l)
        for c in d.get("tool_calls", []):
            if c.get("name") == "replace_file_content":
                args = c.get("args", {})
                tf = parse_val(args.get("TargetFile", ""))
                if "AGENT.md" in tf or "docs/agent-history.md" in tf or "test_async_prefetch.cpp" in tf:
                    continue
                t = parse_val(args.get("TargetContent", ""))
                r = parse_val(args.get("ReplacementContent", ""))
                edits.append((idx, tf, t, r))

print(f"Collected {len(edits)} code edits from transcript_full.jsonl.")
reversed_edits = list(reversed(edits))

files_content = {}
for idx, tf, target, repl in reversed_edits:
    if tf not in files_content:
        with open(tf, "r", encoding="utf-8") as f:
            files_content[tf] = f.read()
    
    if repl in files_content[tf]:
        files_content[tf] = files_content[tf].replace(repl, target, 1)
        print(f"SUCCESS: Reverted Step {idx} in {os.path.basename(tf)}")
    else:
        print(f"FAILED: Step {idx} in {os.path.basename(tf)}")

os.makedirs("src_phase8", exist_ok=True)
for tf, content in files_content.items():
    rel = os.path.relpath(tf, os.getcwd())
    dst = os.path.join("src_phase8", rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    print(f"Wrote Phase 8 file: {dst}")

print("All Phase 8 source files successfully generated in src_phase8/!")
