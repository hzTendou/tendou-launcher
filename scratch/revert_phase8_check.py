import json
import os

p = r"C:\Users\Ali\.gemini\antigravity\brain\2c189719-2ec4-4fe2-8e96-724a59a4fa40\.system_generated\logs\transcript.jsonl"
edits = []
with open(p, "r", encoding="utf-8") as f:
    for idx, l in enumerate(f):
        if idx < 150:
            continue
        d = json.loads(l)
        for c in d.get("tool_calls", []):
            if c.get("name") == "replace_file_content":
                args = c.get("args", {})
                tf = args.get("TargetFile", "").strip()
                if tf.startswith('"') and tf.endswith('"'):
                    tf = tf[1:-1]
                if "AGENT.md" in tf or "docs/agent-history.md" in tf or "test_async_prefetch.cpp" in tf:
                    continue
                t = args.get("TargetContent", "")
                r = args.get("ReplacementContent", "")
                if t.startswith('"') and t.endswith('"'):
                    t = t[1:-1]
                if r.startswith('"') and r.endswith('"'):
                    r = r[1:-1]
                edits.append((idx, tf, t, r))

print(f"Collected {len(edits)} code edits.")
reversed_edits = list(reversed(edits))

files_content = {}
for idx, tf, target, repl in reversed_edits:
    if tf not in files_content:
        with open(tf, "r", encoding="utf-8") as f:
            files_content[tf] = f.read().replace("\r\n", "\n")
    
    r_norm = repl.replace("\r\n", "\n")
    t_norm = target.replace("\r\n", "\n")
    
    if r_norm in files_content[tf]:
        files_content[tf] = files_content[tf].replace(r_norm, t_norm, 1)
        print(f"OK: Reverted Step {idx} in {os.path.basename(tf)}")
    else:
        print(f"MISMATCH in Step {idx} in {os.path.basename(tf)}")

os.makedirs("src_phase8_backup", exist_ok=True)
for tf, content in files_content.items():
    dst = os.path.join("src_phase8_backup", os.path.basename(tf) + (".backend" if "backend" in tf else ""))
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    print(f"Saved Phase 8 file: {dst}")
