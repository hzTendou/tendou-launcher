import sys

with open('AGENT.md', 'r', encoding='utf-8') as f:
    agent_text = f.read()

# Phase 10 entry in AGENT.md
lines = agent_text.splitlines()

# Let's find Phase 10 start and end
phase10_lines = []
in_phase10 = False
other_lines = []

for line in lines:
    if line.startswith('2026-10-20 | Antigravity / Phase 10:'):
        in_phase10 = True
    if in_phase10:
        phase10_lines.append(line)
    else:
        other_lines.append(line)

print(f"Extracted {len(phase10_lines)} lines for Phase 10")
phase10_text = "\n".join(phase10_lines).strip() + "\n\n"

with open('docs/agent-history.md', 'r', encoding='utf-8') as f:
    history = f.read()

header = '# tendou launcher — geçmiş ajan devir kayıtları\n\n'
if history.startswith(header):
    rest = history[len(header):]
else:
    rest = history

new_history = header + phase10_text + rest
with open('docs/agent-history.md', 'w', encoding='utf-8') as f:
    f.write(new_history)

print("docs/agent-history.md updated.")
