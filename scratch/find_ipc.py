with open('src/atlas/atlas-engine.cpp', 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read()

lines = text.splitlines()
for i, line in enumerate(lines):
    if 'generate' in line and ('cmd' in line or 'handle' in line or 'ipc' in line):
        print(f'{i+1}: {line[:100]}')
