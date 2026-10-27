# scratch/patch_tening_runner.py

p = r"C:\Users\Ali\WebProjects\tening\src\tening\tendou_runner.py"
with open(p, 'rb') as f:
    text = f.read().decode('utf-8')

old_engine_def = """DEFAULT_ENGINE_PATH = (
    DEFAULT_TENDOU_ROOT / "releases" / "tendou-v0.11.0-stable" / "bin" / "tendou-engine.exe"
)"""

new_engine_def = """DEFAULT_ENGINE_PATH = (
    DEFAULT_TENDOU_ROOT / "releases" / "tendou-v0.13.0-stable" / "bin" / "tendou-engine.exe"
)"""

old_dll_dirs = """REQUIRED_DLL_DIRS = [
    DEFAULT_TENDOU_ROOT / "releases" / "tendou-v0.11.0-stable" / "bin",
    DEFAULT_TENDOU_ROOT / "llama.cpp" / "build" / "bin" / "Release",
    DEFAULT_TENDOU_ROOT / ".venv-audit" / "Lib" / "site-packages" / "nvidia" / "cu13" / "bin" / "x86_64",
]"""

new_dll_dirs = """REQUIRED_DLL_DIRS = [
    DEFAULT_TENDOU_ROOT / "releases" / "tendou-v0.13.0-stable" / "bin",
    DEFAULT_TENDOU_ROOT / "releases" / "tendou-v0.11.0-stable" / "bin",
    DEFAULT_TENDOU_ROOT / "llama.cpp" / "build" / "bin" / "Release",
    DEFAULT_TENDOU_ROOT / ".venv-audit" / "Lib" / "site-packages" / "nvidia" / "cu13" / "bin" / "x86_64",
]"""

assert old_engine_def.replace('\r\n', '\n') in text.replace('\r\n', '\n')
assert old_dll_dirs.replace('\r\n', '\n') in text.replace('\r\n', '\n')

text = text.replace(old_engine_def.replace('\n', '\r\n'), new_engine_def.replace('\n', '\r\n'))
text = text.replace(old_dll_dirs.replace('\n', '\r\n'), new_dll_dirs.replace('\n', '\r\n'))

with open(p, 'wb') as f:
    f.write(text.encode('utf-8'))
print("Successfully updated tening tendou_runner.py to point to v0.13.0-stable")
