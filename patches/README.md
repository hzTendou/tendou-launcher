# llama.cpp integration patch

The native runtime depends on a local llama.cpp fork. The checkout itself is not
vendored in this repository. `llama.cpp.patch` contains the Tendou-specific delta
against upstream commit `67a17c17caa95742186f8b1ecadd1b5abd6d5ebb`.

Apply it to a clean checkout:

```powershell
git clone https://github.com/ggml-org/llama.cpp.git llama.cpp
git -C llama.cpp checkout 67a17c17caa95742186f8b1ecadd1b5abd6d5ebb
git -C llama.cpp apply ..\patches\llama.cpp.patch
```

The patch is generated from the active local fork, including its current working
tree changes. Regenerate it only after reviewing the nested checkout and rerunning
the native and Python test suites.
