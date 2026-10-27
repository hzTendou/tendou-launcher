with open('src/atlas/atlas-engine.cpp', 'r', encoding='utf-8') as f:
    content = f.read()

target1 = """    // Ensure optimal CPU MoE prefill threshold is active if not set by user environment
    if (!std::getenv("GGML_OP_OFFLOAD_MIN_BATCH")) {
#if defined(_WIN32)
        _putenv_s("GGML_OP_OFFLOAD_MIN_BATCH", "2048");
#else
        setenv("GGML_OP_OFFLOAD_MIN_BATCH", "2048", 0);
#endif
    }"""

replacement1 = """    // Ensure optimal CPU MoE prefill threshold is active if not set by user environment
    if (!std::getenv("GGML_OP_OFFLOAD_MIN_BATCH")) {
#if defined(_WIN32)
        _putenv_s("GGML_OP_OFFLOAD_MIN_BATCH", "2048");
#else
        setenv("GGML_OP_OFFLOAD_MIN_BATCH", "2048", 0);
#endif
    }

    // Ensure optimal CPU MoE prefill expert pruning cap is active (Top-4) to eliminate disk page thrashing
    if (!std::getenv("ATLAS_PREFILL_TOP_K")) {
#if defined(_WIN32)
        _putenv_s("ATLAS_PREFILL_TOP_K", "4");
#else
        setenv("ATLAS_PREFILL_TOP_K", "4", 0);
#endif
    }"""

assert target1 in content, "Target 1 not found"
content = content.replace(target1, replacement1, 1)

target2 = """        if (params.n_batch == 0 || params.n_batch > 256) {
            params.n_batch = 256;
        }"""

replacement2 = """        const char * chunk_sz_env = std::getenv("ATLAS_PREFILL_CHUNK_SIZE");
        if (chunk_sz_env && chunk_sz_env[0] != '\\0') {
            int sz = std::atoi(chunk_sz_env);
            if (sz >= 16 && sz <= 2048) {
                params.n_batch = sz;
            }
        } else if (params.n_batch == 0 || params.n_batch > 256) {
            params.n_batch = 256;
        }"""

assert target2 in content, "Target 2 not found"
content = content.replace(target2, replacement2, 1)

with open('src/atlas/atlas-engine.cpp', 'w', encoding='utf-8') as f:
    f.write(content)
print("Successfully patched src/atlas/atlas-engine.cpp")
