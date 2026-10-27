from pathlib import Path

# 1. Patch src/atlas/atlas-engine.cpp
cpp_path = Path("src/atlas/atlas-engine.cpp")
cpp = cpp_path.read_text(encoding="utf-8")

# Fast blob_path caching in parse_json_map
old_bp_block = (
    '                if (line.find("\\"blob_path\\"") != std::string::npos) {\n'
    '                    std::string bp = json_str_val(line, "blob_path");\n'
    '                    cur_chunk.blob_path.clear();\n'
    '                    for (size_t i = 0; i < bp.size(); ++i) {\n'
    '                        if (bp[i] == \'\\\\\' && i + 1 < bp.size() && bp[i+1] == \'\\\\\') {\n'
    '                            cur_chunk.blob_path += \'\\\\\';\n'
    '                            ++i;\n'
    '                        } else {\n'
    '                            cur_chunk.blob_path += bp[i];\n'
    '                        }\n'
    '                    }\n'
    '                    cur_chunk.blob_idx = get_blob_idx(cur_chunk.blob_path);'
)

new_bp_block = (
    '                if (line.find("\\"blob_path\\"") != std::string::npos) {\n'
    '                    static std::string s_last_raw_bp;\n'
    '                    static std::string s_last_clean_bp;\n'
    '                    static int s_last_blob_idx = -1;\n'
    '                    std::string bp = json_str_val(line, "blob_path");\n'
    '                    if (bp == s_last_raw_bp && !s_last_clean_bp.empty()) {\n'
    '                        cur_chunk.blob_path = s_last_clean_bp;\n'
    '                        cur_chunk.blob_idx = s_last_blob_idx;\n'
    '                    } else {\n'
    '                        cur_chunk.blob_path.clear();\n'
    '                        for (size_t i = 0; i < bp.size(); ++i) {\n'
    '                            if (bp[i] == \'\\\\\' && i + 1 < bp.size() && bp[i+1] == \'\\\\\') {\n'
    '                                cur_chunk.blob_path += \'\\\\\';\n'
    '                                ++i;\n'
    '                            } else {\n'
    '                                cur_chunk.blob_path += bp[i];\n'
    '                            }\n'
    '                        }\n'
    '                        cur_chunk.blob_idx = get_blob_idx(cur_chunk.blob_path);\n'
    '                        s_last_raw_bp = bp;\n'
    '                        s_last_clean_bp = cur_chunk.blob_path;\n'
    '                        s_last_blob_idx = cur_chunk.blob_idx;\n'
    '                    }'
)

assert cpp.count(old_bp_block) == 1, f"old_bp_block count: {cpp.count(old_bp_block)}"
cpp = cpp.replace(old_bp_block, new_bp_block, 1)

# Correct vram_budget and ram_budget calculation to avoid laptop OOM
old_budget_block = (
    "    // Calculate Slab Arena pool capacities\n"
    "    size_t vram_slots = cfg.vram_slab_slots;\n"
    "    if (vram_slots == 0) {\n"
    "        size_t vram_budget = cfg.gpu_first ?\n"
    "            std::max(vram_capacity_bytes, size_t(cfg.vram_cache_mb * 1024 * 1024ULL)) :\n"
    "            vram_capacity_bytes;\n"
    "        vram_slots = vram_budget / expert_slice_bytes;\n"
    "        if (vram_slots == 0) vram_slots = 588;\n"
    "    }\n"
    "\n"
    "    size_t ram_slots = cfg.ram_slab_slots;\n"
    "    if (ram_slots == 0) {\n"
    "        size_t ram_budget = ram_capacity_bytes;\n"
    "        if (cfg.moe_cache_mib > 0) {\n"
    "            ram_budget = std::max(ram_budget, size_t(cfg.moe_cache_mib * 1024 * 1024ULL));\n"
    "        }\n"
    "        ram_slots = ram_budget / expert_slice_bytes;\n"
    "        if (ram_slots == 0) ram_slots = 2048;\n"
    "    }"
)

new_budget_block = (
    "    // Calculate Slab Arena pool capacities\n"
    "    size_t vram_slots = cfg.vram_slab_slots;\n"
    "    if (vram_slots == 0) {\n"
    "        size_t vram_budget = (cfg.vram_cache_mb > 0) ?\n"
    "            size_t(cfg.vram_cache_mb * 1024 * 1024ULL) :\n"
    "            vram_capacity_bytes;\n"
    "        vram_slots = vram_budget / expert_slice_bytes;\n"
    "        if (vram_slots == 0) vram_slots = 588;\n"
    "    }\n"
    "    vram_capacity_bytes = vram_slots * expert_slice_bytes;\n"
    "\n"
    "    size_t ram_slots = cfg.ram_slab_slots;\n"
    "    if (ram_slots == 0) {\n"
    "        size_t ram_budget = (cfg.moe_cache_mib > 0) ?\n"
    "            size_t(cfg.moe_cache_mib * 1024 * 1024ULL) :\n"
    "            ram_capacity_bytes;\n"
    "        ram_slots = ram_budget / expert_slice_bytes;\n"
    "        if (ram_slots == 0) ram_slots = 1024;\n"
    "    }\n"
    "    ram_capacity_bytes = ram_slots * expert_slice_bytes;"
)

assert cpp.count(old_budget_block) == 1, f"old_budget_block count: {cpp.count(old_budget_block)}"
cpp = cpp.replace(old_budget_block, new_budget_block, 1)

cpp_path.write_text(cpp, encoding="utf-8")
print("Successfully patched src/atlas/atlas-engine.cpp")

# 2. Patch scripts/run_agentic_protocols_benchmark.py
bm_path = Path("scripts/run_agentic_protocols_benchmark.py")
bm = bm_path.read_text(encoding="utf-8")

old_preflight = (
    "    # Preflight check\n"
    "    started = False\n"
    "    for _ in range(60):\n"
    "        try:\n"
    '            r = httpx.get(f"{base_url}/health", timeout=0.5)\n'
    "            if r.status_code == 200:\n"
    "                started = True\n"
    "                break\n"
    "        except Exception:\n"
    "            time.sleep(0.1)\n"
    "\n"
    "    if not started:\n"
    '        raise RuntimeError(f"Server failed to start (exe={exe_path.name}, mtp={mtp_mode})")'
)

new_preflight = (
    "    # Preflight check (support up to 60s for cold 75GB multi-shard GGUF & physical map init)\n"
    "    started = False\n"
    "    for _ in range(300):\n"
    "        try:\n"
    '            r = httpx.get(f"{base_url}/health", timeout=0.5)\n'
    "            if r.status_code == 200:\n"
    "                started = True\n"
    "                break\n"
    "        except Exception:\n"
    "            time.sleep(0.2)\n"
    "\n"
    "    if not started:\n"
    '        raise RuntimeError(f"Server failed to start (exe={exe_path.name}, mtp={mtp_mode})")'
)

assert bm.count(old_preflight) == 1, f"old_preflight count: {bm.count(old_preflight)}"
bm = bm.replace(old_preflight, new_preflight, 1)

old_stop = (
    "def stop_server_instance(backend: AtlasSubprocessBackend, server: uvicorn.Server) -> None:\n"
    "    try:\n"
    "        import asyncio\n"
    "        asyncio.run(backend.stop())\n"
    "    except Exception:\n"
    "        pass\n"
    "    try:\n"
    "        server.should_exit = True\n"
    "    except Exception:\n"
    "        pass"
)

new_stop = (
    "def stop_server_instance(backend: AtlasSubprocessBackend, server: uvicorn.Server) -> None:\n"
    "    try:\n"
    "        import asyncio\n"
    "        asyncio.run(backend.stop())\n"
    "    except Exception:\n"
    "        pass\n"
    "    try:\n"
    "        server.should_exit = True\n"
    "    except Exception:\n"
    "        pass\n"
    "    time.sleep(2.0)  # Allow Windows WDDM driver to reclaim GPU VRAM cleanly"
)

assert bm.count(old_stop) == 1, f"old_stop count: {bm.count(old_stop)}"
bm = bm.replace(old_stop, new_stop, 1)

bm_path.write_text(bm, encoding="utf-8")
print("Successfully patched scripts/run_agentic_protocols_benchmark.py")
