import re
from pathlib import Path

cpp_path = Path("src/atlas/atlas-engine.cpp")
content = cpp_path.read_text(encoding="utf-8")

# 1. Parse initial_tier and pinned in parse_json_map
s1_old = 'cur_exp.total_bytes = (size_t)json_uint_val(line, "total_bytes");'
s1_new = (
    'cur_exp.total_bytes = (size_t)json_uint_val(line, "total_bytes");\n'
    '            } else if (line.find("\\"tier\\"") != std::string::npos) {\n'
    '                cur_exp.initial_tier = json_str_val(line, "tier");\n'
    '            } else if (line.find("\\"pinned\\"") != std::string::npos) {\n'
    '                cur_exp.pinned = (line.find("true") != std::string::npos);'
)
assert content.count(s1_old) == 1, f"s1_old matches {content.count(s1_old)} times"
content = content.replace(s1_old, s1_new, 1)

# 2. Expert slice calibration: maximum expert size across all experts
s2_old = (
    "    // Calibrate expert slice bytes\n"
    "    if (!physical_map.empty()) {\n"
    "        for (const auto & [_, pe] : physical_map) {\n"
    "            if (pe.total_bytes > 0) {\n"
    "                expert_slice_bytes = pe.total_bytes;\n"
    "                break;\n"
    "            }\n"
    "        }\n"
    "    }"
)
s2_new = (
    "    // Calibrate expert slice bytes to maximum expert size across all experts\n"
    "    if (!physical_map.empty()) {\n"
    "        for (const auto & [_, pe] : physical_map) {\n"
    "            if (pe.total_bytes > expert_slice_bytes) {\n"
    "                expert_slice_bytes = pe.total_bytes;\n"
    "            }\n"
    "        }\n"
    "    }"
)
assert content.count(s2_old) == 1, f"s2_old matches {content.count(s2_old)} times"
content = content.replace(s2_old, s2_new, 1)

# 3. Add initial 3-tier placement loader at end of ExpertMemoryManager constructor
s3_anchor = "        metrics.resident_vram_experts = get_vram_resident_count();\n    }\n}\n\nExpertMemoryManager::~ExpertMemoryManager()"
s3_inject = (
    "        metrics.resident_vram_experts = get_vram_resident_count();\n    }\n\n"
    "    // Initialize 3-tier working set layout from physical map if specified\n"
    "    if (!physical_map.empty()) {\n"
    "        size_t initial_vram_count = 0;\n"
    "        size_t initial_ram_count = 0;\n"
    "        for (const auto & [k, pe] : physical_map) {\n"
    "            size_t idx = size_t(k.layer) * 512 + size_t(k.expert);\n"
    "            if (idx >= TOTAL_EXPERTS) continue;\n"
    "            if (k.layer < cfg.gpu_expert_layers) continue; // Already GPU anchor\n\n"
    "            if (!pe.initial_tier.empty()) {\n"
    '                if ((pe.initial_tier.find("tier0") != std::string::npos || pe.initial_tier.find("vram") != std::string::npos) && vram_slab_pool.free_slots() > 0) {\n'
    "                    int slot = vram_slab_pool.acquire_slot();\n"
    "                    if (slot >= 0) {\n"
    "                        vram_slot_flat[idx] = slot;\n"
    "                        residency_flat[idx] = MemoryTier::VRAM;\n"
    "                        residency[k] = MemoryTier::VRAM;\n"
    "                        vram_lru.push_front(k);\n"
    "                        vram_iter_flat[idx] = vram_lru.begin();\n"
    "                        vram_present_flat[idx] = 1;\n"
    "                        vram_used_bytes += expert_slice_bytes;\n"
    "                        atlas_set_expert_gpu_slot(k.layer, k.expert, (slot < 21) ? (int16_t)slot : (int16_t)(slot % 21));\n"
    "                        if (pe.pinned) pinned_flat[idx] = 1;\n"
    "                        initial_vram_count++;\n"
    "                    }\n"
    '                } else if ((pe.initial_tier.find("tier1") != std::string::npos || pe.initial_tier.find("ram") != std::string::npos) && ram_slab_pool.free_slots() > 0) {\n'
    "                    int slot = ram_slab_pool.acquire_slot();\n"
    "                    if (slot >= 0) {\n"
    "                        ram_slot_flat[idx] = slot;\n"
    "                        residency_flat[idx] = MemoryTier::RAM;\n"
    "                        residency[k] = MemoryTier::RAM;\n"
    "                        ram_lru.push_front(k);\n"
    "                        ram_iter_flat[idx] = ram_lru.begin();\n"
    "                        ram_present_flat[idx] = 1;\n"
    "                        ram_used_bytes += expert_slice_bytes;\n"
    "                        if (pe.pinned) pinned_flat[idx] = 1;\n"
    "                        initial_ram_count++;\n"
    "                    }\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "        if (initial_vram_count > 0 || initial_ram_count > 0) {\n"
    '            std::printf("[ATLAS] Initialized 3-tier layout: %zu VRAM experts, %zu RAM experts\\n",\n'
    "                initial_vram_count, initial_ram_count);\n"
    "            metrics.resident_vram_experts = get_vram_resident_count();\n"
    "        }\n"
    "    }\n"
    "}\n\nExpertMemoryManager::~ExpertMemoryManager()"
)
assert content.count(s3_anchor) == 1, f"s3_anchor matches {content.count(s3_anchor)} times"
content = content.replace(s3_anchor, s3_inject, 1)

# 4. Remove obsolete clamp in load_physical_map
s4_old = (
    "                PhysicalExpert pe_copy = pe;\n"
    "                // Preserve exact expert byte size for larger models (e.g. Qwen3.6-35B at 1,458,176 B / 1,761,280 B)\n"
    "                // while maintaining the 1,253,376 B ceiling for default Qwen3.8 Flash Next IQ3_XXS\n"
    "                if (pe_copy.total_bytes > 1253376 && pe_copy.total_bytes < 1400000) {\n"
    "                    pe_copy.total_bytes = 1253376;\n"
    "                }\n"
    "                expert_flat[idx] = pe_copy;"
)
s4_new = (
    "                PhysicalExpert pe_copy = pe;\n"
    "                expert_flat[idx] = pe_copy;"
)
assert content.count(s4_old) == 1, f"s4_old matches {content.count(s4_old)} times"
content = content.replace(s4_old, s4_new, 1)

cpp_path.write_text(content, encoding="utf-8")
print("Successfully patched src/atlas/atlas-engine.cpp with 4 surgical updates.")
