# scratch/patch_moe_topology.py

with open('src/atlas/atlas-engine.cpp', 'rb') as f:
    cpp_content = f.read().decode('utf-8')

old_cpp_parse_target = """                if (std::sscanf(key.c_str(), "%d:%d", &l, &e) == 2) {
                    if (in_chunk && !cur_chunk.kind.empty()) {
                        cur_exp.chunks.push_back(cur_chunk);
                        cur_chunk = ChunkLocation{};
                        in_chunk = false;
                    }
                    if (in_expert) {
                        out_map[{cur_exp.layer, cur_exp.expert}] = cur_exp;
                    }
                    cur_exp = PhysicalExpert{};
                    cur_exp.layer = l;
                    cur_exp.expert = e;
                    cur_exp.total_bytes = 3481600;
                    in_expert = true;
                    continue;
                }"""

new_cpp_parse_replacement = """                if (std::sscanf(key.c_str(), "%d:%d", &l, &e) == 2) {
                    if (in_chunk && !cur_chunk.kind.empty()) {
                        cur_exp.chunks.push_back(cur_chunk);
                        cur_chunk = ChunkLocation{};
                        in_chunk = false;
                    }
                    if (in_expert) {
                        if (!cur_exp.chunks.empty()) {
                            size_t sum_bytes = 0;
                            for (const auto & c : cur_exp.chunks) sum_bytes += c.size_bytes;
                            if (sum_bytes > 0 && (cur_exp.total_bytes == 0 || cur_exp.total_bytes == 3481600)) {
                                cur_exp.total_bytes = sum_bytes;
                            }
                        }
                        out_map[{cur_exp.layer, cur_exp.expert}] = cur_exp;
                    }
                    cur_exp = PhysicalExpert{};
                    cur_exp.layer = l;
                    cur_exp.expert = e;
                    cur_exp.total_bytes = 0;
                    in_expert = true;
                    continue;
                }"""

old_cpp_chunk_target = r"""        if (in_expert) {
            if (line.find("\"kind\"") != std::string::npos) {"""

new_cpp_chunk_replacement = r"""        if (in_expert) {
            if (line.find("\"total_bytes\"") != std::string::npos) {
                cur_exp.total_bytes = (size_t)json_uint_val(line, "total_bytes");
            } else if (line.find("\"kind\"") != std::string::npos) {"""

old_cpp_eof_target = """    if (in_chunk && !cur_chunk.kind.empty()) {
        cur_exp.chunks.push_back(cur_chunk);
    }
    if (in_expert) {
        out_map[{cur_exp.layer, cur_exp.expert}] = cur_exp;
    }"""

new_cpp_eof_replacement = """    if (in_chunk && !cur_chunk.kind.empty()) {
        cur_exp.chunks.push_back(cur_chunk);
    }
    if (in_expert) {
        if (!cur_exp.chunks.empty()) {
            size_t sum_bytes = 0;
            for (const auto & c : cur_exp.chunks) sum_bytes += c.size_bytes;
            if (sum_bytes > 0 && (cur_exp.total_bytes == 0 || cur_exp.total_bytes == 3481600)) {
                cur_exp.total_bytes = sum_bytes;
            }
        }
        out_map[{cur_exp.layer, cur_exp.expert}] = cur_exp;
    }"""

old_cpp_clamp_target = """        for (const auto & [k, pe] : physical_map) {
            size_t idx = size_t(k.layer) * 512 + size_t(k.expert);
            if (idx < TOTAL_EXPERTS) {
                PhysicalExpert pe_copy = pe;
                if (pe_copy.total_bytes > 1253376) {
                    pe_copy.total_bytes = 1253376;
                }
                expert_flat[idx] = pe_copy;
                residency_flat[idx] = MemoryTier::NVME;
            }
        }"""

new_cpp_clamp_replacement = """        for (const auto & [k, pe] : physical_map) {
            size_t idx = size_t(k.layer) * 512 + size_t(k.expert);
            if (idx < TOTAL_EXPERTS) {
                PhysicalExpert pe_copy = pe;
                // Preserve exact expert byte size for larger models (e.g. Qwen3.6-35B at 1,458,176 B / 1,761,280 B)
                // while maintaining the 1,253,376 B ceiling for default Qwen3.8 Flash Next IQ3_XXS
                if (pe_copy.total_bytes > 1253376 && pe_copy.total_bytes < 1400000) {
                    pe_copy.total_bytes = 1253376;
                }
                expert_flat[idx] = pe_copy;
                residency_flat[idx] = MemoryTier::NVME;
            }
        }"""

for old_s, new_s in [
    (old_cpp_parse_target, new_cpp_parse_replacement),
    (old_cpp_chunk_target, new_cpp_chunk_replacement),
    (old_cpp_eof_target, new_cpp_eof_replacement),
    (old_cpp_clamp_target, new_cpp_clamp_replacement),
]:
    assert old_s.replace('\r\n', '\n') in cpp_content.replace('\r\n', '\n'), f"Target not found: {old_s[:40]}"
    cpp_content_new = cpp_content.replace(old_s.replace('\n', '\r\n'), new_s.replace('\n', '\r\n'))
    if cpp_content_new == cpp_content:
        cpp_content_new = cpp_content.replace(old_s, new_s)
    cpp_content = cpp_content_new

with open('src/atlas/atlas-engine.cpp', 'wb') as f:
    f.write(cpp_content.encode('utf-8'))
print("Updated src/atlas/atlas-engine.cpp successfully.")
