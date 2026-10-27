import re

with open("src/atlas/atlas-engine.cpp", "r") as f:
    content = f.read()

# Remove old patch if it exists
if "const char* deep_telem" in content:
    content = re.sub(r'const char\* deep_telem = std::getenv\("ATLAS_DEEP_TELEMETRY"\);.*?tracer\.clear\(\);\n    }', '', content, flags=re.DOTALL)

patch = """
    if (prof.timelines.size() < 128) {
        prof.timelines.push_back(current_timeline);
    }

    const char* deep_telem = std::getenv("ATLAS_DEEP_TELEMETRY");
    if (deep_telem && std::string(deep_telem) == "1") {
        auto& tracer = atlas::GlobalTracer::get();
        size_t n = tracer.size();
        const auto* events = tracer.data();
        
        FILE* tf = std::fopen("trace.jsonl", "a");
        if (tf) {
            std::vector<int64_t> layer_router(48, 0);
            std::vector<int64_t> layer_cpu(48, 0);
            std::vector<int64_t> layer_gpu(48, 0);
            std::vector<int64_t> layer_h2d(48, 0);
            
            for (size_t i = 0; i < n; ++i) {
                const auto& e = events[i];
                if (e.layer_id >= 0 && e.layer_id < 48) {
                    if (e.type == atlas::TraceEventType::ROUTER) layer_router[e.layer_id] += e.duration_us;
                    if (e.type == atlas::TraceEventType::CPU_MOE) layer_cpu[e.layer_id] += e.duration_us;
                    if (e.type == atlas::TraceEventType::GPU_DENSE) layer_gpu[e.layer_id] += e.duration_us;
                    if (e.type == atlas::TraceEventType::H2D) layer_h2d[e.layer_id] += e.duration_us;
                }
            }
            
            const auto& metrics = memory_manager.get_metrics();
            for (int l = 0; l < 48; ++l) {
                if (layer_router[l] == 0 && layer_cpu[l] == 0 && layer_gpu[l] == 0) continue;
                
                nlohmann::json trace = {
                    {"event", "layer_trace"},
                    {"token", token_idx},
                    {"layer", l},
                    {"router_us", layer_router[l]},
                    {"expert_acquire_us", 0},
                    {"expert_wait_us", 0},
                    {"cpu_moe_us", layer_cpu[l]},
                    {"h2d_us", layer_h2d[l]},
                    {"gpu_dense_us", layer_gpu[l]},
                    {"cpu_gpu_sync_us", 0},
                    {"layer_total_us", layer_router[l] + layer_cpu[l] + layer_gpu[l]},
                    {"expert_page_bytes", metrics.expert_page_bytes / 48},
                    {"dram_read_bytes", nullptr},
                    {"cpu_gpu_overlap_us", 0}
                };
                std::fprintf(tf, "%s\\n", trace.dump().c_str());
            }
            
            nlohmann::json t_trace = {
                {"event", "token_trace"},
                {"token", token_idx},
                {"token_total_us", (int64_t)(total_decode_ms * 1000.0)},
                {"expert_page_bytes", metrics.expert_page_bytes},
                {"dram_read_bytes", nullptr},
                {"dram_counter_available", false},
                {"h2d_bytes", 0}
            };
            std::fprintf(tf, "%s\\n", t_trace.dump().c_str());
            std::fclose(tf);
        }
        tracer.clear();
    }
"""

content = content.replace("""    if (prof.timelines.size() < 128) {
        prof.timelines.push_back(current_timeline);
    }""", patch)

with open("src/atlas/atlas-engine.cpp", "w") as f:
    f.write(content)
