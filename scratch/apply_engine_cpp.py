#!/usr/bin/env python3
"""Script to update atlas-engine.cpp with OD-MoE, SPICE, and Tutti implementations."""
from pathlib import Path

cpp_path = Path("src/atlas/atlas-engine.cpp")
text = cpp_path.read_text(encoding="utf-8")

# 1. Insert TuttiAsyncPipeline implementation right after PagePrefetcher
tutti_impl = '''
// ---------------------------------------------------------------------------
// Tutti: Async NVMe Pipeline (Decoupled Disk I/O, Pinned Staging & Async H2D)
// ---------------------------------------------------------------------------
TuttiAsyncPipeline::TuttiAsyncPipeline(const Config & cfg, PinnedHostStagingPool & staging_pool)
    : cfg_(cfg), staging_pool_(staging_pool)
{
    init_mappings();
    worker_ = std::thread(&TuttiAsyncPipeline::worker_fn, this);
}

TuttiAsyncPipeline::~TuttiAsyncPipeline() {
    {
        std::unique_lock<std::mutex> lk(q_mtx_);
        running_ = false;
    }
    q_cv_.notify_one();
    if (worker_.joinable()) worker_.join();
    close_all_handles();
}

void TuttiAsyncPipeline::init_mappings() {
    static const char * BLOB_PATHS[3] = {
        "C:\\\\Users\\\\Ali\\\\.cache\\\\huggingface\\\\hub\\\\models--orcarouter--Qwen3.8-Flash-Next-Uncensored-GGUF\\\\blobs\\\\98c001112ee9d661fa21b7c2162e24dbfabb6a08af14e4cabfaa73c351ce365b",
        "C:\\\\Users\\\\Ali\\\\.cache\\\\huggingface\\\\hub\\\\models--orcarouter--Qwen3.8-Flash-Next-Uncensored-GGUF\\\\blobs\\\\d43111ec60a5868cdb60bdcffed89454ac8d51b911d5a847f7d2c2f6f20c6b7a",
        "C:\\\\Users\\\\Ali\\\\.cache\\\\huggingface\\\\hub\\\\models--orcarouter--Qwen3.8-Flash-Next-Uncensored-GGUF\\\\blobs\\\\1cf59e7ae710a43b5dfc3fe03f16fa74ae752f18ea1c555dc87a1c2678401366"
    };
    for (int i = 0; i < 3; ++i) {
#if defined(_WIN32)
        blobs_[i].hFile = CreateFileA(BLOB_PATHS[i], GENERIC_READ, FILE_SHARE_READ, nullptr,
                                      OPEN_EXISTING, FILE_FLAG_RANDOM_ACCESS, nullptr);
        if (blobs_[i].hFile != INVALID_HANDLE_VALUE) {
            LARGE_INTEGER sz{};
            if (GetFileSizeEx((HANDLE)blobs_[i].hFile, &sz)) {
                blobs_[i].file_size = (size_t)sz.QuadPart;
                blobs_[i].hMap = CreateFileMappingA((HANDLE)blobs_[i].hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
                if (blobs_[i].hMap) {
                    blobs_[i].base = MapViewOfFile((HANDLE)blobs_[i].hMap, FILE_MAP_READ, 0, 0, 0);
                }
            }
        }
#else
        int fd = open(BLOB_PATHS[i], O_RDONLY);
        if (fd >= 0) {
            struct stat st;
            if (fstat(fd, &st) == 0) {
                blobs_[i].file_size = st.st_size;
                blobs_[i].base = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
            }
            close(fd);
        }
#endif
    }
}

void TuttiAsyncPipeline::close_all_handles() {
    for (int i = 0; i < 3; ++i) {
        if (!blobs_[i].base) continue;
#if defined(_WIN32)
        UnmapViewOfFile(blobs_[i].base);
        if (blobs_[i].hMap) CloseHandle((HANDLE)blobs_[i].hMap);
        if (blobs_[i].hFile && blobs_[i].hFile != INVALID_HANDLE_VALUE) CloseHandle((HANDLE)blobs_[i].hFile);
#else
        munmap(blobs_[i].base, blobs_[i].file_size);
#endif
        blobs_[i].base = nullptr;
    }
}

bool TuttiAsyncPipeline::enqueue_read(int layer, int expert, const PhysicalExpert * pe) {
    if (!cfg_.enable_tutti || !pe) return false;
    ExpertKey key{layer, expert};
    std::unique_lock<std::mutex> lk(q_mtx_);
    if (in_flight_.find(key) != in_flight_.end()) {
        return true; // Already enqueued or staged
    }
    if (queue_.size() >= cfg_.tutti_queue_depth) {
        return false; // Queue full
    }

    auto done_flag = std::make_shared<std::atomic<bool>>(false);
    in_flight_[key] = done_flag;

    for (const auto & chunk : pe->chunks) {
        if (chunk.blob_idx < 0 || chunk.size_bytes == 0) continue;
        TuttiIoRequest req;
        req.layer = layer;
        req.expert = expert;
        req.blob_idx = chunk.blob_idx;
        req.file_offset = chunk.file_offset;
        req.size_bytes = chunk.size_bytes;
        req.queue_time = clock::now();
        req.done = done_flag;

        if (staging_pool_.is_active()) {
            size_t capacity = staging_pool_.get_capacity();
            if (capacity > 4000000) {
                size_t offset = (total_reads_.load() * 3481600) % (capacity - 3500000);
                req.staging_dest = static_cast<uint8_t *>(staging_pool_.get_base()) + offset;
            }
        }
        queue_.push_back(req);
        total_reads_++;
    }
    q_cv_.notify_one();
    return true;
}

bool TuttiAsyncPipeline::is_ready(int layer, int expert) const {
    ExpertKey key{layer, expert};
    std::unique_lock<std::mutex> lk(q_mtx_);
    auto it = in_flight_.find(key);
    if (it == in_flight_.end()) return false;
    return it->second->load();
}

void TuttiAsyncPipeline::wait_ready(int layer, int expert) {
    ExpertKey key{layer, expert};
    std::shared_ptr<std::atomic<bool>> done_flag;
    {
        std::unique_lock<std::mutex> lk(q_mtx_);
        auto it = in_flight_.find(key);
        if (it == in_flight_.end()) return;
        done_flag = it->second;
    }
    const auto t0 = clock::now();
    while (!done_flag->load() && running_) {
        std::this_thread::yield();
    }
    io_stall_ms_ += elapsed_ms(t0, clock::now());
}

void TuttiAsyncPipeline::clear() {
    std::unique_lock<std::mutex> lk(q_mtx_);
    queue_.clear();
    in_flight_.clear();
}

void TuttiAsyncPipeline::worker_fn() {
    while (true) {
        std::vector<TuttiIoRequest> batch;
        {
            std::unique_lock<std::mutex> lk(q_mtx_);
            q_cv_.wait(lk, [this] { return !queue_.empty() || !running_; });
            if (!running_ && queue_.empty()) break;
            batch.swap(queue_);
        }

#if defined(_WIN32)
        WIN32_MEMORY_RANGE_ENTRY win_entries[64];
        size_t n_win = 0;
        for (auto & req : batch) {
            if (req.blob_idx < 0 || req.blob_idx >= 3) continue;
            MmapBlob & mb = blobs_[req.blob_idx];
            if (!mb.base || req.file_offset + req.size_bytes > mb.file_size) continue;

            void * ptr = static_cast<uint8_t *>(mb.base) + req.file_offset;
            win_entries[n_win++] = { ptr, req.size_bytes };

            if (req.staging_dest) {
                std::memcpy(req.staging_dest, ptr, req.size_bytes);
            }
            bytes_staged_ += req.size_bytes;
            completed_reads_++;
            if (req.done) req.done->store(true);

            if (n_win == 64) {
                PrefetchVirtualMemory(GetCurrentProcess(), (ULONG)n_win, win_entries, 0);
                n_win = 0;
            }
        }
        if (n_win > 0) {
            PrefetchVirtualMemory(GetCurrentProcess(), (ULONG)n_win, win_entries, 0);
        }
#else
        for (auto & req : batch) {
            if (req.blob_idx < 0 || req.blob_idx >= 3) continue;
            MmapBlob & mb = blobs_[req.blob_idx];
            if (!mb.base || req.file_offset + req.size_bytes > mb.file_size) continue;

            void * ptr = static_cast<uint8_t *>(mb.base) + req.file_offset;
            madvise(ptr, req.size_bytes, MADV_WILLNEED);
            if (req.staging_dest) {
                std::memcpy(req.staging_dest, ptr, req.size_bytes);
            }
            bytes_staged_ += req.size_bytes;
            completed_reads_++;
            if (req.done) req.done->store(true);
        }
#endif
    }
}
'''

target_prefetcher_end = '''            if (n_win > 0) {
                PrefetchVirtualMemory(GetCurrentProcess(), (ULONG)n_win, win_entries, 0);
            }
#else
            for (const auto & r : batch) {
                if (r.blob_idx < 0 || r.blob_idx >= 3) continue;
                MmapEntry & me = blobs_[r.blob_idx];
                if (!me.base || r.file_offset + r.size_bytes > me.file_size) continue;
                void * ptr = static_cast<uint8_t *>(me.base) + r.file_offset;
                madvise(ptr, r.size_bytes, MADV_WILLNEED);
                pages_warmed++;
            }
#endif
        }
    }
};'''

assert target_prefetcher_end in text, "target_prefetcher_end not found"
text = text.replace(target_prefetcher_end, target_prefetcher_end + tutti_impl, 1)

# 2. Insert quick_evict_layer_except and fallback_access after ExpertMemoryManager::access
target_access = '''    // Host CPU expert access
    metrics.ram_hits++;
    const auto t_r0 = clock::now();
    if (cur_tl) cur_tl->ram_ms += elapsed_ms(t_r0, clock::now());
    return MemoryTier::RAM;
}'''

new_access_methods = '''    // Host CPU expert access
    metrics.ram_hits++;
    const auto t_r0 = clock::now();
    if (cur_tl) cur_tl->ram_ms += elapsed_ms(t_r0, clock::now());
    return MemoryTier::RAM;
}

void ExpertMemoryManager::quick_evict_layer_except(
    int layer, const std::unordered_set<ExpertKey, KeyHash> & keep_keys, TokenTimeline * cur_tl)
{
    if (!cfg.odmoe_quick_evict) return;
    std::lock_guard<std::mutex> lock(mtx);
    const auto t0 = clock::now();

    for (auto it = vram_lru.begin(); it != vram_lru.end(); ) {
        if (it->layer == layer) {
            ExpertKey victim = *it;
            if (keep_keys.find(victim) == keep_keys.end()) {
                size_t v_idx = size_t(victim.layer) * 512 + size_t(victim.expert);
                if (v_idx < TOTAL_EXPERTS && pinned_flat[v_idx]) {
                    ++it;
                    continue;
                }

                auto erase_it = it++;
                vram_lru.erase(erase_it);
                if (v_idx < TOTAL_EXPERTS) {
                    vram_present_flat[v_idx] = 0;
                    residency_flat[v_idx] = MemoryTier::RAM;
                    ram_lru.push_front(victim);
                    ram_iter_flat[v_idx] = ram_lru.begin();
                    ram_present_flat[v_idx] = 1;
                }
                const size_t sz = (v_idx < TOTAL_EXPERTS && expert_flat[v_idx].total_bytes > 0) ?
                                  expert_flat[v_idx].total_bytes : 3481600;
                vram_used_bytes = (vram_used_bytes > sz) ? (vram_used_bytes - sz) : 0;
                ram_used_bytes += sz;
                metrics.odmoe_quick_evictions++;
                metrics.vram_evictions++;
                continue;
            }
        }
        ++it;
    }

    if (cur_tl) {
        cur_tl->eviction_ms += elapsed_ms(t0, clock::now());
    }
}

MemoryTier ExpertMemoryManager::fallback_access(int layer, int expert, TokenTimeline * cur_tl) {
    auto current = get_residency(layer, expert);
    if (current != MemoryTier::VRAM) {
        metrics.spice_lossless_fallbacks++;
    }
    // Lossless fallback: demand load exact expert weights (never surrogate/approximate)
    return access(layer, expert, false, cur_tl);
}'''

assert target_access in text, "target_access not found"
text = text.replace(target_access, new_access_methods, 1)

# 3. Insert Predictor methods: predict_with_confidence & predict_layer_lead_with_confidence
target_pred_horizon = '''    std::vector<std::pair<int, double>> results(combined_scores.begin(), combined_scores.end());
    std::sort(results.begin(), results.end(), [](const auto & a, const auto & b) {
        return a.second > b.second;
    });
    return results;
}'''

new_pred_methods = '''    std::vector<std::pair<int, double>> results(combined_scores.begin(), combined_scores.end());
    std::sort(results.begin(), results.end(), [](const auto & a, const auto & b) {
        return a.second > b.second;
    });
    return results;
}

std::vector<ExpertPrediction> Predictor::predict_with_confidence(int layer, int k) {
    auto candidates = predict(layer, k);
    std::vector<ExpertPrediction> results;
    if (candidates.empty()) return results;

    auto pit = previous_token.find(layer);
    for (int exp : candidates) {
        double score = 0.20;
        if (pit != previous_token.end() && !pit->second.empty()) {
            for (int prev : pit->second) {
                const TransitionKey tkey{layer, prev};
                auto it = transitions.find(tkey);
                if (it != transitions.end()) {
                    auto nit = it->second.find(exp);
                    if (nit != it->second.end()) {
                        score += 0.40;
                        break;
                    }
                }
            }
            if (std::find(pit->second.begin(), pit->second.end(), exp) != pit->second.end()) {
                score += 0.30;
            }
        }
        score += 0.20 * std::min(1.0, get_sequence_frequency(layer, exp) * 5.0);
        double conf = std::min(1.0, std::max(0.10, score));

        ConfidenceTier ctier = ConfidenceTier::LOW;
        MemoryTier mtier = MemoryTier::NVME;
        if (conf >= cfg.spice_conf_high) {
            ctier = ConfidenceTier::HIGH;
            mtier = MemoryTier::VRAM;
        } else if (conf >= cfg.spice_conf_mid) {
            ctier = ConfidenceTier::MEDIUM;
            mtier = MemoryTier::RAM;
        }

        results.push_back({layer, exp, conf, ctier, mtier});
    }

    std::sort(results.begin(), results.end(), [](const ExpertPrediction & a, const ExpertPrediction & b) {
        return a.confidence > b.confidence;
    });
    return results;
}

std::vector<ExpertPrediction> Predictor::predict_layer_lead_with_confidence(
    int current_layer, int target_layer, const std::vector<int> & current_experts, int k)
{
    std::unordered_map<int, double> scores;
    auto it_l = cross_layer_transitions.find(current_layer);
    if (it_l != cross_layer_transitions.end() && !current_experts.empty()) {
        int src_matches = 0;
        for (int cur_e : current_experts) {
            auto it_src = it_l->second.find(cur_e);
            if (it_src != it_l->second.end() && !it_src->second.empty()) {
                src_matches++;
                uint64_t total = 0;
                for (const auto & kv : it_src->second) total += kv.second;
                if (total > 0) {
                    for (const auto & kv : it_src->second) {
                        scores[kv.first] += 0.55 * (double(kv.second) / double(total));
                    }
                }
            }
        }
        if (src_matches > 0) {
            for (auto & kv : scores) kv.second /= double(src_matches);
        }
    }

    auto pit = previous_token.find(target_layer);
    if (pit != previous_token.end() && !pit->second.empty()) {
        for (int p : pit->second) {
            scores[p] += 0.35;
            const TransitionKey tkey{target_layer, p};
            auto it = transitions.find(tkey);
            if (it != transitions.end() && !it->second.empty()) {
                uint64_t total = 0;
                for (const auto & kv : it->second) total += kv.second;
                if (total > 0) {
                    for (const auto & kv : it->second) {
                        scores[kv.first] += 0.35 * (double(kv.second) / double(total));
                    }
                }
            }
        }
    }

    auto it_cnt = sequence_expert_counts.find(target_layer);
    if (it_cnt != sequence_expert_counts.end() && total_tokens_observed > 0) {
        for (const auto & kv : it_cnt->second) {
            double freq = double(kv.second) / double(total_tokens_observed);
            if (freq > 0.02) {
                scores[kv.first] += 0.20 * std::min(1.0, freq * 5.0);
            }
        }
    }

    if (scores.empty()) {
        for (int i = 0; i < std::max(1, k); ++i) {
            scores[i] = 0.25;
        }
    }

    std::vector<std::pair<int, double>> ranked(scores.begin(), scores.end());
    std::sort(ranked.begin(), ranked.end(), [](const auto & a, const auto & b) {
        return a.second > b.second;
    });

    int budget = std::min<int>(cfg.max_candidates_per_layer, std::min<int>((int)ranked.size(), std::max(1, k)));
    std::vector<ExpertPrediction> results;
    results.reserve(budget);

    for (int i = 0; i < budget; ++i) {
        int exp = ranked[i].first;
        double conf = std::min(1.0, std::max(0.05, ranked[i].second));

        ConfidenceTier ctier = ConfidenceTier::LOW;
        MemoryTier mtier = MemoryTier::NVME;
        if (conf >= cfg.spice_conf_high) {
            ctier = ConfidenceTier::HIGH;
            mtier = MemoryTier::VRAM;
        } else if (conf >= cfg.spice_conf_mid) {
            ctier = ConfidenceTier::MEDIUM;
            mtier = MemoryTier::RAM;
        }

        results.push_back({target_layer, exp, conf, ctier, mtier});
    }

    return results;
}'''

assert target_pred_horizon in text, "target_pred_horizon not found"
text = text.replace(target_pred_horizon, new_pred_methods, 1)

# 4. Update Predictor::observe and Predictor::end_token for cross-layer tracking
target_observe = '''void Predictor::observe(int layer, const std::vector<int> & actual) {
    current_token[layer] = actual;
    for (int exp : actual) {
        sequence_expert_counts[layer][exp]++;
        ExpertKey k{layer, exp};
        auto it = last_observed_token.find(k);
        if (it != last_observed_token.end()) {
            uint64_t dist = total_tokens_observed > it->second ? (total_tokens_observed - it->second) : 1;
            total_reuse_distance += dist;
            reuse_distance_samples++;
        }
        last_observed_token[k] = total_tokens_observed;
    }
}'''

new_observe = '''void Predictor::observe(int layer, const std::vector<int> & actual) {
    current_token[layer] = actual;
    for (int exp : actual) {
        sequence_expert_counts[layer][exp]++;
        ExpertKey k{layer, exp};
        auto it = last_observed_token.find(k);
        if (it != last_observed_token.end()) {
            uint64_t dist = total_tokens_observed > it->second ? (total_tokens_observed - it->second) : 1;
            total_reuse_distance += dist;
            reuse_distance_samples++;
        }
        last_observed_token[k] = total_tokens_observed;
    }
    // OD-MoE & SPICE: Track cross-layer transitions from the preceding layer
    if (last_observed_layer >= 0 && last_observed_layer < layer && current_token.find(last_observed_layer) != current_token.end()) {
        for (int p : current_token[last_observed_layer]) {
            for (int a : actual) {
                cross_layer_transitions[last_observed_layer][p][a]++;
            }
        }
    }
    last_observed_layer = layer;
}'''

assert target_observe in text, "target_observe not found"
text = text.replace(target_observe, new_observe, 1)

target_end_token = '''void Predictor::end_token() {
    total_tokens_observed++;
    if (!previous_token.empty()) {
        for (const auto & [layer, current] : current_token) {
            auto pit = previous_token.find(layer);
            if (pit == previous_token.end()) continue;
            for (int p : pit->second)
                for (int n : current)
                    transitions[{layer, p}][n]++;
        }
    }
    previous_token = current_token;
    current_token.clear();
}'''

new_end_token = '''void Predictor::end_token() {
    total_tokens_observed++;
    if (!previous_token.empty()) {
        for (const auto & [layer, current] : current_token) {
            auto pit = previous_token.find(layer);
            if (pit == previous_token.end()) continue;
            for (int p : pit->second)
                for (int n : current)
                    transitions[{layer, p}][n]++;
        }
    }
    previous_token = current_token;
    current_token.clear();
    last_observed_layer = -1;
}'''

assert target_end_token in text, "target_end_token not found"
text = text.replace(target_end_token, new_end_token, 1)

# 5. Runtime constructor and destructor updates for TuttiAsyncPipeline
target_rt_ctor = '''    if (cfg.page_prefetch) {
        page_prefetcher = new PagePrefetcher();
        std::printf("[ATLAS] PagePrefetcher: ENABLED (batched PrefetchVirtualMemory mode)\\n");
    } else {
        std::printf("[ATLAS] PagePrefetcher: DISABLED (residency-only mode)\\n");
    }'''

new_rt_ctor = '''    if (cfg.enable_tutti) {
        tutti_pipeline = std::make_unique<TuttiAsyncPipeline>(cfg, host_staging_pool);
        std::printf("[ATLAS] Tutti Async NVMe Pipeline: ENABLED (decoupled I/O queue)\\n");
    }
    if (cfg.page_prefetch) {
        page_prefetcher = new PagePrefetcher();
        std::printf("[ATLAS] PagePrefetcher: ENABLED (batched PrefetchVirtualMemory mode)\\n");
    } else {
        std::printf("[ATLAS] PagePrefetcher: DISABLED (residency-only mode)\\n");
    }'''

assert target_rt_ctor in text, "target_rt_ctor not found"
text = text.replace(target_rt_ctor, new_rt_ctor, 1)

target_rt_dtor = '''Runtime::~Runtime() {
    if (page_prefetcher) {
        PagePrefetcher * pp = static_cast<PagePrefetcher *>(page_prefetcher);
        std::printf("[ATLAS] PagePrefetcher: pages_warmed=%" PRIu64 "\\n", pp->pages_warmed);
        delete pp;
        page_prefetcher = nullptr;
    }
    if (trace_file) { std::fclose(trace_file); trace_file = nullptr; }
}'''

new_rt_dtor = '''Runtime::~Runtime() {
    if (tutti_pipeline) {
        std::printf("[ATLAS] Tutti Async NVMe Pipeline: completed_reads=%" PRIu64 " bytes_staged=%.2f MB\\n",
            tutti_pipeline->get_completed_reads(), double(tutti_pipeline->get_bytes_staged()) / (1024.0 * 1024.0));
        tutti_pipeline.reset();
    }
    if (page_prefetcher) {
        PagePrefetcher * pp = static_cast<PagePrefetcher *>(page_prefetcher);
        std::printf("[ATLAS] PagePrefetcher: pages_warmed=%" PRIu64 "\\n", pp->pages_warmed);
        delete pp;
        page_prefetcher = nullptr;
    }
    if (trace_file) { std::fclose(trace_file); trace_file = nullptr; }
}'''

assert target_rt_dtor in text, "target_rt_dtor not found"
text = text.replace(target_rt_dtor, new_rt_dtor, 1)

# 6. Add Runtime::execute_odmoe_spice_prefetch & execute_odmoe_quick_evict
odmoe_runtime_methods = '''
void Runtime::execute_odmoe_spice_prefetch(int current_layer, const std::vector<int> & active_experts) {
    if (!cfg.enable_odmoe) return;
    const auto t0 = clock::now();
    const int lead = std::max(1, cfg.odmoe_layer_lead);
    const int k = expected_top_k > 0 ? expected_top_k : 10;
    auto & m = memory_manager.get_metrics();

    for (int h = 1; h <= lead; ++h) {
        int tgt_layer = (current_layer + h) % 48;
        auto preds = predictor.predict_layer_lead_with_confidence(current_layer, tgt_layer, active_experts, k);
        m.odmoe_prefetch_dispatches += preds.size();

        for (const auto & pred : preds) {
            if (m.odmoe_prefetch_dispatches > 0) {
                m.spice_avg_confidence = (m.spice_avg_confidence * 0.95) + (pred.confidence * 0.05);
            }

            const PhysicalExpert * pe = memory_manager.get_expert_info(tgt_layer, pred.expert);

            if (pred.conf_tier == ConfidenceTier::HIGH) {
                m.spice_vram_scheduled++;
                prefetcher.prefetch_expert_to_tier(tgt_layer, pred.expert, MemoryTier::VRAM, &current_timeline);
                if (cfg.enable_tutti && tutti_pipeline && pe) {
                    tutti_pipeline->enqueue_read(tgt_layer, pred.expert, pe);
                }
            } else if (pred.conf_tier == ConfidenceTier::MEDIUM) {
                m.spice_ram_scheduled++;
                prefetcher.prefetch_expert_to_tier(tgt_layer, pred.expert, MemoryTier::RAM, &current_timeline);
                if (cfg.enable_tutti && tutti_pipeline && pe) {
                    tutti_pipeline->enqueue_read(tgt_layer, pred.expert, pe);
                }
            } else {
                m.spice_nvme_left++;
            }
        }
    }
    current_timeline.prefetch_ms += elapsed_ms(t0, clock::now());
}

void Runtime::execute_odmoe_quick_evict(int completed_layer) {
    if (!cfg.enable_odmoe || !cfg.odmoe_quick_evict) return;
    std::unordered_set<ExpertKey, KeyHash> keep_keys;
    const int lead = std::max(1, cfg.odmoe_layer_lead);
    const int k = expected_top_k > 0 ? expected_top_k : 10;
    for (int h = 1; h <= lead; ++h) {
        int future_l = (completed_layer + h) % 48;
        auto preds = predictor.predict(future_l, k);
        for (int exp : preds) {
            keep_keys.insert({future_l, exp});
        }
    }
    memory_manager.quick_evict_layer_except(completed_layer, keep_keys, &current_timeline);
}
'''

target_gpu_first_end = '''    if (m.tokens_generated > 0) {
        m.expert_eviction_rate = (double)m.vram_evictions / (double)m.tokens_generated;
        m.avg_residency_duration_tokens = std::max(1.0, (double)m.resident_vram_experts / std::max(1.0, m.expert_eviction_rate + 0.1));
    }
}'''

assert target_gpu_first_end in text, "target_gpu_first_end not found"
text = text.replace(target_gpu_first_end, target_gpu_first_end + odmoe_runtime_methods, 1)

# 7. Hook into atlas_eval_callback: fallback_access & execute_odmoe_spice_prefetch & execute_odmoe_quick_evict
target_cb_access = '''            rt->memory_manager.pin(layer, exp_id, &rt->current_timeline);
            rt->memory_manager.access(layer, exp_id, false, &rt->current_timeline);'''

new_cb_access = '''            rt->memory_manager.pin(layer, exp_id, &rt->current_timeline);
            rt->memory_manager.fallback_access(layer, exp_id, &rt->current_timeline);'''

assert target_cb_access in text, "target_cb_access not found"
text = text.replace(target_cb_access, new_cb_access, 1)

target_cb_end = '''    rt->learning_locality.record_batch_experts(layer, batch_unique_layer_experts.data(), (int)batch_unique_layer_experts.size());'''

new_cb_end = '''    rt->learning_locality.record_batch_experts(layer, batch_unique_layer_experts.data(), (int)batch_unique_layer_experts.size());

    // OD-MoE & SPICE: Predictive Multi-Layer Prefetch and Quick Eviction
    if (rt->cfg.enable_odmoe) {
        rt->execute_odmoe_spice_prefetch(layer, batch_unique_layer_experts);
        rt->execute_odmoe_quick_evict(layer);
    }'''

assert target_cb_end in text, "target_cb_end not found"
text = text.replace(target_cb_end, new_cb_end, 1)

# 8. Add CLI argument parsing in main()
target_cli = '''        } else if (arg == "--atlas-grouped-gemm" && i + 1 < argc) {
            cfg.grouped_gemm = std::stoi(argv[++i]);'''

new_cli = '''        } else if (arg == "--atlas-odmoe-lead" && i + 1 < argc) {
            cfg.odmoe_layer_lead = std::stoi(argv[++i]);
            cfg.prefetch_lead = cfg.odmoe_layer_lead;
        } else if (arg == "--atlas-odmoe-quick-evict" && i + 1 < argc) {
            cfg.odmoe_quick_evict = (std::stoi(argv[++i]) != 0);
        } else if (arg == "--atlas-spice-conf-high" && i + 1 < argc) {
            cfg.spice_conf_high = std::stod(argv[++i]);
        } else if (arg == "--atlas-spice-conf-mid" && i + 1 < argc) {
            cfg.spice_conf_mid = std::stod(argv[++i]);
        } else if (arg == "--atlas-tutti-async-io" && i + 1 < argc) {
            cfg.enable_tutti = (std::stoi(argv[++i]) != 0);
        } else if (arg == "--atlas-grouped-gemm" && i + 1 < argc) {
            cfg.grouped_gemm = std::stoi(argv[++i]);'''

assert target_cli in text, "target_cli not found"
text = text.replace(target_cli, new_cli, 1)

# 9. Summary reports in main()
target_summary = '''    std::printf("Active Router Width:   %d\\n", llama_model_n_expert_used(model));
    std::printf("=================================================================\\n\\n");'''

new_summary = '''    if (cfg.enable_odmoe) {
        std::printf("OD-MoE Prefetch:       ACTIVE (Lead=%d layers, Quick-Evict=%s | Dispatches=%" PRIu64 ", Quick Evictions=%" PRIu64 ")\\n",
            cfg.odmoe_layer_lead, cfg.odmoe_quick_evict ? "ON" : "OFF",
            m.odmoe_prefetch_dispatches, m.odmoe_quick_evictions);
    }
    if (cfg.enable_spice) {
        std::printf("SPICE Scheduler:       LOSSLESS (High=%.2f -> VRAM: %" PRIu64 ", Mid=%.2f -> RAM: %" PRIu64 ", Low -> NVMe: %" PRIu64 " | Fallbacks: %" PRIu64 ", AvgConf: %.1f%%)\\n",
            cfg.spice_conf_high, m.spice_vram_scheduled,
            cfg.spice_conf_mid, m.spice_ram_scheduled,
            m.spice_nvme_left, m.spice_lossless_fallbacks, m.spice_avg_confidence * 100.0);
    }
    if (cfg.enable_tutti && runtime.tutti_pipeline) {
        std::printf("Tutti Async NVMe Pipe: ACTIVE (Reads Queued=%" PRIu64 ", Completed=%" PRIu64 ", IO Stall=%.2f ms, Staged=%.2f MB)\\n",
            runtime.tutti_pipeline->get_total_reads(), runtime.tutti_pipeline->get_completed_reads(),
            runtime.tutti_pipeline->get_io_stall_ms(), double(runtime.tutti_pipeline->get_bytes_staged()) / (1024.0 * 1024.0));
    }
    std::printf("Active Router Width:   %d\\n", llama_model_n_expert_used(model));
    std::printf("=================================================================\\n\\n");'''

assert target_summary in text, "target_summary not found"
text = text.replace(target_summary, new_summary, 1)

cpp_path.write_text(text, encoding="utf-8")
print("src/atlas/atlas-engine.cpp updated successfully! Lines:", len(text.splitlines()))
