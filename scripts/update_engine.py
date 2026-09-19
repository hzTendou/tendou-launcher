#!/usr/bin/env python3
"""Script to update atlas-engine.h with OD-MoE, SPICE, and Tutti definitions."""
from pathlib import Path

base = Path('experiments/runtime_audit/atlas-engine.h.before').read_text(encoding='utf-8')

# 1. Update headers with atomic, condition_variable, thread
old_headers = """#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <list>
#include <mutex>
#include <memory>
#include <chrono>"""

new_headers = """#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <list>
#include <mutex>
#include <memory>
#include <chrono>
#include <atomic>
#include <condition_variable>
#include <thread>"""

assert old_headers in base, "old_headers not found"
content = base.replace(old_headers, new_headers, 1)

# 2. Add SPICE Confidence Tiers & Prediction Structure after tier_to_str
old_tier = """inline const char * tier_to_str(MemoryTier t) {
    switch (t) {
        case MemoryTier::NVME: return "NVME";
        case MemoryTier::RAM:  return "RAM";
        case MemoryTier::VRAM: return "VRAM";
        default: return "UNKNOWN";
    }
}"""

new_tier = """inline const char * tier_to_str(MemoryTier t) {
    switch (t) {
        case MemoryTier::NVME: return "NVME";
        case MemoryTier::RAM:  return "RAM";
        case MemoryTier::VRAM: return "VRAM";
        default: return "UNKNOWN";
    }
}

// ---------------------------------------------------------------------------
// SPICE: Confidence Tiers & Prediction Structure (Lossless Guarantee)
// ---------------------------------------------------------------------------
enum class ConfidenceTier {
    LOW = 0,     // Stays on NVMe (< spice_conf_mid)
    MEDIUM = 1,  // Warmed in Host RAM / Pinned Staging Pool (>= spice_conf_mid && < spice_conf_high)
    HIGH = 2     // Prefetched to GPU VRAM (>= spice_conf_high)
};

inline const char * conf_tier_to_str(ConfidenceTier c) {
    switch (c) {
        case ConfidenceTier::LOW:    return "LOW (NVME)";
        case ConfidenceTier::MEDIUM: return "MEDIUM (RAM)";
        case ConfidenceTier::HIGH:   return "HIGH (VRAM)";
        default: return "UNKNOWN";
    }
}

struct ExpertPrediction {
    int layer = -1;
    int expert = -1;
    double confidence = 0.0;
    ConfidenceTier conf_tier = ConfidenceTier::LOW;
    MemoryTier target_tier = MemoryTier::NVME;
};"""

assert old_tier in content, "old_tier not found"
content = content.replace(old_tier, new_tier, 1)

# 3. Update Config struct with MTP, audit changes and OD-MoE, SPICE, Tutti
old_cfg = """    // MTP Speculative Decoding Configuration
    std::string mtp_mode      = "off"; // "off", "ram", "vram", "auto", "atlas"
    int mtp_draft_n           = 3;     // Proposal length N (1, 2, 3, 4, 6, 8)
    std::string mtp_path      = "";    // Empty = auto-detect Q4_K_M primary path
    std::string mtp_quant     = "Q4_K_M";
    std::string mtp_location  = "ram"; // Placement of MTP weights: "ram", "vram", "auto"
    bool enable_mtws          = true;  // Multi-Token Working Set planning & prefetch
    bool adaptive_fallback    = true;  // Adaptive N and fallback under memory pressure
    double target_acceptance  = 0.35;

    // Multi-Threading & Compute Co-Design
    int n_threads             = 14;    // Optimal 14 threads for AMD Ryzen 7 260 (eliminates SMT contention)
    float mtp_p_min           = 0.28f; // Minimum confidence: only verify draft tokens with p >= p_min
    int hotness_bonus         = 2;     // Eviction resistance passes for hot experts

    // Phase 1, 4 & 9: Speed-First Optimization & Quality Floor
    int expert_k              = 2;     // Optimal K=2 for Qwen3-8B-Flash-Next (7.88 - 9.05 TPS @ 94.2% quality)
    float quality_target      = 0.80f; // Target quality floor: 0.80 (Nominal K=10, Active K=2)"""

new_cfg = """    // MTP Speculative Decoding Configuration
    std::string mtp_mode      = "off"; // "off", "ram", "vram", "auto", "atlas"
    int mtp_draft_n           = 1;     // One-token drafting won the 16 GiB laptop sweep.
    bool verify_batch        = false; // Diagnostic: compare batched logits with sequential replay.
    std::string mtp_path      = "";    // Empty = auto-detect Q4_K_M primary path
    std::string mtp_quant     = "Q4_K_M";
    std::string mtp_location  = "ram"; // Placement of MTP weights: "ram", "vram", "auto"
    bool enable_mtws          = true;  // Multi-Token Working Set planning & prefetch
    bool adaptive_fallback    = true;  // Adaptive N and fallback under memory pressure
    double target_acceptance  = 0.35;

    // Multi-Threading & Compute Co-Design
    int n_threads             = 14;    // Optimal 14 threads for AMD Ryzen 7 260 (eliminates SMT contention)
    float mtp_p_min           = 0.28f; // Minimum confidence: only verify draft tokens with p >= p_min
    int hotness_bonus         = 2;     // Eviction resistance passes for hot experts

    // Phase 1, 4 & 9: Speed-First Optimization & Quality Floor
    int expert_k              = 0;     // 0 preserves the model's native router width.
    float quality_target      = 0.85f; // External task evaluation must meet this floor."""

assert old_cfg in content, "old_cfg not found"
content = content.replace(old_cfg, new_cfg, 1)

old_ipc = """    // IPC Mode for OpenAI Server Integration
    bool  ipc_mode            = false; // When true, runs persistent IPC loop for OpenAI API
};"""

new_ipc = """    // IPC Mode for OpenAI Server Integration
    bool  ipc_mode            = false; // When true, runs persistent IPC loop for OpenAI API
    bool  simulate_placement  = false; // Cost-model diagnostics, not CUDA execution.
    size_t prompt_cache_mb   = 256;   // Shared budget for full-prompt and chat-boundary checkpoints.

    // OD-MoE: Predictive Multi-Layer Expert Prefetch & Quick Eviction
    bool  enable_odmoe        = true;  // Multi-layer advance prefetching
    int   odmoe_layer_lead    = 2;     // Number of layers ahead to predict (e.g. 2-3 layers ahead)
    bool  odmoe_quick_evict   = true;  // Fast eviction of non-reused experts after layer compute

    // SPICE: Confidence-Aware Scheduler (Lossless)
    bool  enable_spice        = true;  // Confidence-aware 3-tier scheduling
    double spice_conf_high    = 0.65;  // High confidence threshold -> VRAM prefetch
    double spice_conf_mid     = 0.30;  // Mid confidence threshold -> RAM / pinned staging warm
    // Note: low confidence (< spice_conf_mid) stays on NVMe.
    // Prediction miss activates lossless fallback scheduler (exact weights, 0 surrogate).

    // Tutti: Async NVMe Pipeline
    bool  enable_tutti        = true;  // Decoupled disk I/O, pinned staging, async H2D pipeline
    size_t tutti_queue_depth  = 64;    // Maximum queued async I/O requests
};"""

assert old_ipc in content, "old_ipc not found"
content = content.replace(old_ipc, new_ipc, 1)

# 4. Add TuttiAsyncPipeline declaration after PinnedHostStagingPool
old_pinned = """    ggml_backend_buffer_t buf_ = nullptr;
    void * base_ptr_ = nullptr;
    size_t capacity_ = 0;
    bool enabled_ = false;
};"""

new_pinned = """    ggml_backend_buffer_t buf_ = nullptr;
    void * base_ptr_ = nullptr;
    size_t capacity_ = 0;
    bool enabled_ = false;
};

// ---------------------------------------------------------------------------
// Tutti: Async NVMe Pipeline (Decoupled Disk I/O, Pinned Staging & Async H2D)
// ---------------------------------------------------------------------------
struct TuttiIoRequest {
    int layer = -1;
    int expert = -1;
    int blob_idx = -1;
    size_t file_offset = 0;
    size_t size_bytes = 0;
    void * staging_dest = nullptr;
    clock::time_point queue_time{};
    std::shared_ptr<std::atomic<bool>> done;
};

class TuttiAsyncPipeline {
public:
    explicit TuttiAsyncPipeline(const Config & cfg, PinnedHostStagingPool & staging_pool);
    ~TuttiAsyncPipeline();

    bool enqueue_read(int layer, int expert, const PhysicalExpert * pe);
    bool is_ready(int layer, int expert) const;
    void wait_ready(int layer, int expert);
    void clear();

    uint64_t get_total_reads() const;
    uint64_t get_completed_reads() const;
    double get_io_stall_ms() const;
    size_t get_bytes_staged() const;

private:
    Config cfg_;
    PinnedHostStagingPool & staging_pool_;
    std::atomic<bool> running_{true};
    std::thread worker_;
    mutable std::mutex q_mtx_;
    std::condition_variable q_cv_;
    std::vector<TuttiIoRequest> queue_;
    std::unordered_map<ExpertKey, std::shared_ptr<std::atomic<bool>>, KeyHash> in_flight_;
    std::atomic<uint64_t> total_reads_{0};
    std::atomic<uint64_t> completed_reads_{0};
    std::atomic<size_t> bytes_staged_{0};
    double io_stall_ms_ = 0.0;

    struct MmapBlob {
        void * base = nullptr;
        size_t file_size = 0;
        void * hFile = nullptr;
        void * hMap  = nullptr;
    };
    MmapBlob blobs_[3];

    void init_mappings();
    void close_all_handles();
    void worker_fn();
};"""

assert old_pinned in content, "old_pinned not found"
content = content.replace(old_pinned, new_pinned, 1)

# 5. Add OD-MoE, SPICE, and Tutti metrics to Metrics struct
old_metrics = """    uint64_t useful_prefetches = 0;
    uint64_t wasted_prefetches = 0;

    ProfilerStats profiler;"""

new_metrics = """    uint64_t useful_prefetches = 0;
    uint64_t wasted_prefetches = 0;

    // OD-MoE Metrics
    uint64_t odmoe_prefetch_dispatches = 0;
    uint64_t odmoe_quick_evictions = 0;
    uint64_t odmoe_lead_hits = 0;

    // SPICE Metrics
    uint64_t spice_vram_scheduled = 0;
    uint64_t spice_ram_scheduled = 0;
    uint64_t spice_nvme_left = 0;
    uint64_t spice_lossless_fallbacks = 0;
    double   spice_avg_confidence = 0.0;

    // Tutti Metrics
    uint64_t tutti_io_requests = 0;
    uint64_t tutti_io_completed = 0;
    double   tutti_io_stall_ms = 0.0;
    size_t   tutti_bytes_staged = 0;
    double   tutti_overlap_efficiency = 0.0;

    ProfilerStats profiler;"""

assert old_metrics in content, "old_metrics not found"
content = content.replace(old_metrics, new_metrics, 1)

# 6. Add methods to ExpertMemoryManager
old_mm = """    void evict_vram(size_t needed_bytes, TokenTimeline * cur_tl = nullptr);
    void evict_ram(size_t needed_bytes, TokenTimeline * cur_tl = nullptr);
    void print_summary() const;"""

new_mm = """    void evict_vram(size_t needed_bytes, TokenTimeline * cur_tl = nullptr);
    void evict_ram(size_t needed_bytes, TokenTimeline * cur_tl = nullptr);
    void quick_evict_layer_except(int layer, const std::unordered_set<ExpertKey, KeyHash> & keep_keys, TokenTimeline * cur_tl = nullptr);
    MemoryTier fallback_access(int layer, int expert, TokenTimeline * cur_tl = nullptr);
    void print_summary() const;"""

assert old_mm in content, "old_mm not found"
content = content.replace(old_mm, new_mm, 1)

# 7. Add methods & fields to Predictor
old_pred = """    double get_sequence_frequency(int layer, int expert) const;
    std::vector<ExpertPlacementScore> score_experts(const ExpertCostModel & cost_model, double vram_budget_mb);
    double get_prediction_accuracy() const;
    double get_avg_reuse_distance() const;
    void record_prediction_result(const std::vector<int> & predicted, const std::vector<int> & actual);

private:"""

new_pred = """    double get_sequence_frequency(int layer, int expert) const;
    std::vector<ExpertPlacementScore> score_experts(const ExpertCostModel & cost_model, double vram_budget_mb);
    double get_prediction_accuracy() const;
    double get_avg_reuse_distance() const;
    void record_prediction_result(const std::vector<int> & predicted, const std::vector<int> & actual);

    // OD-MoE & SPICE: Confidence-aware advance predictions
    std::vector<ExpertPrediction> predict_with_confidence(int layer, int k);
    std::vector<ExpertPrediction> predict_layer_lead_with_confidence(int current_layer, int target_layer, const std::vector<int> & current_experts, int k);

private:
    // OD-MoE & SPICE: Cross-layer routing correlations: src_layer -> src_expert -> tgt_expert -> count
    std::unordered_map<int, std::unordered_map<int, std::unordered_map<int, uint64_t>>> cross_layer_transitions;
    int last_observed_layer = -1;"""

assert old_pred in content, "old_pred not found"
content = content.replace(old_pred, new_pred, 1)

# 8. Add Tutti and OD-MoE methods to Runtime
old_rt = """    GroupedGemmStreamPipeline grouped_pipeline;

    explicit Runtime(const Config & c);
    ~Runtime();

    void record_trace_line(const char * phase, long long call_idx, int layer, int k, int n_tok, const int * expert_ids);
    void finish_token_timeline(int token_idx, double total_decode_ms, double sampling_ms, double cuda_sync_ms);
    void print_bottleneck_report() const;
    void print_mtws_report() const;
    void print_gpu_first_report() const;
    void execute_gpu_first_token(int token_idx, const std::vector<int> * layer_experts);
};"""

new_rt = """    GroupedGemmStreamPipeline grouped_pipeline;
    std::unique_ptr<TuttiAsyncPipeline> tutti_pipeline;

    explicit Runtime(const Config & c);
    ~Runtime();

    void record_trace_line(const char * phase, long long call_idx, int layer, int k, int n_tok, const int * expert_ids);
    void finish_token_timeline(int token_idx, double total_decode_ms, double sampling_ms, double cuda_sync_ms);
    void print_bottleneck_report() const;
    void print_mtws_report() const;
    void print_gpu_first_report() const;
    void execute_gpu_first_token(int token_idx, const std::vector<int> * layer_experts);

    // OD-MoE & SPICE runtime orchestration
    void execute_odmoe_spice_prefetch(int current_layer, const std::vector<int> & active_experts);
    void execute_odmoe_quick_evict(int completed_layer);
};"""

assert old_rt in content, "old_rt not found"
content = content.replace(old_rt, new_rt, 1)

Path('src/atlas/atlas-engine.h').write_text(content, encoding='utf-8')
print('src/atlas/atlas-engine.h written successfully! Lines:', len(content.splitlines()))
