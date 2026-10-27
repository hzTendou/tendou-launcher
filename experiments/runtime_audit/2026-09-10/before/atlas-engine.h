#pragma once

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdint>
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
#include <thread>

namespace atlas {

using clock = std::chrono::high_resolution_clock;

enum class MemoryTier {
    NVME = 0,
    RAM = 1,
    VRAM = 2
};

inline const char * tier_to_str(MemoryTier t) {
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
};

struct ExpertKey {
    int layer;
    int expert;
    bool operator==(const ExpertKey & o) const {
        return layer == o.layer && expert == o.expert;
    }
};

struct KeyHash {
    size_t operator()(const ExpertKey & k) const {
        return (size_t(uint32_t(k.layer)) << 32) ^ uint32_t(k.expert);
    }
};

struct ChunkLocation {
    std::string kind;
    std::string tensor;
    std::string blob_path;
    int blob_idx = -1;
    size_t file_offset = 0;
    size_t size_bytes = 0;
    std::string type;
};

struct PrefetchRange {
    int blob_idx = -1;
    size_t file_offset = 0;
    size_t size_bytes = 0;
};

struct PhysicalExpert {
    int layer;
    int expert;
    size_t total_bytes;
    std::vector<ChunkLocation> chunks;
};

struct Config {
    int max_candidates_per_layer = 10;
    int source_top_n = 8;
    int min_count = 1;
    double confidence_floor = 0.22;
    bool verbose = true;

    // Memory Tier Budgets (in Megabytes)
    size_t vram_cap_mb = 6600; // Product-level full VRAM capacity utilization (~6.6 GB VRAM)
    size_t ram_cap_mb  = 6144; // 6 GB RAM for host expert cache tier

    // Diagnostic / Memory-Tier testing mode (Section 24)
    char test_tier = 0; // 'A', 'B', 'C', 'D', 'E', or 0 (normal)
    std::string trace_output_path = "";
    std::string physical_map_path = "config/atlas_physical_map_qwen38.json";

    // Performance & Profiling flags
    bool enable_profiling = true;
    bool fast_cb          = true;  // Fast callback mode: skip synchronous topk readback (avoids 48 GPU syncs/tok)
    bool async_trace      = true;
    int  prefetch_lead    = 1;
    bool page_prefetch    = false; // When true: call PrefetchVirtualMemory on predicted expert pages

    // MTP Speculative Decoding Configuration
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
    float quality_target      = 0.85f; // External task evaluation must meet this floor.
    int gpu_expert_layers     = 2;     // Product-level default: Layers 0-1 full MoE on GPU (6,563 MiB total)

    // Grouped-GEMM & Concurrent CUDA Streams (Section 2 & 8)
    int grouped_gemm          = 2;     // 1, 2, or 3 concurrent Grouped-GEMMs
    int cuda_streams          = 2;     // 1, 2, or 3 concurrent CUDA streams
    bool async_h2d            = true;  // Asynchronous H2D PCIe transfer overlap

    // Dynamic Confidence-Based K Routing
    float dynamic_k_thresh    = 0.0f;  // 0.0 = disabled (static K), > 0.0 = threshold (e.g. 0.65)
    int   dynamic_k_start     = 2;     // First layer where dynamic K is allowed (layers 0-1 are anchors)
    int   dynamic_k_end       = 45;    // Last layer where dynamic K is allowed (layers 46-47 are anchors)

    // Fast MoE Stride & Boost Mode (Target: 15+ TPS decode, 18+ TPS prefill, Quality >= 80%)
    int   moe_stride          = 1;     // 1 = all layers MoE, 2 = every 2nd layer, 3 = every 3rd layer
    bool  boost_mode          = false; // Target boost: 15 TPS decode, 18 TPS prefill, quality >= 80%

    // Product-Level GPU-First Compute Architecture
    bool  gpu_first           = true;  // Product-level default: GPU-first expert execution active
    bool  cpu_only            = false; // When true, runs CPU fallback engine without GPU offloading
    int   vram_cache_mb       = 3200;  // Dedicated VRAM expert cache budget (~919 experts on 8 GB VRAM)
    bool  heterogeneous_exec  = true;  // Parallel CPU + GPU heterogeneous execution
    float gpu_residency_thresh= 0.30f; // Predictor confidence threshold for VRAM promotion

    // Phase 3: Sampled Router Readback (eliminates 2.10 ms/token GPU→CPU overhead)
    int   readback_interval   = 8;     // Full GPU→CPU topk readback every N tokens (0 = every token)
    int   readback_warmup     = 2;     // Always full readback for first N tokens (cache warmup period)

    // IPC Mode for OpenAI Server Integration
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
};

// ---------------------------------------------------------------------------
// Quality Estimator (Phase 10: Multi-Indicator Quality Tracking & Floor Guard)
// ---------------------------------------------------------------------------
class QualityEstimator {
public:
    void record_token(llama_token tok) {
        tokens.push_back(tok);
        token_freq[tok]++;
    }

    const std::vector<llama_token> & get_tokens() const { return tokens; }

    double compute_repetition_penalty() const {
        if (tokens.size() < 8) return 0.5;
        int n_repeats_4 = 0;
        for (size_t i = 4; i < tokens.size(); ++i) {
            if (tokens[i] == tokens[i - 4]) {
                n_repeats_4++;
            }
        }
        double rep_rate = double(n_repeats_4) / double(tokens.size());
        return std::max(0.0, 1.0 - rep_rate * 1.5);
    }

    double compute_entropy_score() const {
        if (tokens.size() < 16) return 0.5;
        double entropy = 0.0;
        const double n = double(tokens.size());
        for (const auto & kv : token_freq) {
            double p = double(kv.second) / n;
            if (p > 0.0) entropy -= p * std::log2(p);
        }
        // Healthy token generation exhibits entropy >= 4.0 bits. Normalize against 4.2 bits.
        double normalized = entropy / 4.2;
        return std::min(1.0, std::max(0.2, normalized));
    }

    double get_quality_score() const {
        if (tokens.empty()) return 0.0;
        double rep = compute_repetition_penalty();
        double ent = compute_entropy_score();
        double base_score = 0.60 * rep + 0.40 * ent;

        // QualityEstimator Bug Fix: Penalize very short or prematurely failed generations.
        // A generation with < 16 tokens cannot claim 100% quality.
        if (tokens.size() < 16) {
            double length_penalty = double(tokens.size()) / 16.0;
            base_score *= length_penalty;
        }

        return base_score;
    }

    size_t token_count() const { return tokens.size(); }

private:
    std::vector<llama_token> tokens;
    std::unordered_map<llama_token, uint32_t> token_freq;
};

// ---------------------------------------------------------------------------
// Self-Learning Expert Locality (Phase 11: Persistent Online Learning)
// ---------------------------------------------------------------------------
class SelfLearningLocality {
public:
    static constexpr size_t NUM_LAYERS = 48;
    static constexpr size_t MAX_TRACKED_EXPERTS = 512;  // Full expert range (was 64); 48x512x4 = 96KB, L2-resident

    uint32_t expert_cooccurrences[NUM_LAYERS][MAX_TRACKED_EXPERTS] = {};

    void record_batch_experts(int layer, const int * experts, int count) {
        if (layer < 0 || layer >= (int)NUM_LAYERS) return;
        for (int i = 0; i < count; ++i) {
            int e = experts[i];
            if (e >= 0 && e < (int)MAX_TRACKED_EXPERTS) {
                expert_cooccurrences[layer][e]++;
            }
        }
    }

    void save_to_file(const std::string & path) const {
        FILE * f = std::fopen(path.c_str(), "wb");
        if (f) {
            std::fwrite(expert_cooccurrences, sizeof(expert_cooccurrences), 1, f);
            std::fclose(f);
        }
    }

    void load_from_file(const std::string & path) {
        FILE * f = std::fopen(path.c_str(), "rb");
        if (f) {
            size_t r = std::fread(expert_cooccurrences, sizeof(expert_cooccurrences), 1, f);
            (void)r;
            std::fclose(f);
        }
    }
};

// ---------------------------------------------------------------------------
// Pinned Host Memory Staging Pool (Section 3: Zero-Copy PCIe Staging)
// ---------------------------------------------------------------------------
class PinnedHostStagingPool {
public:
    explicit PinnedHostStagingPool(size_t capacity_bytes = 668 * 1024 * 1024)
        : capacity_(capacity_bytes) {
        ggml_backend_buffer_type_t host_buft = ggml_backend_cuda_host_buffer_type();
        if (host_buft) {
            buf_ = ggml_backend_buft_alloc_buffer(host_buft, capacity_);
            if (buf_) {
                base_ptr_ = ggml_backend_buffer_get_base(buf_);
                enabled_ = (base_ptr_ != nullptr);
            }
        }
        if (!enabled_) {
            base_ptr_ = std::malloc(capacity_);
            enabled_ = (base_ptr_ != nullptr);
        }
    }

    ~PinnedHostStagingPool() {
        if (buf_) {
            ggml_backend_buffer_free(buf_);
            buf_ = nullptr;
            base_ptr_ = nullptr;
        } else if (base_ptr_) {
            std::free(base_ptr_);
            base_ptr_ = nullptr;
        }
    }

    void * get_base() const { return base_ptr_; }
    size_t get_capacity() const { return capacity_; }
    bool is_active() const { return enabled_; }
    bool is_pinned() const { return buf_ != nullptr; }

private:
    ggml_backend_buffer_t buf_ = nullptr;
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
    void * expert_base_staging = nullptr;
    clock::time_point queue_time{};
    std::shared_ptr<std::atomic<bool>> done;
    std::shared_ptr<std::atomic<int>> remaining_chunks;
};

class TuttiAsyncPipeline {
public:
    explicit TuttiAsyncPipeline(const Config & cfg, PinnedHostStagingPool & staging_pool);
    ~TuttiAsyncPipeline();

    bool enqueue_read(int layer, int expert, const PhysicalExpert * pe);
    bool is_ready(int layer, int expert) const;
    void wait_ready(int layer, int expert);
    void * get_staged_ptr(int layer, int expert) const;
    void clear();

    uint64_t get_total_reads() const { return total_reads_.load(); }
    uint64_t get_completed_reads() const { return completed_reads_.load(); }
    double get_io_stall_ms() const { return io_stall_ms_; }
    size_t get_bytes_staged() const { return bytes_staged_.load(); }

private:
    Config cfg_;
    PinnedHostStagingPool & staging_pool_;
    std::atomic<bool> running_{true};
    std::thread worker_;
    mutable std::mutex q_mtx_;
    std::condition_variable q_cv_;
    std::vector<TuttiIoRequest> queue_;
    std::unordered_map<ExpertKey, std::shared_ptr<std::atomic<bool>>, KeyHash> in_flight_;
    std::unordered_map<ExpertKey, void *, KeyHash> staged_ptrs_;
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
};

// ---------------------------------------------------------------------------
// CPU/GPU Expert Cost Model (Section 5: Dynamic Routing Decision Engine)
// ---------------------------------------------------------------------------
struct ExpertCostModel {
    double cpu_gemm_per_expert_ms = 3.30; // AVX2 GEMM baseline
    double pcie_h2d_per_expert_ms = 0.27; // 3.4 MB @ 12.8 GB/s
    double gpu_gemm_per_expert_ms = 0.05; // RTX 5060 Tensor Cores
    double sync_penalty_ms        = 0.10;

    enum class TargetEngine {
        GPU_HOT_CACHE,
        GPU_ASYNC_H2D,
        CPU_AVX2_FALLBACK
    };

    TargetEngine evaluate(bool in_gpu_cache, bool in_pinned_ram) const {
        if (in_gpu_cache) {
            return TargetEngine::GPU_HOT_CACHE; // 0.05 ms: zero H2D latency
        }
        if (in_pinned_ram) {
            double gpu_cost = pcie_h2d_per_expert_ms + gpu_gemm_per_expert_ms + sync_penalty_ms;
            if (gpu_cost < cpu_gemm_per_expert_ms) {
                return TargetEngine::GPU_ASYNC_H2D; // 0.42 ms vs 3.30 ms CPU
            }
        }
        return TargetEngine::CPU_AVX2_FALLBACK;
    }
};

struct ExpertPlacementScore {
    int layer = -1;
    int expert = -1;
    double probability = 0.0;
    double expected_reuse = 1.0;
    double gpu_execution_benefit = 66.0;
    double expert_compute_cost = 3.30;
    double transfer_cost = 0.27;
    double residency_cost = 0.05;
    double placement_score = 0.0;
    double expected_gpu_savings = 0.0;
    double residency_value = 0.0;
    bool should_be_in_vram = false;
};

// ---------------------------------------------------------------------------
// Grouped-GEMM & Concurrent CUDA Stream Pipeline (Section 2 & 8)
// ---------------------------------------------------------------------------
struct GroupedGemmTask {
    int layer = -1;
    int token_idx = 0;
    std::vector<int> expert_ids;
    size_t batch_size = 0;
    clock::time_point launch_time{};
    clock::time_point complete_time{};
};

class GroupedGemmStreamPipeline {
public:
    static constexpr size_t MAX_STREAMS = 3;

    explicit GroupedGemmStreamPipeline(int n_streams = 2, int n_grouped = 2, bool async_h2d = true)
        : num_streams_(std::max(1, std::min((int)MAX_STREAMS, n_streams))),
          num_grouped_(std::max(1, std::min(3, n_grouped))),
          async_h2d_enabled_(async_h2d) {}

    void dispatch_token_group(int layer, int token_idx, const int * experts, int k) {
        size_t stream_idx = token_idx % num_streams_;
        GroupedGemmTask task;
        task.layer = layer;
        task.token_idx = token_idx;
        task.expert_ids.assign(experts, experts + k);
        task.batch_size = size_t(k);
        task.launch_time = clock::now();

        stream_tasks_[stream_idx].push_back(task);
        invocations_++;
        active_groups_ += (k > 1 ? 1 : 0);
    }

    void finish_batch() {
        for (size_t s = 0; s < num_streams_; ++s) {
            completed_tasks_[s] += stream_tasks_[s].size();
            stream_tasks_[s].clear();
        }
    }

    int get_num_streams() const { return num_streams_; }
    int get_num_grouped() const { return num_grouped_; }
    bool is_async_h2d() const { return async_h2d_enabled_; }
    uint64_t get_invocations() const { return invocations_; }
    uint64_t get_active_groups() const { return active_groups_; }
    uint64_t get_completed_tasks(size_t s) const { return (s < MAX_STREAMS) ? completed_tasks_[s] : 0; }

private:
    int num_streams_ = 2;
    int num_grouped_ = 2;
    bool async_h2d_enabled_ = true;
    uint64_t invocations_ = 0;
    uint64_t active_groups_ = 0;
    std::vector<GroupedGemmTask> stream_tasks_[MAX_STREAMS];
    uint64_t completed_tasks_[MAX_STREAMS] = {};
};

struct TokenTimeline {
    int token_idx = 0;
    double router_ms = 0.0;
    double expert_lookup_ms = 0.0;
    double cache_lookup_ms = 0.0;
    double nvme_ms = 0.0;
    double ram_ms = 0.0;
    double pcie_ms = 0.0;
    double vram_alloc_ms = 0.0;
    double pin_ms = 0.0;
    double eviction_ms = 0.0;
    double prefetch_ms = 0.0;
    double cuda_sync_ms = 0.0;
    double expert_compute_ms = 0.0;
    double sampling_ms = 0.0;
    double total_token_ms = 0.0;
};

struct ProfilerStats {
    double total_router_ms = 0.0;
    double total_expert_lookup_ms = 0.0;
    double total_cache_lookup_ms = 0.0;
    double total_nvme_ms = 0.0;
    double total_ram_ms = 0.0;
    double total_pcie_ms = 0.0;
    double total_vram_alloc_ms = 0.0;
    double total_pin_ms = 0.0;
    double total_eviction_ms = 0.0;
    double total_prefetch_ms = 0.0;
    double total_cuda_sync_ms = 0.0;
    double total_expert_compute_ms = 0.0;
    double total_sampling_ms = 0.0;
    double total_token_ms = 0.0;
    uint64_t token_count = 0;
    std::vector<TokenTimeline> timelines;
};

struct Metrics {
    uint64_t tokens_generated = 0;
    uint64_t router_events = 0;
    uint64_t decode_router_events = 0;
    uint64_t predicted_candidates = 0;
    uint64_t prefetch_requests = 0;
    uint64_t prefetch_planes = 0;
    uint64_t prefetch_bytes = 0;
    uint64_t warmup_events = 0;

    // Atlas Hierarchical Memory Metrics
    uint64_t vram_hits = 0;
    uint64_t ram_hits = 0;
    uint64_t nvme_misses = 0;
    uint64_t vram_evictions = 0;
    uint64_t ram_evictions = 0;
    uint64_t bytes_nvme_to_ram = 0;
    uint64_t bytes_ram_to_vram = 0;

    double startup_time_ms = 0.0;
    double prompt_eval_time_ms = 0.0;
    double generation_time_ms = 0.0;
    uint64_t prompt_tokens = 0;

    // Speculative / MTP Metrics
    std::string mtp_mode = "off";
    std::string mtp_quant = "Q4_K_M";
    std::string mtp_location = "ram";
    int mtp_draft_n = 0;
    uint64_t spec_draft_tokens = 0;
    uint64_t spec_accepted_tokens = 0;
    double spec_draft_time_ms = 0.0;
    double spec_verify_time_ms = 0.0;

    // Atlas MTWS (Multi-Token Working Set) Metrics (Sections 15, 16, 17)
    uint64_t mtws_plans_generated = 0;
    uint64_t mtws_future_candidates = 0;
    uint64_t mtws_unique_predicted = 0;
    uint64_t mtws_predicted_accesses = 0;
    uint64_t mtws_actual_accesses = 0;
    uint64_t mtws_actual_unique = 0;
    uint64_t mtws_correct_unique = 0;      // intersection: predicted unique && actual unique
    uint64_t mtws_expert_reuses = 0;       // expert reused by multiple tokens in same batch
    uint64_t prefetch_completed_before_demand = 0; // prefetch hit
    uint64_t prefetch_late = 0;            // prefetch stall
    uint64_t wasted_prefetch_bytes = 0;
    size_t vram_working_set_bytes = 0;
    size_t ram_working_set_bytes = 0;
    size_t nvme_working_set_bytes = 0;
    double total_prefetch_ms = 0.0;
    uint64_t dynamic_k1_count = 0;
    uint64_t dynamic_k2_count = 0;

    // Experimental GPU-First Compute Architecture Metrics (Phase 1 & Target Goal)
    uint64_t gpu_expert_computes = 0;
    uint64_t cpu_expert_computes = 0;
    uint64_t parallel_heterogeneous_dispatches = 0;
    double total_gpu_expert_compute_ms = 0.0;
    double total_cpu_expert_compute_ms = 0.0;
    double total_h2d_transfer_ms = 0.0;
    double total_d2h_transfer_ms = 0.0;
    double total_transfer_stall_ms = 0.0;
    double total_sync_stall_ms = 0.0;
    double total_gpu_idle_ms = 0.0;
    double total_cpu_idle_ms = 0.0;
    double gpu_utilization_pct = 0.0;
    double cpu_utilization_pct = 0.0;
    double vram_utilization_mb = 0.0;
    double vram_capacity_mb = 8151.0;
    size_t resident_vram_experts = 0;
    double predictor_accuracy = 0.0;
    double vram_cache_hit_rate = 0.0;
    double h2d_bandwidth_gbps = 12.8;
    double d2h_bandwidth_gbps = 13.1;
    double avg_residency_duration_tokens = 0.0;
    double expert_eviction_rate = 0.0;
    double prefetch_success_rate = 0.0;
    double avg_expert_reuse_distance = 0.0;
    uint64_t useful_prefetches = 0;
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

    ProfilerStats profiler;
};

class ExpertMemoryManager {
public:
    explicit ExpertMemoryManager(const Config & cfg);
    ~ExpertMemoryManager();

    bool load_physical_map(const std::string & path);
    MemoryTier access(int layer, int expert, bool is_prefetch, TokenTimeline * cur_tl = nullptr);
    void pin(int layer, int expert, TokenTimeline * cur_tl = nullptr);
    void unpin_all(TokenTimeline * cur_tl = nullptr);
    void flush_vram();
    void flush_all();

    MemoryTier get_residency(int layer, int expert) const;
    size_t get_vram_used_bytes() const { return vram_used_bytes; }
    size_t get_ram_used_bytes() const { return ram_used_bytes; }
    size_t get_vram_capacity_bytes() const { return vram_capacity_bytes; }
    size_t get_ram_capacity_bytes() const { return ram_capacity_bytes; }

    const PhysicalExpert * get_expert_info(int layer, int expert) const;
    const Metrics & get_metrics() const { return metrics; }
    Metrics & get_metrics() { return metrics; }
    void set_residency(int layer, int expert, MemoryTier tier);
    size_t get_vram_resident_count() const;
    void apply_placement_plan(const std::vector<ExpertPlacementScore> & scores);
    void evict_vram(size_t needed_bytes, TokenTimeline * cur_tl = nullptr);
    void evict_ram(size_t needed_bytes, TokenTimeline * cur_tl = nullptr);
    void quick_evict_layer_except(int layer, const std::unordered_set<int> & keep_experts, TokenTimeline * cur_tl = nullptr);
    void quick_evict_layer_except(int layer, const std::unordered_set<ExpertKey, KeyHash> & keep_keys, TokenTimeline * cur_tl = nullptr);
    MemoryTier fallback_access(int layer, int expert, TokenTimeline * cur_tl = nullptr);
    void print_summary() const;

    friend class Prefetcher;

private:
    Config cfg;
    Metrics metrics;
    std::unordered_map<ExpertKey, PhysicalExpert, KeyHash> physical_map;

    // Residency tracking
    std::unordered_map<ExpertKey, MemoryTier, KeyHash> residency;

    // VRAM Cache (LRU — front = most recent, back = evict candidate)
    std::list<ExpertKey> vram_lru;
    std::unordered_map<ExpertKey, std::list<ExpertKey>::iterator, KeyHash> vram_map;
    size_t vram_used_bytes = 0;
    size_t vram_capacity_bytes = 0;

    // RAM Cache (LRU)
    std::list<ExpertKey> ram_lru;
    std::unordered_map<ExpertKey, std::list<ExpertKey>::iterator, KeyHash> ram_map;
    size_t ram_used_bytes = 0;
    size_t ram_capacity_bytes = 0;

    static constexpr size_t TOTAL_EXPERTS = 48 * 512;
    std::vector<PhysicalExpert> expert_flat;
    std::vector<MemoryTier> residency_flat;
    std::vector<uint8_t> pinned_flat;
    std::vector<std::list<ExpertKey>::iterator> vram_iter_flat;
    std::vector<uint8_t> vram_present_flat;
    std::vector<std::list<ExpertKey>::iterator> ram_iter_flat;
    std::vector<uint8_t> ram_present_flat;
    std::vector<uint8_t> hotness_flat;

    std::mutex mtx;
};

class Predictor {
public:
    explicit Predictor(const Config & cfg);

    std::vector<int> predict(int layer, int k);
    std::vector<std::pair<int, double>> predict_horizon(int layer, int horizon, const std::vector<int> & current_experts);
    void observe(int layer, const std::vector<int> & actual);
    void end_token();
    double get_sequence_frequency(int layer, int expert) const;
    std::vector<ExpertPlacementScore> score_experts(const ExpertCostModel & cost_model, double vram_budget_mb);
    double get_prediction_accuracy() const;
    double get_avg_reuse_distance() const;
    void record_prediction_result(const std::vector<int> & predicted, const std::vector<int> & actual);

    // OD-MoE & SPICE: Confidence-aware advance predictions
    std::vector<ExpertPrediction> predict_with_confidence(int layer, int k);
    std::vector<ExpertPrediction> predict_layer_lead_with_confidence(int current_layer, int target_layer, const std::vector<int> & current_experts, int k);
    std::unordered_map<int, std::unordered_set<int>> lead_predictions;

private:
    // OD-MoE & SPICE: Multi-layer cross-layer routing correlations: (src_layer, tgt_layer, src_expert) -> tgt_expert -> count
    struct CrossLayerKey {
        int src_layer;
        int tgt_layer;
        int src_expert;
        bool operator==(const CrossLayerKey & o) const {
            return src_layer == o.src_layer && tgt_layer == o.tgt_layer && src_expert == o.src_expert;
        }
    };
    struct CrossLayerKeyHash {
        size_t operator()(const CrossLayerKey & k) const {
            return (size_t(uint16_t(k.src_layer)) << 48) ^ (size_t(uint16_t(k.tgt_layer)) << 32) ^ uint32_t(k.src_expert);
        }
    };
    std::unordered_map<CrossLayerKey, std::unordered_map<int, uint64_t>, CrossLayerKeyHash> cross_layer_transitions;
    int last_observed_layer = -1;
    struct TransitionKey {
        int layer;
        int expert;
        bool operator==(const TransitionKey & o) const { return layer == o.layer && expert == o.expert; }
    };
    struct TransKeyHash {
        size_t operator()(const TransitionKey & k) const {
            return (size_t(uint32_t(k.layer)) << 32) ^ uint32_t(k.expert);
        }
    };

    Config cfg;
    std::unordered_map<TransitionKey, std::unordered_map<int, uint64_t>, TransKeyHash> transitions;
    std::unordered_map<int, std::vector<int>> previous_token;
    std::unordered_map<int, std::vector<int>> current_token;
    std::unordered_map<int, std::unordered_map<int, uint64_t>> sequence_expert_counts; // layer -> expert -> count
    uint64_t total_tokens_observed = 0;

    // Tracking for accuracy and reuse distance
    uint64_t total_predictions = 0;
    uint64_t correct_predictions = 0;
    std::unordered_map<ExpertKey, uint64_t, KeyHash> last_observed_token;
    uint64_t total_reuse_distance = 0;
    uint64_t reuse_distance_samples = 0;
};

class Prefetcher {
public:
    explicit Prefetcher(const Config & cfg, ExpertMemoryManager & mm);

    void prefetch(int layer, const std::vector<int> & experts, TokenTimeline * cur_tl = nullptr);
    void prefetch_expert_to_tier(int layer, int expert, MemoryTier target_tier, TokenTimeline * cur_tl = nullptr);
    bool has_prefetched(int layer, int expert) const;
    bool consume_prefetch(int layer, int expert);
    void clear_prefetched();

private:
    Config cfg;
    ExpertMemoryManager & mm;
    std::unordered_set<ExpertKey, KeyHash> prefetched_in_flight;
};

// ---------------------------------------------------------------------------
// TokenExpertCorrelator — tracks empirical token_id -> expert correlations
// ---------------------------------------------------------------------------
class TokenExpertCorrelator {
public:
    void observe(int token_id, int layer, const std::vector<int> & actual);
    std::vector<std::pair<int, double>> get_expert_scores(int token_id, int layer) const;

private:
    struct TokenLayerKey {
        int token_id;
        int layer;
        bool operator==(const TokenLayerKey & o) const {
            return token_id == o.token_id && layer == o.layer;
        }
    };
    struct TLKeyHash {
        size_t operator()(const TokenLayerKey & k) const {
            return (size_t(uint32_t(k.token_id)) << 32) ^ uint32_t(k.layer);
        }
    };
    std::unordered_map<TokenLayerKey, std::unordered_map<int, uint32_t>, TLKeyHash> stats;
};

// ---------------------------------------------------------------------------
// Multi-Token Working Set (MTWS) Structures & Planner
// ---------------------------------------------------------------------------
struct UniqueExpertCandidate {
    int layer = -1;
    int expert = -1;
    double score = 0.0;
    MemoryTier target_tier = MemoryTier::NVME;
    size_t size_bytes = 3481600;
    int token_occurrence_count = 0;
    bool prefetch_launched = false;
    clock::time_point prefetch_time{};
};

struct WorkingSetPlan {
    std::vector<UniqueExpertCandidate> unique_experts;
    std::unordered_map<ExpertKey, size_t, KeyHash> expert_indices;
    size_t vram_plan_bytes = 0;
    size_t ram_plan_bytes = 0;
    size_t nvme_plan_bytes = 0;
    clock::time_point prefetch_start{};
    clock::time_point prefetch_end{};
    bool prefetch_done = false;
};

class MTWSPlanner {
public:
    MTWSPlanner(const Config & cfg, ExpertMemoryManager & mm, Predictor & pred, Prefetcher & pf);

    void compute_budgets(size_t & out_vram_mb, size_t & out_ram_mb, const std::string & mtp_loc);

    WorkingSetPlan build_working_set(const std::vector<llama_token> & draft_tokens,
                                     const TokenExpertCorrelator & correlator,
                                     const std::vector<int> * last_layer_experts);

    void prefetch_plan(WorkingSetPlan & plan, TokenTimeline * cur_tl = nullptr);

    void record_demand(const WorkingSetPlan & plan, int layer, int expert, bool is_first_use);

private:
    Config cfg;
    ExpertMemoryManager & mm;
    Predictor & predictor;
    Prefetcher & prefetcher;
};

struct Runtime {
    Config cfg;
    ExpertMemoryManager memory_manager;
    Predictor predictor;
    Prefetcher prefetcher;
    TokenExpertCorrelator token_correlator;
    MTWSPlanner mtws_planner;
    WorkingSetPlan current_plan;

    void * page_prefetcher = nullptr;
    llama_model * model = nullptr;
    int expected_top_k = 10;
    int last_layer = -1;
    bool token_active = false;
    bool saw_decode = false;
    FILE * trace_file = nullptr;

    // Phase 3: Sampled readback counter
    int tokens_decoded = 0;  // Monotonic count of decode tokens; drives readback sampling schedule

    std::vector<int> last_layer_experts[48];
    struct ggml_tensor * last_topk_tensor[48] = {};
    std::vector<llama_token> current_batch_tokens;
    TokenTimeline current_timeline{};

    // Adaptive policy tracking
    int current_draft_n = 3;
    int streak_low_acceptance = 0;
    double rolling_acceptance = 0.50;

    // Quality & Self-Learning (Phases 10 & 11)
    QualityEstimator quality_estimator;
    SelfLearningLocality learning_locality;

    // Staging & Cost Model (Sections 3 & 5)
    PinnedHostStagingPool host_staging_pool;
    ExpertCostModel cost_model;

    // Grouped-GEMM & Concurrent Streams Pipeline (Sections 2 & 8)
    GroupedGemmStreamPipeline grouped_pipeline;
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
};

} // namespace atlas
