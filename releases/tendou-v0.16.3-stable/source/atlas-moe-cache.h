#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <dlfcn.h>
#  include <sys/mman.h>
#  include <unistd.h>
#endif

namespace atlas {

// ---------------------------------------------------------------------------
// Phase 1.3 (PR #31233): O(1) ClassLRU with Exponential CRF Decay
// Shared across Tier-0 CUDA VRAM Arena and Tier-1 Locked DDR5 RAM Arena.
// ---------------------------------------------------------------------------
class ClassLRUPolicy {
public:
    static constexpr int NUM_CLASSES = 8;

    struct SlotNode {
        int      layer              = -1;
        int      expert             = -1;
        float    crf_score          = 0.0f;
        uint64_t last_step          = 0;
        int8_t   cls                = -1;
        int      prev               = -1;
        int      next               = -1;
        bool     occupied           = false;
        bool     in_flight_reserved = false;
    };

    ClassLRUPolicy() = default;

    void init(int max_slots, float half_life_tokens = 128.0f) {
        max_slots_ = std::max(0, max_slots);
        half_life_ = std::max(1.0f, half_life_tokens);
        nodes_.assign(size_t(max_slots_), SlotNode{});
        for (int c = 0; c < NUM_CLASSES; ++c) {
            head_[c] = -1;
            tail_[c] = -1;
            counts_[c] = 0;
        }
        used_count_ = 0;
        reserved_count_ = 0;
        free_stack_.clear();
        free_stack_.reserve(size_t(max_slots_));
        for (int s = max_slots_ - 1; s >= 0; --s) {
            free_stack_.push_back(s);
        }
        key_to_slot_.clear();
        reserved_key_to_slot_.clear();
    }

    int max_slots() const { return max_slots_; }
    int used_slots() const { return used_count_; }
    int reserved_slots() const { return reserved_count_; }
    int free_slots() const { return int(free_stack_.size()); }

    bool is_reserved_key(int layer, int expert) const {
        return reserved_key_to_slot_.find(pack_key(layer, expert)) != reserved_key_to_slot_.end();
    }

    bool is_slot_reserved(int slot) const {
        if (slot < 0 || slot >= max_slots_) return false;
        return nodes_[size_t(slot)].in_flight_reserved;
    }

    int lookup_slot(int layer, int expert) const {
        auto it = key_to_slot_.find(pack_key(layer, expert));
        return (it != key_to_slot_.end()) ? it->second : -1;
    }

    const SlotNode * get_node(int slot) const {
        if (slot < 0 || slot >= max_slots_) return nullptr;
        return &nodes_[size_t(slot)];
    }

    // Update CRF score on hit: S = boost + S_prev * 2^(-(step - last_step) / half_life)
    void touch(int slot, uint64_t step, float layer_boost = 1.0f) {
        if (slot < 0 || slot >= max_slots_) return;
        SlotNode & n = nodes_[size_t(slot)];
        if (!n.occupied) return;

        const uint64_t dt = (step >= n.last_step) ? (step - n.last_step) : 0ULL;
        const float decay = (dt == 0) ? 1.0f : std::exp2f(-float(dt) / half_life_);
        n.crf_score = std::max(1.0f, layer_boost + n.crf_score * decay);
        n.last_step = step;

        const int8_t new_cls = score_to_class(n.crf_score);
        unlink(slot);
        link_front(slot, new_cls);
    }

    // Acquire a free slot and admit (layer, expert). Returns slot index or -1 if full.
    int admit(int layer, int expert, uint64_t step, float layer_boost = 1.0f) {
        if (is_reserved_key(layer, expert)) return -1;
        const int existing = lookup_slot(layer, expert);
        if (existing >= 0) {
            touch(existing, step, layer_boost);
            return existing;
        }
        if (free_stack_.empty()) return -1;
        const int slot = free_stack_.back();
        free_stack_.pop_back();

        SlotNode & n = nodes_[size_t(slot)];
        n.layer              = layer;
        n.expert             = expert;
        n.crf_score          = std::max(1.0f, layer_boost);
        n.last_step          = step;
        n.occupied           = true;
        n.in_flight_reserved = false;
        n.prev               = -1;
        n.next               = -1;
        n.cls                = -1;

        key_to_slot_[pack_key(layer, expert)] = slot;
        used_count_++;
        link_front(slot, score_to_class(n.crf_score));
        return slot;
    }

    // Reserve a free slot for in-flight async DMA without linking into eviction lists or key_to_slot_.
    int reserve_free_slot(int layer, int expert) {
        const uint64_t key = pack_key(layer, expert);
        if (key_to_slot_.find(key) != key_to_slot_.end()) return -1;
        if (reserved_key_to_slot_.find(key) != reserved_key_to_slot_.end()) return -1;
        if (free_stack_.empty()) return -1;

        const int slot = free_stack_.back();
        free_stack_.pop_back();

        SlotNode & n = nodes_[size_t(slot)];
        n.layer              = layer;
        n.expert             = expert;
        n.crf_score          = 0.0f;
        n.last_step          = 0;
        n.occupied           = false;
        n.in_flight_reserved = true;
        n.prev               = -1;
        n.next               = -1;
        n.cls                = -1;

        reserved_key_to_slot_[key] = slot;
        reserved_count_++;
        return slot;
    }

    // Finalize an in-flight reserved slot into an occupied ClassLRU slot.
    bool commit_reserved_slot(int slot, int layer, int expert, uint64_t step, float layer_boost = 1.0f) {
        if (slot < 0 || slot >= max_slots_) return false;
        SlotNode & n = nodes_[size_t(slot)];
        if (!n.in_flight_reserved || n.occupied) return false;
        if (n.layer != layer || n.expert != expert) return false;

        const uint64_t key = pack_key(layer, expert);
        if (key_to_slot_.find(key) != key_to_slot_.end()) return false;

        reserved_key_to_slot_.erase(key);
        if (reserved_count_ > 0) reserved_count_--;

        n.in_flight_reserved = false;
        n.occupied           = true;
        n.crf_score          = std::max(1.0f, layer_boost);
        n.last_step          = step;
        n.prev               = -1;
        n.next               = -1;
        n.cls                = -1;

        key_to_slot_[key] = slot;
        used_count_++;
        link_front(slot, score_to_class(n.crf_score));
        return true;
    }

    // Abort an in-flight reserved slot and return it to the free stack cleanly.
    bool abort_reserved_slot(int slot) {
        if (slot < 0 || slot >= max_slots_) return false;
        SlotNode & n = nodes_[size_t(slot)];
        if (!n.in_flight_reserved || n.occupied) return false;

        reserved_key_to_slot_.erase(pack_key(n.layer, n.expert));
        if (reserved_count_ > 0) reserved_count_--;

        n.in_flight_reserved = false;
        n.occupied           = false;
        n.layer              = -1;
        n.expert             = -1;
        n.crf_score          = 0.0f;
        n.last_step          = 0;
        n.prev               = -1;
        n.next               = -1;
        n.cls                = -1;

        free_stack_.push_back(slot);
        return true;
    }

    // Evict lowest-class LRU slot that is not protected or in-flight reserved. Returns evicted slot id, or -1.
    int evict_victim(const std::function<bool(int, int)> & is_protected = nullptr,
                     int * out_layer = nullptr,
                     int * out_expert = nullptr) {
        for (int c = 0; c < NUM_CLASSES; ++c) {
            int cur = tail_[c];
            int guard = counts_[c] + 1;
            while (cur >= 0 && guard-- > 0) {
                const int prev = nodes_[size_t(cur)].prev;
                const SlotNode & n = nodes_[size_t(cur)];
                if (!n.in_flight_reserved && (!is_protected || !is_protected(n.layer, n.expert))) {
                    if (out_layer)  *out_layer  = n.layer;
                    if (out_expert) *out_expert = n.expert;
                    release(cur);
                    return cur;
                }
                cur = prev;
            }
        }
        return -1;
    }

    bool release(int slot) {
        if (slot < 0 || slot >= max_slots_) return false;
        SlotNode & n = nodes_[size_t(slot)];
        if (!n.occupied || n.in_flight_reserved) return false;
        key_to_slot_.erase(pack_key(n.layer, n.expert));
        unlink(slot);
        n.occupied  = false;
        n.layer     = -1;
        n.expert    = -1;
        n.crf_score = 0.0f;
        free_stack_.push_back(slot);
        if (used_count_ > 0) used_count_--;
        return true;
    }

    void clear() {
        init(max_slots_, half_life_);
    }

    int class_count(int c) const {
        return (c >= 0 && c < NUM_CLASSES) ? counts_[c] : 0;
    }

private:
    int max_slots_ = 0;
    float half_life_ = 128.0f;
    int used_count_ = 0;
    int reserved_count_ = 0;
    std::vector<SlotNode> nodes_;
    std::vector<int> free_stack_;
    std::unordered_map<uint64_t, int> key_to_slot_;
    std::unordered_map<uint64_t, int> reserved_key_to_slot_;
    int head_[NUM_CLASSES]   = {};
    int tail_[NUM_CLASSES]   = {};
    int counts_[NUM_CLASSES] = {};

    static inline uint64_t pack_key(int layer, int expert) {
        return (uint64_t(uint32_t(layer)) << 32) | uint64_t(uint32_t(expert));
    }

    static inline int8_t score_to_class(float score) {
        uint32_t s = (score >= 1.0f) ? uint32_t(score) : 1u;
        int8_t cls = 0;
        while (s >>= 1) {
            if (++cls >= NUM_CLASSES - 1) return NUM_CLASSES - 1;
        }
        return cls;
    }

    void unlink(int slot) {
        SlotNode & n = nodes_[size_t(slot)];
        const int8_t c = n.cls;
        if (c < 0 || c >= NUM_CLASSES) return;
        if (n.prev >= 0) nodes_[size_t(n.prev)].next = n.next;
        else             head_[c] = n.next;
        if (n.next >= 0) nodes_[size_t(n.next)].prev = n.prev;
        else             tail_[c] = n.prev;
        n.prev = -1;
        n.next = -1;
        n.cls  = -1;
        if (counts_[c] > 0) counts_[c]--;
    }

    void link_front(int slot, int8_t c) {
        if (c < 0) c = 0;
        if (c >= NUM_CLASSES) c = NUM_CLASSES - 1;
        SlotNode & n = nodes_[size_t(slot)];
        n.cls  = c;
        n.prev = -1;
        n.next = head_[c];
        if (head_[c] >= 0) nodes_[size_t(head_[c])].prev = slot;
        else               tail_[c] = slot;
        head_[c] = slot;
        counts_[c]++;
    }
};

// ---------------------------------------------------------------------------
// Phase 0.2: Real CUDA Driver Device Arena & Zero-Copy Mapped Host Helper
// Uses libcuda.so.1 / nvcuda.dll dynamically so it works in all binaries.
// ---------------------------------------------------------------------------
class CudaDriverMemBridge {
public:
    struct Api {
        bool initialized = false;
        bool available   = false;
#if defined(_WIN32)
        using cuInit_t                = int (__stdcall *)(unsigned int);
        using cuRetainCtx_t           = int (__stdcall *)(void **, int);
        using cuSetCtx_t              = int (__stdcall *)(void *);
        using cuMemAlloc_t            = int (__stdcall *)(uint64_t *, size_t);
        using cuMemFree_t             = int (__stdcall *)(uint64_t);
        using cuMemHostAlloc_t        = int (__stdcall *)(void **, size_t, unsigned int);
        using cuMemFreeHost_t         = int (__stdcall *)(void *);
        using cuMemHostGetDevPtr_t    = int (__stdcall *)(uint64_t *, void *, unsigned int);
        using cuMemcpyHtoDAsync_t     = int (__stdcall *)(uint64_t, const void *, size_t, void *);
        using cuMemcpyDtoHAsync_t     = int (__stdcall *)(void *, uint64_t, size_t, void *);
        using cuStreamCreate_t        = int (__stdcall *)(void **, unsigned int);
        using cuStreamDestroy_t       = int (__stdcall *)(void *);
        using cuStreamSync_t          = int (__stdcall *)(void *);
        using cuEventCreate_t         = int (__stdcall *)(void **, unsigned int);
        using cuEventDestroy_t        = int (__stdcall *)(void *);
        using cuEventRecord_t         = int (__stdcall *)(void *, void *);
        using cuEventQuery_t          = int (__stdcall *)(void *);
        using cuEventSync_t           = int (__stdcall *)(void *);
#else
        using cuInit_t                = int (*)(unsigned int);
        using cuRetainCtx_t           = int (*)(void **, int);
        using cuSetCtx_t              = int (*)(void *);
        using cuMemAlloc_t            = int (*)(uint64_t *, size_t);
        using cuMemFree_t             = int (*)(uint64_t);
        using cuMemHostAlloc_t        = int (*)(void **, size_t, unsigned int);
        using cuMemFreeHost_t         = int (*)(void *);
        using cuMemHostGetDevPtr_t    = int (*)(uint64_t *, void *, unsigned int);
        using cuMemcpyHtoDAsync_t     = int (*)(uint64_t, const void *, size_t, void *);
        using cuMemcpyDtoHAsync_t     = int (*)(void *, uint64_t, size_t, void *);
        using cuStreamCreate_t        = int (*)(void **, unsigned int);
        using cuStreamDestroy_t       = int (*)(void *);
        using cuStreamSync_t          = int (*)(void *);
        using cuEventCreate_t         = int (*)(void **, unsigned int);
        using cuEventDestroy_t        = int (*)(void *);
        using cuEventRecord_t         = int (*)(void *, void *);
        using cuEventQuery_t          = int (*)(void *);
        using cuEventSync_t           = int (*)(void *);
#endif
        cuInit_t             fnInit           = nullptr;
        cuRetainCtx_t        fnRetainCtx      = nullptr;
        cuSetCtx_t           fnSetCtx         = nullptr;
        cuMemAlloc_t         fnMemAlloc       = nullptr;
        cuMemFree_t          fnMemFree        = nullptr;
        cuMemHostAlloc_t     fnMemHostAlloc   = nullptr;
        cuMemFreeHost_t      fnMemFreeHost    = nullptr;
        cuMemHostGetDevPtr_t fnHostGetDevPtr  = nullptr;
        cuMemcpyHtoDAsync_t  fnHtoDAsync      = nullptr;
        cuMemcpyDtoHAsync_t  fnDtoHAsync      = nullptr;
        cuStreamCreate_t     fnStreamCreate   = nullptr;
        cuStreamDestroy_t    fnStreamDestroy  = nullptr;
        cuStreamSync_t       fnStreamSync     = nullptr;
        cuEventCreate_t      fnEventCreate    = nullptr;
        cuEventDestroy_t     fnEventDestroy   = nullptr;
        cuEventRecord_t      fnEventRecord    = nullptr;
        cuEventQuery_t       fnEventQuery     = nullptr;
        cuEventSync_t        fnEventSync      = nullptr;
    };

    static Api & api() {
        static Api s_api = load_api();
        return s_api;
    }

    static bool ensure_context() {
        Api & a = api();
        if (!a.available) return false;
        if (a.fnInit) a.fnInit(0);
        if (a.fnRetainCtx && a.fnSetCtx) {
            void * ctx = nullptr;
            if (a.fnRetainCtx(&ctx, 0) == 0 && ctx) {
                a.fnSetCtx(ctx);
                return true;
            }
        }
        return false;
    }

    static uint64_t alloc_device(size_t bytes) {
        if (bytes == 0 || !ensure_context()) return 0;
        uint64_t dev_ptr = 0;
        if (api().fnMemAlloc(&dev_ptr, bytes) == 0) return dev_ptr;
        return 0;
    }

    static void free_device(uint64_t dev_ptr) {
        if (!dev_ptr || !ensure_context()) return;
        api().fnMemFree(dev_ptr);
    }

    static bool alloc_mapped_host(size_t bytes, void ** out_host, uint64_t * out_dev) {
        if (!out_host || !out_dev || bytes == 0 || !ensure_context()) return false;
        // CU_MEMHOSTALLOC_PORTABLE (0x01) | CU_MEMHOSTALLOC_DEVICEMAP (0x02)
        void * host_ptr = nullptr;
        if (api().fnMemHostAlloc(&host_ptr, bytes, 0x03u) != 0 || !host_ptr) {
            return false;
        }
        uint64_t dev_ptr = 0;
        if (api().fnHostGetDevPtr(&dev_ptr, host_ptr, 0) != 0 || !dev_ptr) {
            api().fnMemFreeHost(host_ptr);
            return false;
        }
        std::memset(host_ptr, 0, bytes);
        *out_host = host_ptr;
        *out_dev  = dev_ptr;
        return true;
    }

    static void free_mapped_host(void * host_ptr) {
        if (!host_ptr || !ensure_context()) return;
        api().fnMemFreeHost(host_ptr);
    }

    static bool copy_h2d_async(uint64_t dst_dev, const void * src_host, size_t bytes, void * stream = nullptr) {
        if (!dst_dev || !src_host || bytes == 0 || !ensure_context()) return false;
        return api().fnHtoDAsync(dst_dev, src_host, bytes, stream) == 0;
    }

    static bool copy_d2h_async(void * dst_host, uint64_t src_dev, size_t bytes, void * stream = nullptr) {
        if (!dst_host || !src_dev || bytes == 0 || !ensure_context()) return false;
        return api().fnDtoHAsync(dst_host, src_dev, bytes, stream) == 0;
    }

    static void * create_stream(bool non_blocking = true) {
        if (!ensure_context() || !api().fnStreamCreate) return nullptr;
        void * s = nullptr;
        if (api().fnStreamCreate(&s, non_blocking ? 1u : 0u) == 0) return s;
        return nullptr;
    }

    static void destroy_stream(void * s) {
        if (s && ensure_context() && api().fnStreamDestroy) api().fnStreamDestroy(s);
    }

    static bool sync_stream(void * s) {
        if (!ensure_context() || !api().fnStreamSync) return false;
        return api().fnStreamSync(s) == 0;
    }

private:
    static Api load_api() {
        Api a;
        a.initialized = true;
#if defined(_WIN32)
        HMODULE h = GetModuleHandleA("nvcuda.dll");
        if (!h) h = LoadLibraryA("nvcuda.dll");
        if (!h) return a;
        auto sym = [&](const char * name) -> void * { return (void *)GetProcAddress(h, name); };
#else
        void * h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
        if (!h) h = dlopen("libcuda.so.1", RTLD_LAZY);
        if (!h) return a;
        auto sym = [&](const char * name) -> void * { return dlsym(h, name); };
#endif
        a.fnInit          = (Api::cuInit_t)sym("cuInit");
        a.fnRetainCtx     = (Api::cuRetainCtx_t)sym("cuDevicePrimaryCtxRetain");
        a.fnSetCtx        = (Api::cuSetCtx_t)sym("cuCtxSetCurrent");
        a.fnMemAlloc      = (Api::cuMemAlloc_t)sym("cuMemAlloc_v2");
        if (!a.fnMemAlloc) a.fnMemAlloc = (Api::cuMemAlloc_t)sym("cuMemAlloc");
        a.fnMemFree       = (Api::cuMemFree_t)sym("cuMemFree_v2");
        if (!a.fnMemFree) a.fnMemFree = (Api::cuMemFree_t)sym("cuMemFree");
        a.fnMemHostAlloc  = (Api::cuMemHostAlloc_t)sym("cuMemHostAlloc");
        a.fnMemFreeHost   = (Api::cuMemFreeHost_t)sym("cuMemFreeHost");
        a.fnHostGetDevPtr = (Api::cuMemHostGetDevPtr_t)sym("cuMemHostGetDevicePointer_v2");
        if (!a.fnHostGetDevPtr) a.fnHostGetDevPtr = (Api::cuMemHostGetDevPtr_t)sym("cuMemHostGetDevicePointer");
        a.fnHtoDAsync     = (Api::cuMemcpyHtoDAsync_t)sym("cuMemcpyHtoDAsync_v2");
        if (!a.fnHtoDAsync) a.fnHtoDAsync = (Api::cuMemcpyHtoDAsync_t)sym("cuMemcpyHtoDAsync");
        a.fnDtoHAsync     = (Api::cuMemcpyDtoHAsync_t)sym("cuMemcpyDtoHAsync_v2");
        if (!a.fnDtoHAsync) a.fnDtoHAsync = (Api::cuMemcpyDtoHAsync_t)sym("cuMemcpyDtoHAsync");
        a.fnStreamCreate  = (Api::cuStreamCreate_t)sym("cuStreamCreate");
        a.fnStreamDestroy = (Api::cuStreamDestroy_t)sym("cuStreamDestroy_v2");
        if (!a.fnStreamDestroy) a.fnStreamDestroy = (Api::cuStreamDestroy_t)sym("cuStreamDestroy");
        a.fnStreamSync    = (Api::cuStreamSync_t)sym("cuStreamSynchronize");
        a.fnEventCreate   = (Api::cuEventCreate_t)sym("cuEventCreate");
        a.fnEventDestroy  = (Api::cuEventDestroy_t)sym("cuEventDestroy_v2");
        a.fnEventRecord   = (Api::cuEventRecord_t)sym("cuEventRecord");
        a.fnEventQuery    = (Api::cuEventQuery_t)sym("cuEventQuery");
        a.fnEventSync     = (Api::cuEventSync_t)sym("cuEventSynchronize");
        a.available = (a.fnMemAlloc && a.fnMemFree && a.fnHtoDAsync && a.fnDtoHAsync);
        return a;
    }
};

// ---------------------------------------------------------------------------
// Phase 1.2 (PR #32076): Lockless Zero-Copy Mapped Host Router Mailbox
// Avoids per-layer cudaDeviceSynchronize / ggml_backend_tensor_get stalls.
// ---------------------------------------------------------------------------
struct alignas(64) LayerRouterMailbox {
    uint64_t seq_ready;         // GPU writes monotonic seq after routing + D2H copy
    uint64_t gpu_hit_mask;      // Bit i = 1 if top-k slot i executed on GPU
    uint64_t cpu_miss_mask;     // Bit i = 1 if top-k slot i must execute on CPU
    int32_t  layer;
    int32_t  n_tokens;
    int32_t  k;
    int32_t  reserved;
    int32_t  expert_ids[64];
    float    expert_weights[64];
};

class MappedRouterMailboxRing {
public:
    static constexpr int MAX_LAYERS = 64;

    MappedRouterMailboxRing() = default;
    ~MappedRouterMailboxRing() { release(); }

    MappedRouterMailboxRing(const MappedRouterMailboxRing &) = delete;
    MappedRouterMailboxRing & operator=(const MappedRouterMailboxRing &) = delete;

    bool init() {
        if (host_slots_) return true;
        const size_t bytes = size_t(MAX_LAYERS) * sizeof(LayerRouterMailbox);
        void * h = nullptr;
        uint64_t d = 0;
        if (CudaDriverMemBridge::alloc_mapped_host(bytes, &h, &d)) {
            host_slots_ = static_cast<LayerRouterMailbox *>(h);
            dev_slots_  = d;
            is_cuda_mapped_ = true;
            return true;
        }
        fallback_slots_.assign(size_t(MAX_LAYERS), LayerRouterMailbox{});
        host_slots_ = fallback_slots_.data();
        dev_slots_  = 0;
        is_cuda_mapped_ = false;
        return true;
    }

    void release() {
        if (is_cuda_mapped_ && host_slots_) {
            CudaDriverMemBridge::free_mapped_host(host_slots_);
        }
        host_slots_ = nullptr;
        dev_slots_  = 0;
        is_cuda_mapped_ = false;
        fallback_slots_.clear();
    }

    void publish_cpu_snapshot(int layer, uint64_t seq, uint64_t gpu_mask, uint64_t cpu_mask,
                              int n_tokens, int k, const int32_t * ids, const float * weights) {
        if (!host_slots_ || layer < 0 || layer >= MAX_LAYERS) return;
        LayerRouterMailbox & m = host_slots_[layer];
        m.layer         = layer;
        m.n_tokens      = n_tokens;
        m.k             = k;
        m.gpu_hit_mask  = gpu_mask;
        m.cpu_miss_mask = cpu_mask;
        const int count = std::min(64, std::max(0, n_tokens * k));
        for (int i = 0; i < count; ++i) {
            m.expert_ids[i]     = ids ? ids[i] : -1;
            m.expert_weights[i] = weights ? weights[i] : 0.0f;
        }
        std::atomic_thread_fence(std::memory_order_release);
        reinterpret_cast<std::atomic<uint64_t> *>(&m.seq_ready)->store(seq, std::memory_order_release);
    }

    bool poll_layer_ready(int layer, uint64_t expected_seq, LayerRouterMailbox & out, int max_spins = 20000) const {
        if (!host_slots_ || layer < 0 || layer >= MAX_LAYERS) return false;
        const auto * seq_ptr = reinterpret_cast<const std::atomic<uint64_t> *>(&host_slots_[layer].seq_ready);
        for (int s = 0; s < max_spins; ++s) {
            if (seq_ptr->load(std::memory_order_acquire) >= expected_seq) {
                std::atomic_thread_fence(std::memory_order_acquire);
                out = host_slots_[layer];
                return true;
            }
            if ((s & 63) == 63) std::this_thread::yield();
        }
        return false;
    }

    LayerRouterMailbox * host_ptr() const { return host_slots_; }
    uint64_t device_ptr() const { return dev_slots_; }
    bool is_cuda_mapped() const { return is_cuda_mapped_; }

private:
    LayerRouterMailbox * host_slots_ = nullptr;
    uint64_t dev_slots_ = 0;
    bool is_cuda_mapped_ = false;
    std::vector<LayerRouterMailbox> fallback_slots_;
};

// ---------------------------------------------------------------------------
// Phase 1.1 (PR #30966 Evolution): Closed-Loop Adaptive CPU Share Controller
// Balances GPU hit + upload work vs CPU AVX-512 miss work without stalls.
// ---------------------------------------------------------------------------
struct AdaptiveCpuShareDecision {
    double   effective_cpu_share = 1.0;
    uint64_t cpu_miss_mask       = 0;
    uint64_t gpu_upload_mask     = 0;
    int      cpu_count           = 0;
    int      gpu_upload_count    = 0;
};

class AdaptiveCpuShareController {
public:
    // user_share < 0 => auto closed-loop mode; [0.0, 1.0] => fixed user share
    explicit AdaptiveCpuShareController(double user_share = -1.0,
                                        double cpu_expert_ms = 0.45,
                                        double gpu_upload_expert_ms = 0.32,
                                        double gpu_dense_overlap_ms = 0.90)
        : user_share_(user_share),
          cpu_expert_ms_(std::max(0.02, cpu_expert_ms)),
          gpu_upload_expert_ms_(std::max(0.02, gpu_upload_expert_ms)),
          gpu_dense_overlap_ms_(std::max(0.05, gpu_dense_overlap_ms)) {}

    void set_user_share(double s) { user_share_ = s; }
    double get_user_share() const { return user_share_; }

    void update_observed_timings(double cpu_ms_per_exp, double gpu_overlap_ms) {
        if (cpu_ms_per_exp > 0.01 && cpu_ms_per_exp < 50.0) {
            cpu_expert_ms_ = 0.85 * cpu_expert_ms_ + 0.15 * cpu_ms_per_exp;
        }
        if (gpu_overlap_ms > 0.01 && gpu_overlap_ms < 50.0) {
            gpu_dense_overlap_ms_ = 0.85 * gpu_dense_overlap_ms_ + 0.15 * gpu_overlap_ms;
        }
    }

    AdaptiveCpuShareDecision plan_misses(int k,
                                         uint64_t gpu_hit_mask,
                                         int free_or_evictable_vram_slots,
                                         bool allow_async_gpu_upload) const {
        AdaptiveCpuShareDecision out;
        int miss_indices[64];
        int n_miss = 0;
        const int clamped_k = std::min(64, std::max(0, k));
        for (int i = 0; i < clamped_k; ++i) {
            if (((gpu_hit_mask >> i) & 1ULL) == 0ULL) {
                miss_indices[n_miss++] = i;
            }
        }
        if (n_miss == 0) {
            out.effective_cpu_share = 0.0;
            return out;
        }

        double share = 1.0;
        if (user_share_ >= 0.0) {
            share = std::min(1.0, std::max(0.0, user_share_));
        } else if (allow_async_gpu_upload && free_or_evictable_vram_slots > 0) {
            // How many misses can CPU finish completely hidden behind GPU dense+hits?
            const int hidden_cpu_cap = std::max(1, int(std::floor(gpu_dense_overlap_ms_ / cpu_expert_ms_)));
            if (n_miss <= hidden_cpu_cap) {
                share = 1.0;
            } else {
                // Split excess misses proportionally between CPU time and GPU H2D+compute time
                const int optimal_cpu = std::max(hidden_cpu_cap,
                    int(std::round((double(n_miss) * gpu_upload_expert_ms_) / (cpu_expert_ms_ + gpu_upload_expert_ms_))));
                share = std::min(1.0, double(optimal_cpu) / double(n_miss));
            }
        }

        int n_cpu = int(std::round(share * double(n_miss)));
        if (!allow_async_gpu_upload || free_or_evictable_vram_slots <= 0) {
            n_cpu = n_miss;
        }
        n_cpu = std::max(0, std::min(n_miss, n_cpu));
        const int n_gpu_up = std::min(n_miss - n_cpu, std::max(0, free_or_evictable_vram_slots));
        n_cpu = n_miss - n_gpu_up; // Ensure every miss is assigned!

        for (int idx = 0; idx < n_miss; ++idx) {
            const int bit = miss_indices[idx];
            if (idx < n_gpu_up) {
                out.gpu_upload_mask |= (1ULL << bit);
                out.gpu_upload_count++;
            } else {
                out.cpu_miss_mask |= (1ULL << bit);
                out.cpu_count++;
            }
        }
        out.effective_cpu_share = double(out.cpu_count) / double(n_miss);
        return out;
    }

private:
    double user_share_            = -1.0;
    double cpu_expert_ms_         = 0.45;
    double gpu_upload_expert_ms_  = 0.32;
    double gpu_dense_overlap_ms_  = 0.90;
};

// ---------------------------------------------------------------------------
// Phase 18 WP1.2: Layer-Concentrated VRAM Pinning & Adaptive RAM Lock Policy
// ---------------------------------------------------------------------------
inline uint64_t compute_recommended_vram_layer_mask(int n_layers, int vram_slots, int layer_cap = 0) {
    if (vram_slots <= 0) return 0ULL;
    n_layers = std::max(1, std::min(48, n_layers));
    const int cap = (layer_cap > 0)
        ? std::min(layer_cap, n_layers)
        : std::max(1, std::min(n_layers, vram_slots / 64));
    if (cap >= n_layers) {
        return (n_layers >= 64) ? ~0ULL : ((1ULL << n_layers) - 1ULL);
    }
    const int front = cap / 2;
    const int back  = cap - front;
    uint64_t mask = 0ULL;
    for (int i = 0; i < front; ++i) {
        mask |= (1ULL << i);
    }
    for (int i = n_layers - back; i < n_layers; ++i) {
        if (i >= 0 && i < 64) {
            mask |= (1ULL << i);
        }
    }
    return mask;
}

inline size_t compute_adaptive_ram_budget(size_t requested_ram_bytes,
                                          size_t total_os_ram_bytes,
                                          size_t avail_os_ram_bytes,
                                          size_t model_file_bytes,
                                          int active_k,
                                          size_t prompt_cache_bytes,
                                          const std::string & ram_lock_policy = "auto") {
    if (ram_lock_policy == "off" || requested_ram_bytes == 0) {
        return 0;
    }
    const size_t base_guard = size_t(2560ULL) * 1024ULL * 1024ULL;
    const size_t ram_guard = (prompt_cache_bytes > (SIZE_MAX - base_guard))
        ? SIZE_MAX
        : (base_guard + prompt_cache_bytes);
    const size_t safe_ram = (avail_os_ram_bytes > ram_guard)
        ? (avail_os_ram_bytes - ram_guard)
        : 0;
    if (ram_lock_policy == "on") {
        return std::min(requested_ram_bytes, safe_ram);
    }
    // "auto" (default)
    constexpr size_t k17GiB  = 17ULL * 1024ULL * 1024ULL * 1024ULL;
    constexpr size_t k8GiB   = 8ULL  * 1024ULL * 1024ULL * 1024ULL;
    constexpr size_t k512MiB = 512ULL * 1024ULL * 1024ULL;
    if (total_os_ram_bytes > 0 &&
        total_os_ram_bytes <= k17GiB &&
        model_file_bytes > 2ULL * total_os_ram_bytes &&
        active_k > 0 && active_k <= 5) {
        return 0;
    }
    if (avail_os_ram_bytes < k8GiB) {
        return std::min(std::min(requested_ram_bytes, safe_ram), k512MiB);
    }
    return std::min(requested_ram_bytes, safe_ram);
}

// ---------------------------------------------------------------------------
// v0.16.3: Zero-Copy Boost-Aware Hot-RAM THP Pinning Planner
// Maximizes Host RAM residency for CPU MoE tensors directly in mmap without
// duplicating bytes into a separate heap copy or causing OS swap/OOM.
// ---------------------------------------------------------------------------
struct ZeroCopyHotRamLayer {
    int    layer = -1;
    size_t bytes = 0;
};

struct ZeroCopyHotRamPlan {
    size_t           budget_bytes             = 0;
    size_t           guard_bytes              = 0;
    size_t           planned_bytes            = 0;
    size_t           total_cpu_moe_bytes      = 0;
    bool             streaming_pagecache_mode = false;
    bool             use_mlock                = false;
    std::vector<int> selected_layers;
};

inline int zero_copy_layer_priority(int layer, int n_layers) {
    if (layer < 2) return 3;                                // Anchor layers (K=4)
    if (layer >= 15 && layer <= 30) return 3;               // Core reasoning band (K=2..4)
    if (n_layers > 4 && layer >= n_layers - 4) return 3;    // Final output layers (K=2..4)
    if (layer > 30) return 2;                               // Upper mid layers
    return 1;                                               // Outer layers (2..14)
}

inline ZeroCopyHotRamPlan compute_zero_copy_hot_ram_plan(
        const std::vector<ZeroCopyHotRamLayer> & cpu_layers,
        int n_model_layers,
        size_t avail_os_ram_bytes,
        size_t prompt_cache_bytes,
        uint32_t n_ctx,
        const std::string & ram_lock_policy = "auto") {
    ZeroCopyHotRamPlan plan;
    for (const auto & cl : cpu_layers) {
        if (cl.layer >= 0 && cl.bytes > 0) {
            plan.total_cpu_moe_bytes += cl.bytes;
        }
    }

    // Three-regime Zero-Copy Hot-RAM Planner (v0.16.3):
    // 1. Full-Fit CPU MoE (total_cpu_moe <= avail_ram - safe_guard, e.g. 35B on >=24 GiB RAM):
    //    Zero unpinned layers remain, so zero-copy mlock pins 100% of CPU MoE weights safely.
    // 2. Near-Fit CPU MoE (safe_budget < total_cpu_moe <= 1.5x avail_ram, e.g. 35B ~11.6 GiB on 16 GiB RAM):
    //    Whole-layer mlock of ~9.5 GiB traps cold unused experts in unreclaimable RAM and starves the
    //    remaining unpinned layers (causing kswapd stalls & warm TPS regression). Instead, "auto" uses
    //    zero-copy-prewarm (MADV_HUGEPAGE on selected layers + MADV_WILLNEED + parallel page-touch) so
    //    ~9.5 GiB is hot in RAM at startup while Linux LRU can seamlessly evict unused cold experts to
    //    keep active experts from ALL layers resident during warm multi-turn decode.
    // 3. Heavy NVMe-Streaming CPU MoE (> 1.5x avail_ram, e.g. Qwen3.8 75.8 GB / 122B on 16 GiB RAM):
    //    Skip whole-layer prewarm/mlock/THP in "auto" mode; GPU-shadow page-cache reclamation dedicates
    //    all available Host RAM to dynamic Zipfian-hot experts across all layers.
    const bool heavy_nvme_streaming =
        (avail_os_ram_bytes > 0) &&
        (plan.total_cpu_moe_bytes > (avail_os_ram_bytes / 2ULL) * 3ULL);

    const size_t base_guard =
        (!heavy_nvme_streaming && avail_os_ram_bytes >= (size_t(10ULL) << 30))
            ? (size_t(1536ULL) * 1024ULL * 1024ULL)
            : (size_t(2048ULL) * 1024ULL * 1024ULL);
    const size_t ctx_extra  = (n_ctx >= 32768u) ? (size_t(512ULL) * 1024ULL * 1024ULL) : 0ULL;
    const size_t safe_pc    = std::min<size_t>(prompt_cache_bytes, size_t(2048ULL) * 1024ULL * 1024ULL);
    plan.guard_bytes        = base_guard + safe_pc + ctx_extra;

    if (ram_lock_policy == "off" || avail_os_ram_bytes <= plan.guard_bytes || cpu_layers.empty()) {
        return plan;
    }

    if (heavy_nvme_streaming && ram_lock_policy != "on") {
        plan.streaming_pagecache_mode = true;
        return plan;
    }

    size_t raw_budget = avail_os_ram_bytes - plan.guard_bytes;
    const size_t full_fit_guard = std::max<size_t>(plan.guard_bytes, size_t(2048ULL) * 1024ULL * 1024ULL + safe_pc + ctx_extra);
    const bool full_fit = (!heavy_nvme_streaming) &&
                          (avail_os_ram_bytes > full_fit_guard) &&
                          (plan.total_cpu_moe_bytes <= (avail_os_ram_bytes - full_fit_guard));

    if (heavy_nvme_streaming) {
        // Explicit --ram-lock-policy on on heavy streaming: cap whole-layer pinning to 25% of free RAM
        raw_budget = std::min<size_t>(raw_budget, avail_os_ram_bytes / 4ULL);
        plan.use_mlock = true;
    } else if (full_fit) {
        // All CPU MoE layers fit with safe OS headroom -> zero unpinned layers -> safe full zero-copy mlock
        plan.guard_bytes = full_fit_guard;
        raw_budget = avail_os_ram_bytes - full_fit_guard;
        plan.use_mlock = true;
    } else if (ram_lock_policy == "on") {
        // Explicit --ram-lock-policy on in near-fit regime: cap mlock to 60% of avail RAM so unpinned
        // layers + OS keep >=40% reclaimable page cache (prevents kswapd/NVMe thrashing).
        raw_budget = std::min<size_t>(raw_budget, (avail_os_ram_bytes * 3ULL) / 5ULL);
        plan.use_mlock = true;
    } else {
        // Near-fit "auto": zero-copy-prewarm into Hot RAM (reclaimable page cache, zero starvation)
        plan.use_mlock = false;
    }
    plan.budget_bytes = raw_budget;

    struct ScoredLayer {
        int    layer;
        size_t bytes;
        int    prio;
    };
    std::vector<ScoredLayer> ranked;
    ranked.reserve(cpu_layers.size());
    for (const auto & cl : cpu_layers) {
        if (cl.layer >= 0 && cl.bytes > 0) {
            ranked.push_back({cl.layer, cl.bytes, zero_copy_layer_priority(cl.layer, n_model_layers)});
        }
    }
    std::sort(ranked.begin(), ranked.end(), [](const ScoredLayer & a, const ScoredLayer & b) {
        if (a.prio != b.prio) return a.prio > b.prio;
        return a.layer < b.layer;
    });

    for (const auto & r : ranked) {
        if (plan.planned_bytes + r.bytes <= plan.budget_bytes) {
            plan.selected_layers.push_back(r.layer);
            plan.planned_bytes += r.bytes;
        }
    }
    return plan;
}

// ---------------------------------------------------------------------------
// Phase 18 WP3: Race-Free 3-Tensor Async NVMe -> Tier-0 VRAM Reservation
// ---------------------------------------------------------------------------
struct AsyncVramReservation {
    int      slot        = -1;
    int      layer       = -1;
    int      expert      = -1;
    uint64_t dev_ptrs[3] = {0, 0, 0};
    size_t   bytes[3]    = {0, 0, 0};
    bool valid() const { return slot >= 0; }
};

// ---------------------------------------------------------------------------
// Phase 0.2 + 1.3 (tandem.cpp 13 Nov): Unified Real Tier-0 CUDA + Tier-1 RAM Cache
// Backed by genuine CUdeviceptr VRAM and mlock/VirtualLock DDR5 expert weight storage.
// ---------------------------------------------------------------------------
class UnifiedTieredExpertCache {
public:
    static constexpr int MAX_LAYERS  = 64;
    static constexpr int MAX_EXPERTS = 512;
    using AsyncVramReservation = atlas::AsyncVramReservation;

    struct ExpertTensorView {
        const uint8_t * gate_host = nullptr;
        const uint8_t * up_host   = nullptr;
        const uint8_t * down_host = nullptr;
        size_t gate_bytes = 0;
        size_t up_bytes   = 0;
        size_t down_bytes = 0;
        size_t total_bytes() const { return gate_bytes + up_bytes + down_bytes; }
    };

    struct LayerMeta {
        size_t gate_expert_bytes = 0;
        size_t up_expert_bytes   = 0;
        size_t down_expert_bytes = 0;
        size_t slot_bytes() const {
            return align256(gate_expert_bytes) + align256(up_expert_bytes) + align256(down_expert_bytes);
        }
        size_t up_offset() const { return align256(gate_expert_bytes); }
        size_t down_offset() const { return align256(gate_expert_bytes) + align256(up_expert_bytes); }
        int32_t gate_type = -1;
        int32_t up_type   = -1;
        int32_t down_type = -1;
        int64_t n_embd    = 0;
        int64_t n_ff      = 0;
        const uint8_t * gate_base = nullptr;
        const uint8_t * up_base   = nullptr;
        const uint8_t * down_base = nullptr;
    };

    UnifiedTieredExpertCache() = default;
    ~UnifiedTieredExpertCache() { shutdown(); }

    UnifiedTieredExpertCache(const UnifiedTieredExpertCache &) = delete;
    UnifiedTieredExpertCache & operator=(const UnifiedTieredExpertCache &) = delete;

    bool init(size_t vram_budget_bytes,
              size_t ram_budget_bytes,
              size_t slot_stride_bytes,
              float crf_half_life = 128.0f) {
        std::lock_guard<std::mutex> lk(mtx_);
        shutdown_unlocked();

        slot_stride_bytes_ = align256(std::max<size_t>(4096, slot_stride_bytes));
        mailbox_.init();

        // 1. Real Tier-0 CUDA VRAM Arena
        const int v_slots = int(vram_budget_bytes / slot_stride_bytes_);
        if (v_slots > 0) {
            vram_dev_base_ = CudaDriverMemBridge::alloc_device(size_t(v_slots) * slot_stride_bytes_);
            if (vram_dev_base_ != 0) {
                vram_slots_ = v_slots;
                vram_policy_.init(v_slots, crf_half_life);
                h2d_stream_ = CudaDriverMemBridge::create_stream(true);
            }
        }

        // 2. Real Tier-1 Locked Host RAM Arena (genuine expert weight bytes, zero mmap thrash)
        const int r_slots = int(ram_budget_bytes / slot_stride_bytes_);
        if (r_slots > 0) {
            const size_t r_bytes = size_t(r_slots) * slot_stride_bytes_;
            ram_host_base_ = alloc_locked_ram(r_bytes, ram_locked_);
            if (ram_host_base_) {
                ram_slots_ = r_slots;
                ram_policy_.init(r_slots, crf_half_life * 2.0f);
            }
        }

        std::fill(&vram_lut_[0][0], &vram_lut_[0][0] + MAX_LAYERS * MAX_EXPERTS, int32_t(-1));
        std::fill(&ram_lut_[0][0],  &ram_lut_[0][0]  + MAX_LAYERS * MAX_EXPERTS, int32_t(-1));
        std::memset(access_freq_, 0, sizeof(access_freq_));
        std::memset(step_hits_,   0, sizeof(step_hits_));
        touched_pairs_.clear();
        initialized_ = true;
        return true;
    }

    void shutdown() {
        std::lock_guard<std::mutex> lk(mtx_);
        shutdown_unlocked();
    }

    void register_layer_weights(int layer, const LayerMeta & meta) {
        if (layer < 0 || layer >= MAX_LAYERS) return;
        std::lock_guard<std::mutex> lk(mtx_);
        layer_meta_[layer] = meta;
    }

    // Lookup Tier-0 VRAM slot (-1 if miss)
    inline int32_t get_vram_slot(int layer, int expert) const {
        if (layer < 0 || layer >= MAX_LAYERS || expert < 0 || expert >= MAX_EXPERTS) return -1;
        return vram_lut_[layer][expert];
    }

    // Lookup Tier-1 Locked RAM pointer for specific tensor (kind: 0=gate, 1=up, 2=down)
    inline const void * get_ram_tensor_ptr(int layer, int expert, int kind) const {
        if (!ram_host_base_ || layer < 0 || layer >= MAX_LAYERS || expert < 0 || expert >= MAX_EXPERTS) return nullptr;
        const int32_t rslot = ram_lut_[layer][expert];
        if (rslot < 0) return nullptr;
        const LayerMeta & m = layer_meta_[layer];
        const size_t base_off = size_t(rslot) * slot_stride_bytes_;
        if (kind == 0) return ram_host_base_ + base_off;
        if (kind == 1) return ram_host_base_ + base_off + align256(m.gate_expert_bytes);
        if (kind == 2) return ram_host_base_ + base_off + align256(m.gate_expert_bytes) + align256(m.up_expert_bytes);
        return nullptr;
    }

    inline uint64_t get_vram_slot_dev_ptr(int slot, int layer, int kind) const {
        if (!vram_dev_base_ || slot < 0 || slot >= vram_slots_ || layer < 0 || layer >= MAX_LAYERS) return 0;
        const LayerMeta & m = layer_meta_[layer];
        const size_t base_off = size_t(slot) * slot_stride_bytes_;
        if (kind == 0) return vram_dev_base_ + base_off;
        if (kind == 1) return vram_dev_base_ + base_off + align256(m.gate_expert_bytes);
        if (kind == 2) return vram_dev_base_ + base_off + align256(m.gate_expert_bytes) + align256(m.up_expert_bytes);
        return 0;
    }

    // Record access and optionally promote missing expert to Tier-1 RAM (real memcpy from mmap)
    // and/or Tier-0 VRAM (real cuMemcpyHtoDAsync). Returns true if resident in Tier-0 VRAM.
    using SlotEventCallback = std::function<void(int layer, int expert, int slot, bool admitted)>;

    void set_ram_callback(SlotEventCallback cb) {
        std::lock_guard<std::mutex> lk(mtx_);
        on_ram_event_ = std::move(cb);
    }

    void set_vram_callback(SlotEventCallback cb) {
        std::lock_guard<std::mutex> lk(mtx_);
        on_vram_event_ = std::move(cb);
    }

    // Phase 18 WP1.2: Layer-concentrated Tier-0 VRAM mask (default ~0ULL = all layers)
    void set_vram_layer_mask(uint64_t mask) {
        vram_layer_mask_.store(mask, std::memory_order_release);
    }

    uint64_t vram_layer_mask() const {
        return vram_layer_mask_.load(std::memory_order_acquire);
    }

    // Phase 18 WP3: Async DMA LUT dirty tracking
    bool consume_lut_dirty() {
        return lut_dirty_.exchange(false, std::memory_order_acq_rel);
    }

    bool is_lut_dirty() const {
        return lut_dirty_.load(std::memory_order_acquire);
    }

    void clear_lut_dirty() {
        lut_dirty_.store(false, std::memory_order_release);
    }

    // Phase 18 WP3: Reserve a non-evictable Tier-0 VRAM destination slot for async 3-tensor DMA.
    // Evicts lowest-priority non-reserved slot if full (immediately clearing old owner's LUT entry
    // and marking lut_dirty_ = true) without publishing the new (layer, expert) until commit.
    AsyncVramReservation reserve_async_vram_slot(
            int layer,
            int expert,
            const std::function<bool(int, int)> & is_protected = nullptr) {
        AsyncVramReservation out;
        if (layer < 0 || layer >= MAX_LAYERS || expert < 0 || expert >= MAX_EXPERTS) return out;

        std::lock_guard<std::mutex> lk(mtx_);
        if (!initialized_ || !has_real_vram()) return out;

        const uint64_t mask = vram_layer_mask_.load(std::memory_order_acquire);
        if (((mask >> layer) & 1ULL) == 0ULL) return out;

        const LayerMeta & m = layer_meta_[layer];
        const size_t needed = m.slot_bytes();
        if (needed == 0 || needed > slot_stride_bytes_) return out;

        if (vram_lut_[layer][expert] >= 0) return out;
        if (vram_policy_.is_reserved_key(layer, expert)) return out;

        if (vram_policy_.free_slots() == 0) {
            int ev_l = -1, ev_e = -1;
            const int ev = vram_policy_.evict_victim(is_protected, &ev_l, &ev_e);
            if (ev < 0) {
                return out; // All VRAM slots are in-flight reserved or protected
            }
            if (ev_l >= 0 && ev_l < MAX_LAYERS && ev_e >= 0 && ev_e < MAX_EXPERTS) {
                vram_lut_[ev_l][ev_e] = -1;
                lut_dirty_.store(true, std::memory_order_release);
                if (on_vram_event_) on_vram_event_(ev_l, ev_e, ev, false);
            }
        }

        const int slot = vram_policy_.reserve_free_slot(layer, expert);
        if (slot < 0) return out;

        const uint64_t dev_base = vram_dev_base_ + size_t(slot) * slot_stride_bytes_;
        out.slot        = slot;
        out.layer       = layer;
        out.expert      = expert;
        out.dev_ptrs[0] = dev_base;
        out.dev_ptrs[1] = dev_base + m.up_offset();
        out.dev_ptrs[2] = dev_base + m.down_offset();
        out.bytes[0]    = m.gate_expert_bytes;
        out.bytes[1]    = m.up_expert_bytes;
        out.bytes[2]    = m.down_expert_bytes;
        return out;
    }

    // Finalize an async NVMe -> VRAM promotion after CUDA DMA event completes.
    bool commit_async_vram_slot(int layer, int expert, int slot) {
        if (layer < 0 || layer >= MAX_LAYERS || expert < 0 || expert >= MAX_EXPERTS) return false;

        std::lock_guard<std::mutex> lk(mtx_);
        if (!initialized_ || !has_real_vram()) return false;
        if (slot < 0 || slot >= vram_slots_) return false;

        step_++;
        const float boost = layer_importance(layer);
        if (!vram_policy_.commit_reserved_slot(slot, layer, expert, step_, boost)) {
            return false;
        }

        vram_lut_[layer][expert] = slot;
        lut_dirty_.store(true, std::memory_order_release);
        dma_promoted_.fetch_add(1, std::memory_order_relaxed);
        const LayerMeta & m = layer_meta_[layer];
        h2d_bytes_copied_.fetch_add(
            m.gate_expert_bytes + m.up_expert_bytes + m.down_expert_bytes,
            std::memory_order_relaxed);
        if (on_vram_event_) on_vram_event_(layer, expert, slot, true);
        return true;
    }

    // Release an in-flight reserved VRAM slot cleanly on DMA error or shutdown.
    void abort_async_vram_slot(int slot) {
        std::lock_guard<std::mutex> lk(mtx_);
        if (!initialized_ || slot < 0 || slot >= vram_slots_) return;
        vram_policy_.abort_reserved_slot(slot);
    }

    bool access_and_promote(int layer, int expert, bool promote_to_ram, bool promote_to_vram,
                            const std::function<bool(int, int)> & is_protected = nullptr) {
        if (layer < 0 || layer >= MAX_LAYERS || expert < 0 || expert >= MAX_EXPERTS) return false;
        std::lock_guard<std::mutex> lk(mtx_);
        step_++;
        const float boost = layer_importance(layer);

        const int32_t vslot = vram_lut_[layer][expert];
        if (vslot >= 0) {
            vram_policy_.touch(vslot, step_, boost);
            vram_hits_++;
            return true;
        }
        vram_misses_++;

        int32_t rslot = ram_lut_[layer][expert];
        if (rslot >= 0) {
            ram_policy_.touch(rslot, step_, boost);
            ram_hits_++;
        } else {
            ram_misses_++;
            if (promote_to_ram && ram_slots_ > 0) {
                rslot = admit_ram_unlocked(layer, expert, boost, is_protected);
            }
        }

        const uint64_t mask = vram_layer_mask_.load(std::memory_order_acquire);
        const bool layer_vram_allowed = (layer >= 0 && layer < 64 && ((mask >> layer) & 1ULL) != 0ULL);
        if (promote_to_vram && layer_vram_allowed && vram_slots_ > 0 && vram_dev_base_ != 0) {
            const int new_vslot = admit_vram_unlocked(layer, expert, rslot, boost, is_protected, true);
            if (new_vslot >= 0) return true;
        }
        return false;
    }

    // Race-free two-phase promotion API (zero heap allocs, zero layer starvation):
    // During llama_decode, accumulate per-(layer, expert) access histograms without mutating active LUTs.
    // Between llama_decode calls, commit_pending_promotions() fills free VRAM/RAM slots immediately and
    // applies scan-resistant 2-hit admission + bounded replacement once caches are warm.
    void touch_or_queue_candidate(int layer, int expert, float score = 1.0f) {
        if (layer < 0 || layer >= MAX_LAYERS || expert < 0 || expert >= MAX_EXPERTS) return;
        std::lock_guard<std::mutex> lk(mtx_);
        step_++;
        if (access_freq_[layer][expert] < 65535u) {
            access_freq_[layer][expert]++;
        }
        const float boost = layer_importance(layer) * std::max(0.25f, score);
        const int32_t vslot = vram_lut_[layer][expert];
        if (vslot >= 0) {
            vram_policy_.touch(vslot, step_, boost);
            vram_hits_++;
            return;
        }
        vram_misses_++;
        const int32_t rslot = ram_lut_[layer][expert];
        if (rslot >= 0) {
            ram_policy_.touch(rslot, step_, boost);
            ram_hits_++;
        } else {
            ram_misses_++;
        }
        if (step_hits_[layer][expert] == 0) {
            touched_pairs_.push_back({layer, expert});
        }
        step_hits_[layer][expert]++;
    }

    // Returns true if vram_lut changed (caller should sync vram_lut to CPU + CUDA).
    bool commit_pending_promotions(
            AdaptiveCpuShareController & share_ctrl,
            bool promote_ram,
            bool promote_vram,
            int max_vram_uploads = 48,
            const std::function<bool(int, int)> & is_protected = nullptr) {
        std::lock_guard<std::mutex> lk(mtx_);
        if (touched_pairs_.empty()) {
            return lut_dirty_.exchange(false, std::memory_order_acq_rel);
        }

        std::vector<PendingCandidate> unique_cands;
        unique_cands.reserve(touched_pairs_.size());
        for (const auto & p : touched_pairs_) {
            const uint32_t hits = step_hits_[p.first][p.second];
            step_hits_[p.first][p.second] = 0;
            if (hits == 0) continue;
            // Prioritize recurring experts across ALL 48 layers using cumulative frequency + step hits
            const float val = layer_importance(p.first) * (float(hits) + 0.5f * std::min<float>(float(access_freq_[p.first][p.second]), 16.0f));
            unique_cands.push_back({p.first, p.second, val});
        }
        touched_pairs_.clear();
        if (unique_cands.empty()) {
            return lut_dirty_.exchange(false, std::memory_order_acq_rel);
        }

        std::sort(unique_cands.begin(), unique_cands.end(), [](const PendingCandidate & a, const PendingCandidate & b) {
            return a.score > b.score;
        });

        bool vram_changed = false;
        int vram_uploaded = 0;
        int ram_uploaded  = 0;

        const uint64_t layer_mask = vram_layer_mask_.load(std::memory_order_acquire);
        const bool vram_has_free = (vram_policy_.free_slots() > 0);
        const bool ram_has_free  = (ram_policy_.free_slots() > 0);
        const int warm_vram_cap  = std::min<int>(max_vram_uploads, vram_has_free ? std::min<int>(vram_slots_, 256) : 6);
        const int warm_ram_cap   = ram_has_free ? std::min<int>(ram_slots_, 512) : 16;

        // Compute PR #30966 share budget on effective miss pool
        const int n_pool = std::min<int>(64, (int)unique_cands.size());
        const auto dec = share_ctrl.plan_misses(n_pool, 0ULL, std::max(1, vram_slots_), promote_vram);
        int vram_cap = 0;
        if (promote_vram) {
            if (vram_has_free) {
                vram_cap = std::min<int>(warm_vram_cap, (int)unique_cands.size());
            } else {
                const int by_share = std::max(1, dec.gpu_upload_count);
                vram_cap = std::min<int>(warm_vram_cap, by_share);
            }
        }

        for (const auto & c : unique_cands) {
            int32_t rslot = ram_lut_[c.layer][c.expert];
            if (rslot < 0 && promote_ram && ram_slots_ > 0 && ram_uploaded < warm_ram_cap) {
                if (ram_has_free || access_freq_[c.layer][c.expert] >= 2) {
                    rslot = admit_ram_unlocked(c.layer, c.expert, c.score, is_protected);
                    if (rslot >= 0) ram_uploaded++;
                }
            }
            const bool layer_vram_allowed = (c.layer >= 0 && c.layer < 64 && ((layer_mask >> c.layer) & 1ULL) != 0ULL);
            if (promote_vram && layer_vram_allowed && vram_slots_ > 0 && vram_dev_base_ != 0 &&
                vram_lut_[c.layer][c.expert] < 0 && vram_uploaded < vram_cap) {
                // Scan-resistant 2-hit admission once VRAM is full (never evicts warm experts for 1-off scans)
                if (vram_has_free || access_freq_[c.layer][c.expert] >= 2) {
                    const int new_vslot = admit_vram_unlocked(c.layer, c.expert, rslot, c.score, is_protected, false);
                    if (new_vslot >= 0) {
                        vram_uploaded++;
                        vram_changed = true;
                    }
                }
            }
        }
        if (vram_uploaded > 0 && h2d_stream_) {
            CudaDriverMemBridge::sync_stream(h2d_stream_);
        }
        const bool was_dirty = lut_dirty_.exchange(false, std::memory_order_acq_rel);
        return vram_changed || was_dirty;
    }

    bool has_real_vram() const { return vram_dev_base_ != 0 && vram_slots_ > 0; }
    bool has_real_ram()  const { return ram_host_base_ != nullptr && ram_slots_ > 0; }
    int  vram_slots()    const { return vram_slots_; }
    int  ram_slots()     const { return ram_slots_; }
    int  vram_used()     const { return vram_policy_.used_slots(); }
    int  vram_reserved() const {
        std::lock_guard<std::mutex> lk(mtx_);
        return vram_policy_.reserved_slots();
    }
    bool is_vram_slot_reserved(int slot) const {
        std::lock_guard<std::mutex> lk(mtx_);
        return vram_policy_.is_slot_reserved(slot);
    }
    int  ram_used()      const { return ram_policy_.used_slots(); }
    size_t slot_stride_bytes() const { return slot_stride_bytes_; }
    uint64_t vram_dev_base()   const { return vram_dev_base_; }
    bool is_ram_locked()       const { return ram_locked_; }
    uint64_t vram_hits()   const { return vram_hits_; }
    uint64_t vram_misses() const { return vram_misses_; }
    uint64_t ram_hits()    const { return ram_hits_; }
    uint64_t ram_misses()  const { return ram_misses_; }
    uint64_t h2d_bytes()   const { return h2d_bytes_copied_.load(std::memory_order_relaxed); }
    uint64_t vram_bytes_copied() const { return h2d_bytes_copied_.load(std::memory_order_relaxed); }
    uint64_t ram_bytes()   const { return ram_bytes_copied_; }
    uint64_t get_dma_promoted() const { return dma_promoted_.load(std::memory_order_relaxed); }

    MappedRouterMailboxRing & mailbox() { return mailbox_; }
    const int32_t * vram_lut_flat() const { return &vram_lut_[0][0]; }

private:
    mutable std::mutex mtx_;
    bool initialized_ = false;
    uint64_t step_ = 0;
    size_t slot_stride_bytes_ = 0;

    int vram_slots_ = 0;
    uint64_t vram_dev_base_ = 0;
    void * h2d_stream_ = nullptr;
    ClassLRUPolicy vram_policy_;

    int ram_slots_ = 0;
    uint8_t * ram_host_base_ = nullptr;
    bool ram_locked_ = false;
    ClassLRUPolicy ram_policy_;

    int32_t vram_lut_[MAX_LAYERS][MAX_EXPERTS];
    int32_t ram_lut_[MAX_LAYERS][MAX_EXPERTS];
    uint16_t access_freq_[MAX_LAYERS][MAX_EXPERTS] = {};
    uint32_t step_hits_[MAX_LAYERS][MAX_EXPERTS] = {};
    std::vector<std::pair<int, int>> touched_pairs_;
    LayerMeta layer_meta_[MAX_LAYERS];

    MappedRouterMailboxRing mailbox_;
    SlotEventCallback on_ram_event_;
    SlotEventCallback on_vram_event_;

    uint64_t vram_hits_ = 0;
    uint64_t vram_misses_ = 0;
    uint64_t ram_hits_ = 0;
    uint64_t ram_misses_ = 0;
    std::atomic<uint64_t> h2d_bytes_copied_{0};
    uint64_t ram_bytes_copied_ = 0;
    std::atomic<uint64_t> vram_layer_mask_{~0ULL};
    std::atomic<bool>     lut_dirty_{false};
    std::atomic<uint64_t> dma_promoted_{0};

    struct PendingCandidate {
        int   layer  = 0;
        int   expert = 0;
        float score  = 1.0f;
    };

    static inline size_t align256(size_t x) {
        return (x + 255ULL) & ~size_t(255ULL);
    }

    static inline float layer_importance(int layer) {
        if (layer < 2 || (layer >= 15 && layer <= 30) || layer >= 44) return 1.25f;
        return 1.0f;
    }

    static uint8_t * alloc_locked_ram(size_t bytes, bool & out_locked) {
        out_locked = false;
        if (bytes == 0) return nullptr;
#if defined(_WIN32)
        void * p = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        if (p) {
            out_locked = (VirtualLock(p, bytes) != 0);
            return static_cast<uint8_t *>(p);
        }
#else
        const size_t aligned = (bytes + 4095ULL) & ~size_t(4095ULL);
        void * p = mmap(nullptr, aligned, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p != MAP_FAILED && p != nullptr) {
            madvise(p, aligned, MADV_HUGEPAGE);
            out_locked = (mlock(p, std::min<size_t>(aligned, 512ULL * 1024 * 1024)) == 0);
            return static_cast<uint8_t *>(p);
        }
#endif
        return nullptr;
    }

    static void free_locked_ram(uint8_t * p, size_t bytes) {
        if (!p || bytes == 0) return;
#if defined(_WIN32)
        VirtualUnlock(p, bytes);
        VirtualFree(p, 0, MEM_RELEASE);
#else
        const size_t aligned = (bytes + 4095ULL) & ~size_t(4095ULL);
        munlock(p, std::min<size_t>(aligned, 512ULL * 1024 * 1024));
        munmap(p, aligned);
#endif
    }

    int admit_ram_unlocked(int layer, int expert, float boost,
                           const std::function<bool(int, int)> & is_protected) {
        const LayerMeta & m = layer_meta_[layer];
        const size_t needed = m.slot_bytes();
        if (needed == 0 || needed > slot_stride_bytes_) return -1;

        if (ram_policy_.free_slots() == 0) {
            int ev_l = -1, ev_e = -1;
            const int ev = ram_policy_.evict_victim(is_protected, &ev_l, &ev_e);
            if (ev >= 0 && ev_l >= 0 && ev_e >= 0) {
                ram_lut_[ev_l][ev_e] = -1;
                if (on_ram_event_) on_ram_event_(ev_l, ev_e, ev, false);
            }
        }
        const int rslot = ram_policy_.admit(layer, expert, step_, boost);
        if (rslot < 0) return -1;

        uint8_t * dst = ram_host_base_ + size_t(rslot) * slot_stride_bytes_;
        size_t off = 0;
        if (m.gate_base && m.gate_expert_bytes > 0) {
            std::memcpy(dst + off, m.gate_base + size_t(expert) * m.gate_expert_bytes, m.gate_expert_bytes);
            off += align256(m.gate_expert_bytes);
        }
        if (m.up_base && m.up_expert_bytes > 0) {
            std::memcpy(dst + off, m.up_base + size_t(expert) * m.up_expert_bytes, m.up_expert_bytes);
            off += align256(m.up_expert_bytes);
        }
        if (m.down_base && m.down_expert_bytes > 0) {
            std::memcpy(dst + off, m.down_base + size_t(expert) * m.down_expert_bytes, m.down_expert_bytes);
            off += align256(m.down_expert_bytes);
        }
        ram_bytes_copied_ += m.gate_expert_bytes + m.up_expert_bytes + m.down_expert_bytes;
        ram_lut_[layer][expert] = rslot;
        if (on_ram_event_) on_ram_event_(layer, expert, rslot, true);
        return rslot;
    }

    int admit_vram_unlocked(int layer, int expert, int rslot, float boost,
                            const std::function<bool(int, int)> & is_protected,
                            bool sync_now = true) {
        if (vram_policy_.is_reserved_key(layer, expert)) return -1;
        const LayerMeta & m = layer_meta_[layer];
        const size_t needed = m.slot_bytes();
        if (needed == 0 || needed > slot_stride_bytes_) return -1;

        if (vram_policy_.free_slots() == 0) {
            int ev_l = -1, ev_e = -1;
            const int ev = vram_policy_.evict_victim(is_protected, &ev_l, &ev_e);
            if (ev >= 0 && ev_l >= 0 && ev_e >= 0) {
                vram_lut_[ev_l][ev_e] = -1;
                if (on_vram_event_) on_vram_event_(ev_l, ev_e, ev, false);
            }
        }
        const int vslot = vram_policy_.admit(layer, expert, step_, boost);
        if (vslot < 0) return -1;

        const uint64_t dev_dst = vram_dev_base_ + size_t(vslot) * slot_stride_bytes_;
        bool copied = false;
        if (rslot >= 0 && ram_host_base_) {
            const uint8_t * src_ram = ram_host_base_ + size_t(rslot) * slot_stride_bytes_;
            copied = CudaDriverMemBridge::copy_h2d_async(dev_dst, src_ram, needed, h2d_stream_);
            if (copied) h2d_bytes_copied_.fetch_add(needed, std::memory_order_relaxed);
        } else {
            size_t off = 0;
            bool ok = true;
            if (m.gate_base && m.gate_expert_bytes > 0) {
                ok &= CudaDriverMemBridge::copy_h2d_async(
                    dev_dst + off, m.gate_base + size_t(expert) * m.gate_expert_bytes, m.gate_expert_bytes, h2d_stream_);
                off += align256(m.gate_expert_bytes);
            }
            if (ok && m.up_base && m.up_expert_bytes > 0) {
                ok &= CudaDriverMemBridge::copy_h2d_async(
                    dev_dst + off, m.up_base + size_t(expert) * m.up_expert_bytes, m.up_expert_bytes, h2d_stream_);
                off += align256(m.up_expert_bytes);
            }
            if (ok && m.down_base && m.down_expert_bytes > 0) {
                ok &= CudaDriverMemBridge::copy_h2d_async(
                    dev_dst + off, m.down_base + size_t(expert) * m.down_expert_bytes, m.down_expert_bytes, h2d_stream_);
            }
            copied = ok;
            if (copied) {
                h2d_bytes_copied_.fetch_add(
                    m.gate_expert_bytes + m.up_expert_bytes + m.down_expert_bytes,
                    std::memory_order_relaxed);
            }
        }
        if (!copied) {
            vram_policy_.release(vslot);
            return -1;
        }
        if (sync_now) {
            CudaDriverMemBridge::sync_stream(h2d_stream_);
        }
        vram_lut_[layer][expert] = vslot;
        if (on_vram_event_) on_vram_event_(layer, expert, vslot, true);
        return vslot;
    }

    void shutdown_unlocked() {
        mailbox_.release();
        if (h2d_stream_) {
            CudaDriverMemBridge::destroy_stream(h2d_stream_);
            h2d_stream_ = nullptr;
        }
        if (vram_dev_base_) {
            CudaDriverMemBridge::free_device(vram_dev_base_);
            vram_dev_base_ = 0;
        }
        if (ram_host_base_) {
            free_locked_ram(ram_host_base_, size_t(ram_slots_) * slot_stride_bytes_);
            ram_host_base_ = nullptr;
        }
        vram_policy_.init(0);
        ram_policy_.init(0);
        lut_dirty_.store(false, std::memory_order_release);
        vram_slots_ = 0;
        ram_slots_  = 0;
        ram_locked_ = false;
        initialized_ = false;
    }
};

} // namespace atlas
