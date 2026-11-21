#pragma once

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>
#include <mutex>
#include <atomic>
#include <memory>

#if defined(_WIN32)
#include <malloc.h>
#elif defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#include <dlfcn.h>
#include <sys/mman.h>
#include <linux/io_uring.h>
#endif

namespace atlas {

// ---------------------------------------------------------------------------
// Dynamic liburing bindings & fallback structures for Linux io_uring
// ---------------------------------------------------------------------------
#if defined(__linux__)

struct AppIoUringSq {
    unsigned * khead;
    unsigned * ktail;
    unsigned * kring_mask;
    unsigned * kring_entries;
    unsigned * kflags;
    unsigned * kdropped;
    unsigned * array;
    struct io_uring_sqe * sqes;
    unsigned sqe_head;
    unsigned sqe_tail;
    size_t ring_sz;
    void * ring_ptr;
    unsigned ring_mask;
    unsigned ring_entries;
    unsigned pad[2];
};

struct AppIoUringCq {
    unsigned * khead;
    unsigned * ktail;
    unsigned * kring_mask;
    unsigned * kring_entries;
    unsigned * kflags;
    unsigned * koverflow;
    struct io_uring_cqe * cqes;
    size_t ring_sz;
    void * ring_ptr;
    unsigned ring_mask;
    unsigned ring_entries;
    unsigned pad[2];
};

struct AppIoUring {
    struct AppIoUringSq sq;
    struct AppIoUringCq cq;
    unsigned flags;
    int ring_fd;
    unsigned features;
    int enter_ring_fd;
    __u8 int_flags;
    __u8 pad[3];
    unsigned pad2;
};

typedef int (*fn_io_uring_queue_init_params)(unsigned entries, struct AppIoUring * ring, struct io_uring_params * p);
typedef int (*fn_io_uring_queue_init)(unsigned entries, struct AppIoUring * ring, unsigned flags);
typedef void (*fn_io_uring_queue_exit)(struct AppIoUring * ring);
typedef struct io_uring_sqe * (*fn_io_uring_get_sqe)(struct AppIoUring * ring);
typedef int (*fn_io_uring_submit)(struct AppIoUring * ring);
typedef int (*fn_io_uring_wait_cqes)(struct AppIoUring * ring, struct io_uring_cqe ** cqe_ptr, unsigned wait_nr, void * ts, void * sig);

#endif // __linux__

// Function prototypes from atlas-engine.h
void * atlas_get_libcuda_handle();
bool atlas_cuda_host_register(void * ptr, size_t size);
void atlas_cuda_host_unregister(void * ptr);

struct DmaSlot {
    uint8_t * host_pinned_ptr = nullptr;
    void    * dev_vram_ptr     = nullptr;
    size_t    size_bytes       = 0;
    size_t    payload_offset   = 0;
    int       layer            = -1;
    int       expert_id        = -1;
    uint64_t  file_offset      = 0;
    bool      in_flight        = false;
    bool      is_completed     = false;
    bool      reserved_by_job  = false;
    uint64_t  request_id       = 0;
};

// Phase 18 WP3: Asynchronous 3-tensor (gate/up/down) NVMe -> Tier-0 VRAM DMA job handle
struct ExpertDmaJob {
    int    layer          = -1;
    int    expert         = -1;
    int    vram_slot      = -1;
    int    dma_slots[3]   = {-1, -1, -1};
    int    n_chunks       = 0;
    void * cuda_event     = nullptr;
    bool   completed_sync = false;
    bool   valid          = false;
};

// ===========================================================================
// DirectStorageIoUringDMA: High-performance O_DIRECT + io_uring NVMe DMA Engine
// Bypasses Linux page cache, VFS buffers, and FUSE to stream weights directly
// from NVMe SSD into Pinned Host Ring and asynchronously DMA to GPU GDDR6 VRAM.
// ===========================================================================
class DirectStorageIoUringDMA {
public:
    static constexpr size_t SECTOR_ALIGNMENT = 4096;
    static constexpr size_t NUM_SLOTS        = 4;
    static constexpr size_t SLOT_SIZE        = 32 * 1024 * 1024; // 32 MiB circular staging slot (total 128 MiB)
    static constexpr size_t MAX_SLICE_BYTES  = SLOT_SIZE - SECTOR_ALIGNMENT; // Guaranteed aligned fit for any offset
    static constexpr size_t RING_ENTRIES     = 64;
    using ExpertDmaJob = atlas::ExpertDmaJob;

    explicit DirectStorageIoUringDMA(int model_fd = -1, void * cuda_dma_stream = nullptr)
        : fd_(model_fd), stream_(cuda_dma_stream) {
        fds_[0] = model_fd;
        allocate_pinned_ring();
        init_uring();
    }

    ~DirectStorageIoUringDMA() {
        shutdown_uring();
        free_pinned_ring();
        for (int i = 0; i < 3; ++i) {
            if (fds_[i] >= 0) {
#if defined(__linux__)
                close(fds_[i]);
#endif
                fds_[i] = -1;
            }
        }
        fd_ = -1;
    }

    // Non-copyable, movable
    DirectStorageIoUringDMA(const DirectStorageIoUringDMA &) = delete;
    DirectStorageIoUringDMA & operator=(const DirectStorageIoUringDMA &) = delete;

    void set_blob_fd(int blob_idx, int fd) {
        if (blob_idx >= 0 && blob_idx < 3) {
            fds_[blob_idx] = fd;
            if (blob_idx == 0) fd_ = fd;
        }
    }

    int get_blob_fd(int blob_idx) const {
        if (blob_idx >= 0 && blob_idx < 3) return fds_[blob_idx];
        return -1;
    }

    bool is_active() const { return ring_initialized_ || fd_ >= 0; }
    bool is_sqpoll() const { return is_sqpoll_; }
    bool is_cuda_pinned() const { return is_cuda_pinned_; }
    size_t slot_capacity() const { return SLOT_SIZE; }
    size_t num_slots() const { return NUM_SLOTS; }
    void set_cuda_stream(void * cuda_dma_stream) { stream_ = cuda_dma_stream; }
    void * get_cuda_stream() const { return stream_; }

    // Phase 18 WP3: Thread-safe slot reservation so concurrent DMA jobs never collide
    int acquire_free_slot() {
        std::lock_guard<std::mutex> lock(mtx_);
        for (int i = 0; i < (int)NUM_SLOTS; ++i) {
            DmaSlot & s = slots_[i];
            if (!s.in_flight && !s.reserved_by_job && s.host_pinned_ptr != nullptr) {
                s.reserved_by_job = true;
                s.is_completed    = false;
                return i;
            }
        }
        return -1;
    }

    void release_slot(int slot_idx) {
        if (slot_idx < 0 || slot_idx >= (int)NUM_SLOTS) return;
        std::lock_guard<std::mutex> lock(mtx_);
        DmaSlot & s = slots_[slot_idx];
        s.in_flight       = false;
        s.is_completed    = false;
        s.reserved_by_job = false;
    }

    // Enqueue an asynchronous O_DIRECT sector-aligned read from NVMe SSD
    bool enqueue_ssd_read(int slot_idx, uint64_t file_offset, size_t bytes, int layer, int expert, int blob_idx = 0) {
        if (slot_idx < 0 || slot_idx >= (int)NUM_SLOTS) return false;
        if (bytes == 0 || bytes > SLOT_SIZE) return false;
        int active_fd = (blob_idx >= 0 && blob_idx < 3 && fds_[blob_idx] >= 0) ? fds_[blob_idx] : fd_;
        if (active_fd < 0) return false;

        std::lock_guard<std::mutex> lock(mtx_);
        DmaSlot & s = slots_[slot_idx];
        if (!s.host_pinned_ptr) return false;

        // Ensure 4096-byte alignment for O_DIRECT
        uint64_t aligned_offset = (file_offset / SECTOR_ALIGNMENT) * SECTOR_ALIGNMENT;
        size_t offset_diff = file_offset - aligned_offset;
        if (bytes + offset_diff > SLOT_SIZE) return false;
        size_t aligned_bytes = ((bytes + offset_diff + SECTOR_ALIGNMENT - 1) / SECTOR_ALIGNMENT) * SECTOR_ALIGNMENT;
        if (aligned_bytes > SLOT_SIZE) aligned_bytes = SLOT_SIZE;

        s.layer = layer;
        s.expert_id = expert;
        s.file_offset = file_offset;
        s.payload_offset = offset_diff;
        s.size_bytes = bytes;
        s.in_flight = true;
        s.is_completed = false;
        s.request_id = ++submitted_requests_;

#if defined(__linux__)
        if (ring_initialized_ && fn_get_sqe_ && fn_submit_) {
            struct io_uring_sqe * sqe = fn_get_sqe_(&ring_);
            if (sqe) {
                std::memset(sqe, 0, sizeof(*sqe));
                sqe->opcode = IORING_OP_READ;
                sqe->fd = active_fd;
                sqe->off = aligned_offset;
                sqe->addr = (uint64_t)s.host_pinned_ptr;
                sqe->len = aligned_bytes;
                sqe->user_data = (uint64_t)&s;
                int ret = fn_submit_(&ring_);
                if (ret >= 0) {
                    return true;
                }
            }
        }

        // Fallback: synchronous aligned pread (fast on kernel ntfs3)
        ssize_t n = pread(active_fd, s.host_pinned_ptr, aligned_bytes, aligned_offset);
        if (n >= 0) {
            s.in_flight = false;
            s.is_completed = true;
            completed_requests_++;
            bytes_streamed_ += bytes;
            return true;
        }
#endif
        s.in_flight = false;
        return false;
    }

    // Wait for in-flight NVMe read completion on given slot (mutex-protected, multi-CQE safe)
    bool wait_slot_completed(int slot_idx) {
        if (slot_idx < 0 || slot_idx >= (int)NUM_SLOTS) return false;
        std::lock_guard<std::mutex> lock(mtx_);
        DmaSlot & s = slots_[slot_idx];
        if (s.is_completed) return true;
        if (!s.in_flight) return false;

#if defined(__linux__)
        if (ring_initialized_ && fn_wait_cqes_) {
            int max_drain = (int)RING_ENTRIES;
            while (!s.is_completed && s.in_flight && max_drain-- > 0) {
                struct io_uring_cqe * cqe = nullptr;
                int ret = fn_wait_cqes_(&ring_, &cqe, 1, nullptr, nullptr);
                if (ret == 0 && cqe) {
                    DmaSlot * completed_slot = (DmaSlot *)cqe->user_data;
                    const bool ok = (cqe->res >= 0);
                    if (completed_slot) {
                        completed_slot->in_flight = false;
                        completed_slot->is_completed = ok;
                        if (ok) {
                            completed_requests_++;
                            bytes_streamed_ += completed_slot->size_bytes;
                        }
                    }
                    if (ring_.cq.khead) {
                        __atomic_store_n(ring_.cq.khead, *ring_.cq.khead + 1, __ATOMIC_RELEASE);
                    }
                } else {
                    if (ret == -4) { // -EINTR
                        continue;
                    }
                    s.in_flight = false;
                    s.is_completed = false;
                    break;
                }
            }
            return s.is_completed;
        }
#endif
        return s.is_completed;
    }

    // Transfer completed slot to GPU VRAM Slab Arena via PCIe Gen4 DMA
    // Safely skips alignment offset padding so tensor payload lands at exact VRAM destination.
    bool dispatch_to_vram_async(int slot_idx, void * d_dest_vram, size_t bytes) {
        if (slot_idx < 0 || slot_idx >= (int)NUM_SLOTS || !d_dest_vram || bytes == 0) return false;
        uint8_t * host_base = nullptr;
        bool completed = false;
        {
            std::lock_guard<std::mutex> lock(mtx_);
            host_base = slots_[slot_idx].host_pinned_ptr;
            completed = slots_[slot_idx].is_completed;
        }
        if (!host_base) return false;
        if (!completed) {
            if (!wait_slot_completed(slot_idx)) return false;
        }
        size_t payload_off = 0;
        {
            std::lock_guard<std::mutex> lock(mtx_);
            payload_off = slots_[slot_idx].payload_offset;
        }
        if (payload_off + bytes > SLOT_SIZE) return false;

#if defined(__linux__)
        // High-priority asynchronous CUDA memory copy: host pinned staging -> device VRAM
        const CudaDmaApi & api = get_cuda_api();
        if (ensure_cuda_ctx(api) && api.fnMemcpyHtoD) {
            const uint8_t * src_payload = host_base + payload_off;
            int res = api.fnMemcpyHtoD(d_dest_vram, src_payload, bytes, stream_);
            if (res == 0) {
                {
                    std::lock_guard<std::mutex> lock(mtx_);
                    slots_[slot_idx].dev_vram_ptr = d_dest_vram;
                }
                dma_transfers_++;
                return true;
            }
        }
#endif
        return false;
    }

    // Phase 18 WP3: Race-free 3-tensor async NVMe -> Tier-0 VRAM transfer with large-tensor chunking
    // and non-blocking CUevent completion tracking.
    bool enqueue_expert_3tensor_read(int layer,
                                     int expert,
                                     int vram_slot,
                                     const int blob_idxs[3],
                                     const uint64_t file_offsets[3],
                                     const size_t byte_sizes[3],
                                     const uint64_t vram_dev_ptrs[3],
                                     int n_chunks,
                                     ExpertDmaJob & out_job) {
        out_job = ExpertDmaJob{};
        if (n_chunks <= 0 || n_chunks > 3) return false;
        if (!blob_idxs || !file_offsets || !byte_sizes || !vram_dev_ptrs) return false;

        int non_zero_chunks = 0;
        bool all_single_slice = true;
        for (int c = 0; c < n_chunks; ++c) {
            if (byte_sizes[c] == 0) continue;
            if (vram_dev_ptrs[c] == 0) return false;
            non_zero_chunks++;
            if (byte_sizes[c] > MAX_SLICE_BYTES) {
                all_single_slice = false;
            }
        }
        if (non_zero_chunks == 0) return false;

        int acquired[3] = {-1, -1, -1};
        for (int c = 0; c < n_chunks; ++c) {
            if (byte_sizes[c] == 0) continue;
            acquired[c] = acquire_free_slot();
            if (acquired[c] < 0) {
                for (int k = 0; k < c; ++k) {
                    if (acquired[k] >= 0) release_slot(acquired[k]);
                }
                return false;
            }
        }

        auto cleanup_failed = [&](bool drain_h2d) {
            if (drain_h2d) {
                cuda_sync_stream();
            }
            for (int k = 0; k < n_chunks; ++k) {
                if (acquired[k] >= 0) release_slot(acquired[k]);
            }
        };

        bool any_h2d_launched = false;
        if (all_single_slice) {
            // Fast path: submit all NVMe reads concurrently (QD=n_chunks) then dispatch async H2D
            int submitted_upto = 0;
            for (int c = 0; c < n_chunks; ++c) {
                if (acquired[c] < 0) continue;
                if (!enqueue_ssd_read(acquired[c], file_offsets[c], byte_sizes[c], layer, expert, blob_idxs[c])) {
                    for (int k = 0; k < submitted_upto; ++k) {
                        if (acquired[k] >= 0) wait_slot_completed(acquired[k]);
                    }
                    cleanup_failed(false);
                    return false;
                }
                submitted_upto = c + 1;
            }

            for (int c = 0; c < n_chunks; ++c) {
                if (acquired[c] < 0) continue;
                if (!wait_slot_completed(acquired[c])) {
                    for (int k = c + 1; k < n_chunks; ++k) {
                        if (acquired[k] >= 0) wait_slot_completed(acquired[k]);
                    }
                    cleanup_failed(any_h2d_launched);
                    return false;
                }
                void * dst_ptr = reinterpret_cast<void *>(static_cast<uintptr_t>(vram_dev_ptrs[c]));
                if (!dispatch_to_vram_async(acquired[c], dst_ptr, byte_sizes[c])) {
                    for (int k = c + 1; k < n_chunks; ++k) {
                        if (acquired[k] >= 0) wait_slot_completed(acquired[k]);
                    }
                    cleanup_failed(any_h2d_launched);
                    return false;
                }
                any_h2d_launched = true;
            }
        } else {
            // Large-tensor safe path: chunk tensors exceeding MAX_SLICE_BYTES (SLOT_SIZE - SECTOR_ALIGNMENT)
            // synchronizing stream between intermediate slices so pinned host buffer is never overwritten mid-DMA.
            for (int c = 0; c < n_chunks; ++c) {
                if (acquired[c] < 0) continue;
                const int slot = acquired[c];
                const size_t total_bytes = byte_sizes[c];
                const uint64_t base_off  = file_offsets[c];
                const uint64_t base_vram = vram_dev_ptrs[c];
                const int blob_idx       = blob_idxs[c];

                size_t pos = 0;
                while (pos < total_bytes) {
                    const size_t slice_bytes = std::min(total_bytes - pos, MAX_SLICE_BYTES);
                    if (!enqueue_ssd_read(slot, base_off + pos, slice_bytes, layer, expert, blob_idx)) {
                        cleanup_failed(any_h2d_launched);
                        return false;
                    }
                    if (!wait_slot_completed(slot)) {
                        cleanup_failed(any_h2d_launched);
                        return false;
                    }
                    void * dst_ptr = reinterpret_cast<void *>(static_cast<uintptr_t>(base_vram + pos));
                    if (!dispatch_to_vram_async(slot, dst_ptr, slice_bytes)) {
                        cleanup_failed(any_h2d_launched);
                        return false;
                    }
                    any_h2d_launched = true;
                    pos += slice_bytes;
                    if (pos < total_bytes) {
                        cuda_sync_stream();
                    }
                }
            }
        }

        out_job.layer          = layer;
        out_job.expert         = expert;
        out_job.vram_slot      = vram_slot;
        out_job.n_chunks       = n_chunks;
        out_job.cuda_event     = nullptr;
        out_job.completed_sync = false;
        out_job.valid          = true;
        for (int c = 0; c < 3; ++c) {
            out_job.dma_slots[c] = (c < n_chunks) ? acquired[c] : -1;
        }

        void * ev = cuda_create_and_record_event();
        if (ev != nullptr) {
            out_job.cuda_event     = ev;
            out_job.completed_sync = false;
        } else {
            cuda_sync_stream();
            out_job.cuda_event     = nullptr;
            out_job.completed_sync = true;
        }
        return true;
    }

    // Non-blocking poll for async expert DMA job completion; releases DMA staging slots and CUevent when done.
    // Preserves job.layer, job.expert, job.vram_slot so caller can immediately commit the VRAM slot.
    bool poll_job_complete(ExpertDmaJob & job) {
        if (!job.valid) return false;

        bool done = false;
        if (job.completed_sync) {
            done = true;
        } else if (job.cuda_event != nullptr) {
            done = cuda_query_event_complete(job.cuda_event);
        } else {
            done = true;
        }
        if (!done) return false;

        if (job.cuda_event != nullptr) {
            cuda_destroy_event(job.cuda_event);
            job.cuda_event = nullptr;
        }
        for (int c = 0; c < job.n_chunks && c < 3; ++c) {
            if (job.dma_slots[c] >= 0) {
                release_slot(job.dma_slots[c]);
                job.dma_slots[c] = -1;
            }
        }
        job.completed_sync = false;
        job.valid          = false;
        return true;
    }

    // Safely drain/destroy CUevent and release all DMA staging slots on error or shutdown.
    void abort_job(ExpertDmaJob & job) {
        if (job.cuda_event != nullptr) {
            cuda_sync_event(job.cuda_event);
            cuda_destroy_event(job.cuda_event);
            job.cuda_event = nullptr;
        }
        for (int c = 0; c < 3; ++c) {
            if (job.dma_slots[c] >= 0) {
                release_slot(job.dma_slots[c]);
                job.dma_slots[c] = -1;
            }
        }
        job.n_chunks       = 0;
        job.completed_sync = false;
        job.valid          = false;
    }

    void * get_slot_ptr(int slot_idx) const {
        if (slot_idx < 0 || slot_idx >= (int)NUM_SLOTS) return nullptr;
        return slots_[slot_idx].host_pinned_ptr;
    }

    size_t get_slot_payload_offset(int slot_idx) const {
        if (slot_idx < 0 || slot_idx >= (int)NUM_SLOTS) return 0;
        return slots_[slot_idx].payload_offset;
    }

    const DmaSlot & get_slot(int slot_idx) const {
        return slots_[slot_idx];
    }

    uint64_t get_submitted_requests() const { return submitted_requests_.load(); }
    uint64_t get_completed_requests() const { return completed_requests_.load(); }
    uint64_t get_dma_transfers() const { return dma_transfers_.load(); }
    uint64_t get_bytes_streamed() const { return bytes_streamed_.load(); }

private:
    int fd_ = -1;
    int fds_[3] = {-1, -1, -1};
    void * stream_ = nullptr;
    DmaSlot slots_[NUM_SLOTS];
    bool ring_initialized_ = false;
    bool is_sqpoll_ = false;
    bool is_cuda_pinned_ = false;
    std::mutex mtx_;

    std::atomic<uint64_t> submitted_requests_{0};
    std::atomic<uint64_t> completed_requests_{0};
    std::atomic<uint64_t> dma_transfers_{0};
    std::atomic<uint64_t> bytes_streamed_{0};

#if defined(__linux__)
    struct CudaDmaApi {
        using cuInit_t                   = int (*)(unsigned int);
        using cuCtxGetCurrent_t          = int (*)(void **);
        using cuDevicePrimaryCtxRetain_t = int (*)(void **, int);
        using cuCtxSetCurrent_t          = int (*)(void *);
        using cuMemcpyHtoDAsync_t        = int (*)(void *, const void *, size_t, void *);
        using cuStreamSync_t             = int (*)(void *);
        using cuEventCreate_t            = int (*)(void **, unsigned int);
        using cuEventDestroy_t           = int (*)(void *);
        using cuEventRecord_t            = int (*)(void *, void *);
        using cuEventQuery_t             = int (*)(void *);
        using cuEventSync_t              = int (*)(void *);

        bool                       loaded         = false;
        cuInit_t                   fnInit         = nullptr;
        cuCtxGetCurrent_t          fnGetCtx       = nullptr;
        cuDevicePrimaryCtxRetain_t fnRetainCtx    = nullptr;
        cuCtxSetCurrent_t          fnSetCtx       = nullptr;
        cuMemcpyHtoDAsync_t        fnMemcpyHtoD   = nullptr;
        cuStreamSync_t             fnStreamSync   = nullptr;
        cuEventCreate_t            fnEventCreate  = nullptr;
        cuEventDestroy_t           fnEventDestroy = nullptr;
        cuEventRecord_t            fnEventRecord  = nullptr;
        cuEventQuery_t             fnEventQuery   = nullptr;
        cuEventSync_t              fnEventSync    = nullptr;
    };

    static CudaDmaApi get_cuda_api() {
        static std::mutex s_api_mtx;
        static CudaDmaApi s_api;
        std::lock_guard<std::mutex> lk(s_api_mtx);
        if (!s_api.loaded) {
            void * hCuda = atlas_get_libcuda_handle();
            if (hCuda) {
                s_api.fnInit       = (CudaDmaApi::cuInit_t)dlsym(hCuda, "cuInit");
                s_api.fnGetCtx     = (CudaDmaApi::cuCtxGetCurrent_t)dlsym(hCuda, "cuCtxGetCurrent");
                s_api.fnRetainCtx  = (CudaDmaApi::cuDevicePrimaryCtxRetain_t)dlsym(hCuda, "cuDevicePrimaryCtxRetain");
                s_api.fnSetCtx     = (CudaDmaApi::cuCtxSetCurrent_t)dlsym(hCuda, "cuCtxSetCurrent");
                s_api.fnMemcpyHtoD = (CudaDmaApi::cuMemcpyHtoDAsync_t)dlsym(hCuda, "cuMemcpyHtoDAsync_v2");
                if (!s_api.fnMemcpyHtoD) s_api.fnMemcpyHtoD = (CudaDmaApi::cuMemcpyHtoDAsync_t)dlsym(hCuda, "cuMemcpyHtoDAsync");
                s_api.fnStreamSync   = (CudaDmaApi::cuStreamSync_t)dlsym(hCuda, "cuStreamSynchronize");
                s_api.fnEventCreate  = (CudaDmaApi::cuEventCreate_t)dlsym(hCuda, "cuEventCreate");
                s_api.fnEventDestroy = (CudaDmaApi::cuEventDestroy_t)dlsym(hCuda, "cuEventDestroy_v2");
                if (!s_api.fnEventDestroy) s_api.fnEventDestroy = (CudaDmaApi::cuEventDestroy_t)dlsym(hCuda, "cuEventDestroy");
                s_api.fnEventRecord  = (CudaDmaApi::cuEventRecord_t)dlsym(hCuda, "cuEventRecord");
                s_api.fnEventQuery   = (CudaDmaApi::cuEventQuery_t)dlsym(hCuda, "cuEventQuery");
                s_api.fnEventSync    = (CudaDmaApi::cuEventSync_t)dlsym(hCuda, "cuEventSynchronize");
                s_api.loaded = true;
            }
        }
        return s_api;
    }

    static bool ensure_cuda_ctx(const CudaDmaApi & api) {
        if (!api.loaded) return false;
        if (api.fnGetCtx) {
            void * cur = nullptr;
            if (api.fnGetCtx(&cur) == 0 && cur != nullptr) {
                return true;
            }
        }
        if (api.fnInit) api.fnInit(0);
        if (api.fnRetainCtx && api.fnSetCtx) {
            void * ctx = nullptr;
            if (api.fnRetainCtx(&ctx, 0) == 0 && ctx != nullptr) {
                return api.fnSetCtx(ctx) == 0;
            }
        }
        return false;
    }

    void * cuda_create_and_record_event() const {
        const CudaDmaApi & api = get_cuda_api();
        if (!ensure_cuda_ctx(api) || !api.fnEventCreate || !api.fnEventRecord || !api.fnEventQuery || !api.fnEventDestroy) {
            return nullptr;
        }
        void * ev = nullptr;
        // CU_EVENT_DISABLE_TIMING (0x02) minimizes driver overhead for completion polling
        if (api.fnEventCreate(&ev, 0x02u) != 0 || !ev) {
            return nullptr;
        }
        if (api.fnEventRecord(ev, stream_) != 0) {
            api.fnEventDestroy(ev);
            return nullptr;
        }
        return ev;
    }

    static bool cuda_query_event_complete(void * ev) {
        if (!ev) return true;
        const CudaDmaApi & api = get_cuda_api();
        if (!ensure_cuda_ctx(api) || !api.fnEventQuery) return true;
        return api.fnEventQuery(ev) == 0;
    }

    static void cuda_sync_event(void * ev) {
        if (!ev) return;
        const CudaDmaApi & api = get_cuda_api();
        if (ensure_cuda_ctx(api) && api.fnEventSync) {
            api.fnEventSync(ev);
        }
    }

    static void cuda_destroy_event(void * ev) {
        if (!ev) return;
        const CudaDmaApi & api = get_cuda_api();
        if (ensure_cuda_ctx(api) && api.fnEventDestroy) {
            api.fnEventDestroy(ev);
        }
    }

    void cuda_sync_stream() const {
        const CudaDmaApi & api = get_cuda_api();
        if (ensure_cuda_ctx(api) && api.fnStreamSync) {
            api.fnStreamSync(stream_);
        }
    }

    struct AppIoUring ring_{};
    void * liburing_handle_ = nullptr;
    fn_io_uring_queue_init_params fn_init_params_ = nullptr;
    fn_io_uring_queue_init        fn_init_ = nullptr;
    fn_io_uring_queue_exit        fn_exit_ = nullptr;
    fn_io_uring_get_sqe           fn_get_sqe_ = nullptr;
    fn_io_uring_submit            fn_submit_ = nullptr;
    fn_io_uring_wait_cqes         fn_wait_cqes_ = nullptr;
#else
    void * cuda_create_and_record_event() const { return nullptr; }
    static bool cuda_query_event_complete(void *) { return true; }
    static void cuda_sync_event(void *) {}
    static void cuda_destroy_event(void *) {}
    void cuda_sync_stream() const {}
#endif

    void allocate_pinned_ring() {
        bool all_pinned = true;
        for (size_t i = 0; i < NUM_SLOTS; ++i) {
            void * ptr = nullptr;
#if defined(_WIN32)
            ptr = _aligned_malloc(SLOT_SIZE, SECTOR_ALIGNMENT);
#elif defined(__linux__)
            if (posix_memalign(&ptr, SECTOR_ALIGNMENT, SLOT_SIZE) != 0) {
                ptr = nullptr;
            }
#else
            ptr = std::malloc(SLOT_SIZE);
#endif
            if (ptr) {
                std::memset(ptr, 0, 4096); // Fault first page
                slots_[i].host_pinned_ptr = static_cast<uint8_t *>(ptr);
                slots_[i].size_bytes = SLOT_SIZE;
                slots_[i].in_flight = false;
                slots_[i].is_completed = false;
                slots_[i].reserved_by_job = false;

                if (atlas_cuda_host_register(ptr, SLOT_SIZE)) {
                    // Pinned successfully
                } else {
                    all_pinned = false;
                }
            } else {
                all_pinned = false;
            }
        }
        is_cuda_pinned_ = all_pinned;
    }

    void free_pinned_ring() {
        for (size_t i = 0; i < NUM_SLOTS; ++i) {
            if (slots_[i].host_pinned_ptr) {
                atlas_cuda_host_unregister(slots_[i].host_pinned_ptr);
#if defined(_WIN32)
                _aligned_free(slots_[i].host_pinned_ptr);
#elif defined(__linux__)
                std::free(slots_[i].host_pinned_ptr);
#else
                std::free(slots_[i].host_pinned_ptr);
#endif
                slots_[i].host_pinned_ptr = nullptr;
            }
        }
    }

    void init_uring() {
#if defined(__linux__)
        liburing_handle_ = dlopen("liburing.so.2", RTLD_NOW);
        if (!liburing_handle_) {
            liburing_handle_ = dlopen("liburing.so", RTLD_NOW);
        }
        if (liburing_handle_) {
            fn_init_params_ = (fn_io_uring_queue_init_params)dlsym(liburing_handle_, "io_uring_queue_init_params");
            fn_init_        = (fn_io_uring_queue_init)dlsym(liburing_handle_, "io_uring_queue_init");
            fn_exit_        = (fn_io_uring_queue_exit)dlsym(liburing_handle_, "io_uring_queue_exit");
            fn_get_sqe_     = (fn_io_uring_get_sqe)dlsym(liburing_handle_, "io_uring_get_sqe");
            fn_submit_      = (fn_io_uring_submit)dlsym(liburing_handle_, "io_uring_submit");
            fn_wait_cqes_   = (fn_io_uring_wait_cqes)dlsym(liburing_handle_, "io_uring_wait_cqes");

            if (fn_init_params_ && fn_exit_ && fn_get_sqe_ && fn_submit_) {
                struct io_uring_params params{};
                params.flags = IORING_SETUP_SQPOLL; // Kernel SQ polling thread
                params.sq_thread_idle = 2000;
                int ret = fn_init_params_(RING_ENTRIES, &ring_, &params);
                if (ret == 0) {
                    ring_initialized_ = true;
                    is_sqpoll_ = true;
                    return;
                }
            }

            if (fn_init_ && fn_exit_ && fn_get_sqe_ && fn_submit_) {
                int ret = fn_init_(RING_ENTRIES, &ring_, 0);
                if (ret == 0) {
                    ring_initialized_ = true;
                    is_sqpoll_ = false;
                    return;
                }
            }
        }
#endif
    }

    void shutdown_uring() {
#if defined(__linux__)
        if (ring_initialized_ && fn_exit_) {
            fn_exit_(&ring_);
            ring_initialized_ = false;
        }
        if (liburing_handle_) {
            dlclose(liburing_handle_);
            liburing_handle_ = nullptr;
        }
#endif
    }
};

} // namespace atlas
