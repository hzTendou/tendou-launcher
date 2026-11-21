#pragma once

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>
#include <mutex>
#include <atomic>
#include <memory>

#if defined(__linux__)
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
    uint64_t  request_id       = 0;
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
    static constexpr size_t RING_ENTRIES     = 64;

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

    // Enqueue an asynchronous O_DIRECT sector-aligned read from NVMe SSD
    bool enqueue_ssd_read(int slot_idx, uint64_t file_offset, size_t bytes, int layer, int expert, int blob_idx = 0) {
        if (slot_idx < 0 || slot_idx >= (int)NUM_SLOTS) return false;
        if (bytes == 0 || bytes > SLOT_SIZE) return false;
        int active_fd = (blob_idx >= 0 && blob_idx < 3 && fds_[blob_idx] >= 0) ? fds_[blob_idx] : fd_;
        if (active_fd < 0) return false;

        std::lock_guard<std::mutex> lock(mtx_);

        // Ensure 4096-byte alignment for O_DIRECT
        uint64_t aligned_offset = (file_offset / SECTOR_ALIGNMENT) * SECTOR_ALIGNMENT;
        size_t offset_diff = file_offset - aligned_offset;
        size_t aligned_bytes = ((bytes + offset_diff + SECTOR_ALIGNMENT - 1) / SECTOR_ALIGNMENT) * SECTOR_ALIGNMENT;
        if (aligned_bytes > SLOT_SIZE) aligned_bytes = SLOT_SIZE;

        DmaSlot & s = slots_[slot_idx];
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

    // Wait for in-flight NVMe read completion on given slot
    bool wait_slot_completed(int slot_idx) {
        if (slot_idx < 0 || slot_idx >= (int)NUM_SLOTS) return false;
        DmaSlot & s = slots_[slot_idx];
        if (s.is_completed) return true;
        if (!s.in_flight) return false;

#if defined(__linux__)
        if (ring_initialized_ && fn_wait_cqes_) {
            struct io_uring_cqe * cqe = nullptr;
            int ret = fn_wait_cqes_(&ring_, &cqe, 1, nullptr, nullptr);
            if (ret == 0 && cqe) {
                DmaSlot * completed_slot = (DmaSlot *)cqe->user_data;
                if (completed_slot) {
                    completed_slot->in_flight = false;
                    completed_slot->is_completed = true;
                    completed_requests_++;
                    bytes_streamed_ += completed_slot->size_bytes;
                }
                if (ring_.cq.khead) {
                    (*ring_.cq.khead)++;
                }
                return s.is_completed;
            }
        }
#endif
        return s.is_completed;
    }

    // Transfer completed slot to GPU VRAM Slab Arena via PCIe Gen4 DMA
    // Safely skips alignment offset padding so tensor payload lands at exact VRAM destination.
    bool dispatch_to_vram_async(int slot_idx, void * d_dest_vram, size_t bytes) {
        if (slot_idx < 0 || slot_idx >= (int)NUM_SLOTS || !d_dest_vram) return false;
        DmaSlot & s = slots_[slot_idx];
        if (!s.is_completed) {
            if (!wait_slot_completed(slot_idx)) return false;
        }

#if defined(__linux__)
        // High-priority asynchronous CUDA memory copy: host pinned staging -> device VRAM
        void * hCuda = atlas_get_libcuda_handle();
        if (hCuda) {
            using cuDevicePrimaryCtxRetain_t = int (*)(void **, int);
            using cuCtxSetCurrent_t = int (*)(void *);
            auto fnRetainCtx = (cuDevicePrimaryCtxRetain_t)dlsym(hCuda, "cuDevicePrimaryCtxRetain");
            auto fnSetCtx = (cuCtxSetCurrent_t)dlsym(hCuda, "cuCtxSetCurrent");
            if (fnRetainCtx && fnSetCtx) {
                void * ctx = nullptr;
                if (fnRetainCtx(&ctx, 0) == 0 && ctx) {
                    fnSetCtx(ctx);
                }
            }
            using cuMemcpyHtoDAsync_t = int (*)(void * dstDevice, const void * srcHost, size_t ByteCount, void * hStream);
            auto fnMemcpy = (cuMemcpyHtoDAsync_t)dlsym(hCuda, "cuMemcpyHtoDAsync_v2");
            if (!fnMemcpy) fnMemcpy = (cuMemcpyHtoDAsync_t)dlsym(hCuda, "cuMemcpyHtoDAsync");
            if (fnMemcpy) {
                const uint8_t * src_payload = s.host_pinned_ptr + s.payload_offset;
                int res = fnMemcpy(d_dest_vram, src_payload, bytes, stream_);
                if (res == 0) {
                    s.dev_vram_ptr = d_dest_vram;
                    dma_transfers_++;
                    return true;
                }
            }
        }
#endif
        return false;
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
    struct AppIoUring ring_{};
    void * liburing_handle_ = nullptr;
    fn_io_uring_queue_init_params fn_init_params_ = nullptr;
    fn_io_uring_queue_init        fn_init_ = nullptr;
    fn_io_uring_queue_exit        fn_exit_ = nullptr;
    fn_io_uring_get_sqe           fn_get_sqe_ = nullptr;
    fn_io_uring_submit            fn_submit_ = nullptr;
    fn_io_uring_wait_cqes         fn_wait_cqes_ = nullptr;
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
