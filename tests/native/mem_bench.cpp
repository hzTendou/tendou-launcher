#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <immintrin.h>
#include <windows.h>
#include <atomic>
#include <iomanip>

struct alignas(64) ThreadData {
    const __m256i* start;
    size_t count;
    std::atomic<int>* barrier1;
    std::atomic<int>* barrier2;
    int num_threads;
    double duration_s;
    int thread_id;
};

void worker(ThreadData* td) {
    DWORD_PTR mask = 1ULL << td->thread_id;
    SetThreadAffinityMask(GetCurrentThread(), mask);

    ++(*td->barrier1);
    while (td->barrier1->load(std::memory_order_acquire) < td->num_threads) { _mm_pause(); }

    const __m256i* ptr = td->start;
    size_t count = td->count;

    auto t0 = std::chrono::high_resolution_clock::now();

    __m256i sum0 = _mm256_setzero_si256();
    __m256i sum1 = _mm256_setzero_si256();
    __m256i sum2 = _mm256_setzero_si256();
    __m256i sum3 = _mm256_setzero_si256();

    size_t i = 0;
    for (; i + 3 < count; i += 4) {
        sum0 = _mm256_add_epi64(sum0, _mm256_load_si256(ptr + i + 0));
        sum1 = _mm256_add_epi64(sum1, _mm256_load_si256(ptr + i + 1));
        sum2 = _mm256_add_epi64(sum2, _mm256_load_si256(ptr + i + 2));
        sum3 = _mm256_add_epi64(sum3, _mm256_load_si256(ptr + i + 3));
    }
    for (; i < count; ++i) {
        sum0 = _mm256_add_epi64(sum0, _mm256_load_si256(ptr + i));
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    td->duration_s = std::chrono::duration<double>(t1 - t0).count();

    volatile __m256i sink = _mm256_add_epi64(_mm256_add_epi64(sum0, sum1), _mm256_add_epi64(sum2, sum3));

    ++(*td->barrier2);
    while (td->barrier2->load(std::memory_order_acquire) < td->num_threads) { _mm_pause(); }
}

void run_test(const __m256i* data, size_t total_count, int num_threads) {
    std::atomic<int> b1{0};
    std::atomic<int> b2{0};

    std::vector<ThreadData> tds(num_threads);
    std::vector<std::thread> threads;

    size_t chunk = total_count / num_threads;

    for (int i = 0; i < num_threads; ++i) {
        tds[i].start = data + i * chunk;
        tds[i].count = (i == num_threads - 1) ? (total_count - i * chunk) : chunk;
        tds[i].barrier1 = &b1;
        tds[i].barrier2 = &b2;
        tds[i].num_threads = num_threads;
        tds[i].thread_id = (i < 8) ? (i * 2) : ((i - 8) * 2 + 1);
        threads.emplace_back(worker, &tds[i]);
    }

    for (auto& t : threads) t.join();

    double max_time = 0;
    for (int i = 0; i < num_threads; ++i) {
        if (tds[i].duration_s > max_time) max_time = tds[i].duration_s;
    }

    double gb = (double)(total_count * 32) / (1024 * 1024 * 1024);
    double gbps = gb / max_time;

    std::cout << "Threads: " << std::setw(2) << num_threads << " | "
              << "Bandwidth: " << std::setprecision(5) << gbps << " GB/s | "
              << "Time: " << max_time << " s\n";
}

int main() {
    size_t size_gb = 4;
    size_t total_bytes = size_gb * 1024ULL * 1024ULL * 1024ULL;
    size_t num_elements = total_bytes / 32;

    std::cout << "Allocating " << size_gb << " GB...\n";
    __m256i* data = (__m256i*)_aligned_malloc(total_bytes, 64);
    if (!data) return 1;

    memset(data, 1, total_bytes);

    int thread_configs[] = {1, 2, 4, 8, 16};
    
    for (int tc : thread_configs) {
        for (int rep = 0; rep < 5; ++rep) {
            run_test(data, num_elements, tc);
        }
        std::cout << "--------------------------------\n";
    }

    _aligned_free(data);
    return 0;
}
