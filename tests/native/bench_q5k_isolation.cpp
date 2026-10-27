#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <cassert>
#include <windows.h>
#include <intrin.h>
#include <immintrin.h>

#include "ggml.h"
#include "ggml-quants.h"

#define QK_K 256

#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

static inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

static inline __m256i get_scale_shuffle_k4(int i) {
    static const uint8_t k_shuffle[256] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
         6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
    };
    return _mm256_loadu_si256((const __m256i*)k_shuffle + i);
}

static inline float fp16_to_fp32(ggml_fp16_t x) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x)));
}

static inline ggml_fp16_t fp32_to_fp16(float x) {
    return (ggml_fp16_t)_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(x), 0), 0);
}

static void my_vec_dot_q5_K_q8_K(int n, float * s, const void * vx, const void * vy) {
    assert(n % QK_K == 0);

    const block_q5_K * x = (const block_q5_K *)vx;
    const block_q8_K * y = (const block_q8_K *)vy;

    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m128i mzero = _mm_setzero_si128();
    const __m256i mone  = _mm256_set1_epi8(1);

    __m256 acc = _mm256_setzero_ps();
    float summs = 0.f;

    for (int i = 0; i < nb; ++i) {
        const uint8_t * q5 = x[i].qs;
        const int8_t  * q8 = y[i].qs;

        const float d = y[i].d * fp16_to_fp32(x[i].d);
        const float dmin = -y[i].d * fp16_to_fp32(x[i].dmin);

        std::memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));

        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        const __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
        summs += dmin * _mm_extract_epi32(hsum, 0);

        const __m128i sc128  = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = MM256_SET_M128I(sc128, sc128);

        const __m256i hbits = _mm256_loadu_si256((const __m256i*)x[i].qh);
        __m256i hmask = mone;

        __m256i sumi = _mm256_setzero_si256();
        int bit = 0;

        for (int j = 0; j < QK_K/64; ++j) {
            const __m256i scale_0 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+0));
            const __m256i scale_1 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+1));

            const __m256i q5bits = _mm256_loadu_si256((const __m256i*)q5); q5 += 32;

            const __m256i q5l_0 = _mm256_and_si256(q5bits, m4);
            const __m256i q5h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            const __m256i q5_0  = _mm256_add_epi8(q5l_0, q5h_0);
            hmask = _mm256_slli_epi16(hmask, 1);

            const __m256i q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);
            const __m256i q5h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            const __m256i q5_1  = _mm256_add_epi8(q5l_1, q5h_1);
            hmask = _mm256_slli_epi16(hmask, 1);

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            __m256i p16_0 = _mm256_maddubs_epi16(q5_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q5_1, q8_1);

            p16_0 = _mm256_madd_epi16(scale_0, p16_0);
            p16_1 = _mm256_madd_epi16(scale_1, p16_1);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
        }

        __m256 vd = _mm256_set1_ps(d);
        acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);
    }

    *s = hsum_float_8(acc) + summs;
}

// Qwen3.8 Flash Next parameters
static constexpr int N_DIM = 2560;
static constexpr size_t Q5_K_ROW_BYTES = (N_DIM / 256) * sizeof(block_q5_K); // 10 * 176 = 1760 bytes
static constexpr size_t Q8_K_ROW_BYTES = (N_DIM / 256) * sizeof(block_q8_K); // 10 * 260 = 2600 bytes

struct TestResult {
    double duration_ms;
    double gb_per_sec;
    double vec_dots_per_sec;
    double cycles_per_dot;
    double scaling_eff;
};

void run_vec_dot_benchmark(
    const std::string& name,
    int num_threads,
    bool use_smt,
    bool is_hot,
    bool is_expert_layout,
    int num_iterations_per_thread,
    double single_thread_dots_per_sec,
    TestResult& out_res
) {
    const size_t num_rows = is_hot ? 256 : (64 * 1024);
    std::vector<uint8_t> q5_buf(num_rows * Q5_K_ROW_BYTES, 0x55);
    std::vector<uint8_t> q8_buf(Q8_K_ROW_BYTES, 0xAA);

    for (size_t r = 0; r < num_rows; ++r) {
        block_q5_K* b = (block_q5_K*)(q5_buf.data() + r * Q5_K_ROW_BYTES);
        for (int k = 0; k < N_DIM / 256; ++k) {
            b[k].d = fp32_to_fp16(1.0f);
            b[k].dmin = fp32_to_fp16(0.5f);
        }
    }
    block_q8_K* b8 = (block_q8_K*)q8_buf.data();
    for (int k = 0; k < N_DIM / 256; ++k) {
        b8[k].d = 1.0f;
    }

    std::atomic<int> start_barrier{0};
    std::atomic<int> stop_barrier{0};
    std::vector<std::thread> workers;
    workers.reserve(num_threads);

    std::vector<double> thread_durations(num_threads, 0.0);
    std::vector<uint64_t> thread_cycles(num_threads, 0);

    for (int t = 0; t < num_threads; ++t) {
        workers.emplace_back([&, t]() {
            DWORD_PTR mask = 1ULL << (use_smt ? t : (t * 2));
            SetThreadAffinityMask(GetCurrentThread(), mask);

            float dot_res = 0.0f;
            const void* vy = q8_buf.data();

            if (is_hot) {
                for (size_t i = 0; i < num_rows; ++i) {
                    const void* vx = q5_buf.data() + i * Q5_K_ROW_BYTES;
                    my_vec_dot_q5_K_q8_K(N_DIM, &dot_res, vx, vy);
                }
            }

            start_barrier.fetch_add(1);
            while (start_barrier.load(std::memory_order_acquire) < num_threads) {
                _mm_pause();
            }

            uint64_t c0 = __rdtsc();
            auto t0 = std::chrono::high_resolution_clock::now();

            size_t curr_row = (t * 1024) % num_rows;
            for (int it = 0; it < num_iterations_per_thread; ++it) {
                const void* vx = q5_buf.data() + curr_row * Q5_K_ROW_BYTES;
                my_vec_dot_q5_K_q8_K(N_DIM, &dot_res, vx, vy);

                if (is_hot) {
                    curr_row = (curr_row + 1) % num_rows;
                } else if (is_expert_layout) {
                    curr_row = (curr_row + 1024) % num_rows;
                } else {
                    curr_row = (curr_row + 1) % num_rows;
                }
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            uint64_t c1 = __rdtsc();

            thread_durations[t] = std::chrono::duration<double, std::milli>(t1 - t0).count();
            thread_cycles[t] = c1 - c0;

            stop_barrier.fetch_add(1);
            while (stop_barrier.load(std::memory_order_acquire) < num_threads) {
                _mm_pause();
            }
        });
    }

    for (auto& w : workers) {
        w.join();
    }

    double max_dur_ms = 0.0;
    uint64_t total_cycles = 0;
    for (int t = 0; t < num_threads; ++t) {
        if (thread_durations[t] > max_dur_ms) max_dur_ms = thread_durations[t];
        total_cycles += thread_cycles[t];
    }

    int64_t total_dots = (int64_t)num_threads * num_iterations_per_thread;
    double total_bytes = (double)total_dots * Q5_K_ROW_BYTES;
    double effective_gb_s = (total_bytes / (1024.0 * 1024.0 * 1024.0)) / (max_dur_ms / 1000.0);
    double dots_per_sec = (double)total_dots / (max_dur_ms / 1000.0);
    double avg_cycles_per_dot = (double)total_cycles / total_dots;

    double scaling_eff = 1.0;
    if (single_thread_dots_per_sec > 0.0) {
        scaling_eff = (dots_per_sec / (single_thread_dots_per_sec * num_threads));
    }

    out_res = { max_dur_ms, effective_gb_s, dots_per_sec, avg_cycles_per_dot, scaling_eff };

    std::cout << std::left << std::setw(28) << name
              << " | " << std::setw(2) << num_threads << "T"
              << " | " << std::fixed << std::setprecision(2) << std::setw(8) << max_dur_ms << " ms"
              << " | " << std::setw(7) << effective_gb_s << " GB/s"
              << " | " << std::setw(9) << (dots_per_sec / 1000.0) << " kdot/s"
              << " | " << std::setw(6) << (int)avg_cycles_per_dot << " cyc/dot"
              << " | " << std::setw(6) << (scaling_eff * 100.0) << "% eff"
              << std::endl;
}

int main() {
    std::cout << "========================================================================================\n";
    std::cout << "  PHASE 10 — Q5_K x Q8_K VEC_DOT ISOLATION BENCHMARK (N=" << N_DIM << ", Row=" << Q5_K_ROW_BYTES << "B)\n";
    std::cout << "========================================================================================\n";
    std::cout << std::left << std::setw(28) << "Configuration"
              << " | Thrd"
              << " | Duration   "
              << " | Eff GB/s"
              << " | Throughput "
              << " | Cycles  "
              << " | Scaling\n";
    std::cout << "----------------------------------------------------------------------------------------\n";

    const int ITERS_1T = 50000;
    const int ITERS_MT = 20000;

    TestResult base_hot_cont;
    run_vec_dot_benchmark("HOT contiguous", 1, false, true, false, ITERS_1T, 0.0, base_hot_cont);

    TestResult base_cold_cont;
    run_vec_dot_benchmark("COLD contiguous", 1, false, false, false, ITERS_1T, 0.0, base_cold_cont);

    TestResult base_hot_exp;
    run_vec_dot_benchmark("HOT real expert layout", 1, false, true, true, ITERS_1T, 0.0, base_hot_exp);

    TestResult base_cold_exp;
    run_vec_dot_benchmark("COLD real expert layout", 1, false, false, true, ITERS_1T, 0.0, base_cold_exp);

    std::cout << "----------------------------------------------------------------------------------------\n";
    std::cout << "=== 8 PHYSICAL THREADS ===\n";
    TestResult res_8p_hot_cont;
    run_vec_dot_benchmark("HOT contiguous", 8, false, true, false, ITERS_MT, base_hot_cont.vec_dots_per_sec, res_8p_hot_cont);

    TestResult res_8p_cold_cont;
    run_vec_dot_benchmark("COLD contiguous", 8, false, false, false, ITERS_MT, base_cold_cont.vec_dots_per_sec, res_8p_cold_cont);

    TestResult res_8p_hot_exp;
    run_vec_dot_benchmark("HOT real expert layout", 8, false, true, true, ITERS_MT, base_hot_exp.vec_dots_per_sec, res_8p_hot_exp);

    TestResult res_8p_cold_exp;
    run_vec_dot_benchmark("COLD real expert layout", 8, false, false, true, ITERS_MT, base_cold_exp.vec_dots_per_sec, res_8p_cold_exp);

    std::cout << "----------------------------------------------------------------------------------------\n";
    std::cout << "=== 16 SMT THREADS ===\n";
    TestResult res_16s_hot_cont;
    run_vec_dot_benchmark("HOT contiguous", 16, true, true, false, ITERS_MT, base_hot_cont.vec_dots_per_sec, res_16s_hot_cont);

    TestResult res_16s_cold_cont;
    run_vec_dot_benchmark("COLD contiguous", 16, true, false, false, ITERS_MT, base_cold_cont.vec_dots_per_sec, res_16s_cold_cont);

    TestResult res_16s_hot_exp;
    run_vec_dot_benchmark("HOT real expert layout", 16, true, true, true, ITERS_MT, base_hot_exp.vec_dots_per_sec, res_16s_hot_exp);

    TestResult res_16s_cold_exp;
    run_vec_dot_benchmark("COLD real expert layout", 16, true, false, true, ITERS_MT, base_cold_exp.vec_dots_per_sec, res_16s_cold_exp);

    std::cout << "========================================================================================\n";
    std::cout << "ANALYSIS & BOTTLENECK CLASSIFICATION:\n";
    double scaling_8p = res_8p_cold_exp.scaling_eff * 100.0;
    double scaling_16s = res_16s_cold_exp.scaling_eff * 100.0;
    std::cout << "  * 8 Physical Cores Scaling Efficiency: " << scaling_8p << "%\n";
    std::cout << "  * 16 SMT Threads Scaling Efficiency:   " << scaling_16s << "%\n";
    std::cout << "  * HOT vs COLD ratio (1T):              " << (base_hot_cont.vec_dots_per_sec / base_cold_cont.vec_dots_per_sec) << "x\n";
    std::cout << "  * Effective GB/s at 8P (Cold Expert):  " << res_8p_cold_exp.gb_per_sec << " GB/s (System ceiling: 52.2 GB/s)\n";
    if (res_8p_cold_exp.gb_per_sec < 25.0) {
        std::cout << "  * Classification: COMPUTE-BOUND / INSTRUCTION-THROUGHPUT BOUND (dequantization + SIMD dependency chains).\n";
        std::cout << "    DRAM bus is not saturated (effective bandwidth < ceiling). SMT contention severely degrades throughput.\n";
    } else {
        std::cout << "  * Classification: MEMORY / CACHE LATENCY BOUND.\n";
    }
    std::cout << "========================================================================================\n";

    return 0;
}
