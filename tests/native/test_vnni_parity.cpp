#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cassert>
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
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15
    };
    return _mm256_loadu_si256((const __m256i*)k_shuffle + i);
}

static inline float fp16_to_fp32(ggml_fp16_t x) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x)));
}

static inline ggml_fp16_t fp32_to_fp16(float x) {
    return (ggml_fp16_t)_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(x), 0), 0);
}

static void vec_dot_q5_K_q8_K_baseline(int n, float * s, const void * vx, const void * vy) {
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

static void vec_dot_q5_K_q8_K_vnni(int n, float * s, const void * vx, const void * vy) {
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
        const __m256i scales_32 = _mm256_cvtepu16_epi32(sc128);
        alignas(32) int32_t sc32_arr[8];
        _mm256_store_si256((__m256i*)sc32_arr, scales_32);

        const __m256i hbits = _mm256_loadu_si256((const __m256i*)x[i].qh);
        __m256i hmask = mone;

        __m512i sumi_512 = _mm512_setzero_si512();
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

            const __m512i q5_512 = _mm512_inserti64x4(_mm512_castsi256_si512(q5_0), q5_1, 1);
            const __m512i q8_512 = _mm512_loadu_si512((const __m512i*)q8); q8 += 64;

            const __m512i sc_512 = _mm512_inserti64x4(_mm512_castsi256_si512(scale_0), scale_1, 1);

            __m512i p16_512 = _mm512_maddubs_epi16(q5_512, q8_512);
            p16_512 = _mm512_madd_epi16(sc_512, p16_512);

            sumi_512 = _mm512_add_epi32(sumi_512, p16_512);
        }

        const __m256i sumi = _mm256_add_epi32(_mm512_castsi512_si256(sumi_512), _mm512_extracti64x4_epi64(sumi_512, 1));
        __m256 vd = _mm256_set1_ps(d);
        acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);
    }

    *s = hsum_float_8(acc) + summs;
}

int main() {
    constexpr int N = 2560;
    constexpr size_t Q5_K_ROW_BYTES = (N / 256) * sizeof(block_q5_K);
    constexpr size_t Q8_K_ROW_BYTES = (N / 256) * sizeof(block_q8_K);

    std::vector<uint8_t> q5_buf(Q5_K_ROW_BYTES);
    std::vector<uint8_t> q8_buf(Q8_K_ROW_BYTES);

    // Fill with non-trivial pattern
    for (size_t i = 0; i < q5_buf.size(); ++i) q5_buf[i] = (uint8_t)(i * 37 + 13);
    for (size_t i = 0; i < q8_buf.size(); ++i) q8_buf[i] = (uint8_t)(i * 59 + 7);

    block_q5_K* b5 = (block_q5_K*)q5_buf.data();
    for (int k = 0; k < N / 256; ++k) {
        b5[k].d = fp32_to_fp16(1.234f);
        b5[k].dmin = fp32_to_fp16(0.567f);
    }
    block_q8_K* b8 = (block_q8_K*)q8_buf.data();
    for (int k = 0; k < N / 256; ++k) {
        b8[k].d = 0.891f;
    }

    float s_base = 0.f;
    float s_vnni = 0.f;

    vec_dot_q5_K_q8_K_baseline(N, &s_base, q5_buf.data(), q8_buf.data());
    vec_dot_q5_K_q8_K_vnni(N, &s_vnni, q5_buf.data(), q8_buf.data());

    std::cout << "Baseline: " << s_base << "\n";
    std::cout << "VNNI:     " << s_vnni << "\n";
    std::cout << "Diff:     " << std::abs(s_base - s_vnni) << "\n";

    if (std::abs(s_base - s_vnni) < 1e-4f) {
        std::cout << "PASS: 100% Numerical Parity!\n";
    } else {
        std::cout << "FAIL: Parity Mismatch!\n";
        return 1;
    }

    // Benchmark loop
    const int ITERS = 100000;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < ITERS; ++it) {
        vec_dot_q5_K_q8_K_baseline(N, &s_base, q5_buf.data(), q8_buf.data());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_base = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto t2 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < ITERS; ++it) {
        vec_dot_q5_K_q8_K_vnni(N, &s_vnni, q5_buf.data(), q8_buf.data());
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double ms_vnni = std::chrono::duration<double, std::milli>(t3 - t2).count();

    std::cout << "Baseline time: " << ms_base << " ms (" << (ITERS / (ms_base / 1000.0) / 1000.0) << " kdot/s)\n";
    std::cout << "VNNI time:     " << ms_vnni << " ms (" << (ITERS / (ms_vnni / 1000.0) / 1000.0) << " kdot/s)\n";
    std::cout << "Speedup:       " << (ms_base / ms_vnni) << "x\n";

    return 0;
}
