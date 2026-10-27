with open("llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c", "r", encoding="utf-8") as f:
    text = f.read()

# Replace atomic counter and implement per-thread cache-line isolated slot counter
old_counter_decl = """static volatile int64_t s_vnni_nrc2_call_count = 0;

GGML_BACKEND_API int64_t ggml_vnni_nrc2_get_call_count(void) {
#if defined(_MSC_VER)
    return _InterlockedCompareExchange64((__int64 volatile *)&s_vnni_nrc2_call_count, 0, 0);
#else
    return __atomic_load_n(&s_vnni_nrc2_call_count, __ATOMIC_SEQ_CST);
#endif
}

GGML_BACKEND_API void ggml_vnni_nrc2_reset_call_count(void) {
#if defined(_MSC_VER)
    _InterlockedExchange64((__int64 volatile *)&s_vnni_nrc2_call_count, 0);
#else
    __atomic_store_n(&s_vnni_nrc2_call_count, 0, __ATOMIC_SEQ_CST);
#endif
}"""

new_counter_decl = """static volatile int64_t s_vnni_nrc2_call_count = 0;

#define VNNI_MAX_SLOTS 64
typedef struct {
    int64_t count;
    char pad[56]; // 64-byte cache line isolated
} vnni_counter_slot_t;

static vnni_counter_slot_t s_vnni_thread_slots[VNNI_MAX_SLOTS] = {0};

#if defined(_MSC_VER)
static __declspec(thread) int s_vnni_local_slot = -1;
#else
static _Thread_local int s_vnni_local_slot = -1;
#endif
static volatile long s_vnni_slot_alloc = 0;

static inline void record_vnni_call(int count) {
#if defined(ATLAS_TELEMETRY_ATOMIC)
    #if defined(_MSC_VER)
        _InterlockedExchangeAdd64((__int64 volatile *)&s_vnni_nrc2_call_count, count);
    #else
        __atomic_fetch_add(&s_vnni_nrc2_call_count, count, __ATOMIC_RELAXED);
    #endif
#else
    #if defined(_OPENMP)
    if (omp_in_parallel()) {
        int tid = omp_get_thread_num();
        if (tid >= 0 && tid < VNNI_MAX_SLOTS) {
            s_vnni_thread_slots[tid].count += count;
            return;
        }
    }
    #endif
    if (s_vnni_local_slot < 0) {
        #if defined(_MSC_VER)
        s_vnni_local_slot = (int)(_InterlockedIncrement(&s_vnni_slot_alloc) - 1) & (VNNI_MAX_SLOTS - 1);
        #else
        s_vnni_local_slot = (int)(__atomic_fetch_add(&s_vnni_slot_alloc, 1, __ATOMIC_RELAXED)) & (VNNI_MAX_SLOTS - 1);
        #endif
    }
    s_vnni_thread_slots[s_vnni_local_slot].count += count;
#endif
}

GGML_BACKEND_API int64_t ggml_vnni_nrc2_get_call_count(void) {
    int64_t total = 0;
#if defined(_MSC_VER)
    total += _InterlockedCompareExchange64((__int64 volatile *)&s_vnni_nrc2_call_count, 0, 0);
#else
    total += __atomic_load_n(&s_vnni_nrc2_call_count, __ATOMIC_SEQ_CST);
#endif
    for (int i = 0; i < VNNI_MAX_SLOTS; ++i) {
        total += s_vnni_thread_slots[i].count;
    }
    return total;
}

GGML_BACKEND_API void ggml_vnni_nrc2_reset_call_count(void) {
#if defined(_MSC_VER)
    _InterlockedExchange64((__int64 volatile *)&s_vnni_nrc2_call_count, 0);
#else
    __atomic_store_n(&s_vnni_nrc2_call_count, 0, __ATOMIC_SEQ_CST);
#endif
    for (int i = 0; i < VNNI_MAX_SLOTS; ++i) {
        s_vnni_thread_slots[i].count = 0;
    }
}"""

assert old_counter_decl in text, "old_counter_decl not found!"
text = text.replace(old_counter_decl, new_counter_decl, 1)

# Check assertion on nrc
old_check = """    assert(n % QK_K == 0);
    if (nrc != 1 && nrc != 2) {
        assert(nrc == 1 || nrc == 2);
        return;
    }"""

new_check = """    assert(n % QK_K == 0);
    if (nrc != 1 && nrc != 2 && nrc != 4) {
        assert(nrc == 1 || nrc == 2 || nrc == 4);
        return;
    }"""

assert old_check in text, "old_check not found!"
text = text.replace(old_check, new_check, 1)

# Replace the atomic increment in nrc == 2
old_nrc2_inc = """    // --- Multi-token path: nrc == 2 weight sharing ---
#if defined(_MSC_VER)
    _InterlockedIncrement64((__int64 volatile *)&s_vnni_nrc2_call_count);
#else
    __atomic_fetch_add(&s_vnni_nrc2_call_count, 1, __ATOMIC_RELAXED);
#endif"""

new_nrc4_and_nrc2 = """    // --- Multi-token path: nrc == 4 weight sharing (4 tokens with single weight decode) ---
    if (nrc == 4) {
        record_vnni_call(2);

        if (bs == 0) {
            bs = sizeof(float);
        }
        if (by == 0) {
            by = (n / QK_K) * sizeof(block_q8_K);
        }

        const block_iq3_xxs * GGML_RESTRICT x  = (const block_iq3_xxs *)vx;
        const block_q8_K    * GGML_RESTRICT y0 = (const block_q8_K *)((const char *)vy + 0 * by);
        const block_q8_K    * GGML_RESTRICT y1 = (const block_q8_K *)((const char *)vy + 1 * by);
        const block_q8_K    * GGML_RESTRICT y2 = (const block_q8_K *)((const char *)vy + 2 * by);
        const block_q8_K    * GGML_RESTRICT y3 = (const block_q8_K *)((const char *)vy + 3 * by);

        float * GGML_RESTRICT s0 = (float *)((char *)s + 0 * bs);
        float * GGML_RESTRICT s1 = (float *)((char *)s + 1 * bs);
        float * GGML_RESTRICT s2 = (float *)((char *)s + 2 * bs);
        float * GGML_RESTRICT s3 = (float *)((char *)s + 3 * bs);

        const int nb = n / QK_K;

#if defined(__AVX512F__) && (defined(__AVX512VNNI__) || defined(__AVXVNNI__))
        uint32_t aux32_0[2];
        uint32_t aux32_1[2];

        __m512 accumf0 = _mm512_setzero_ps();
        __m512 accumf1 = _mm512_setzero_ps();
        __m512 accumf2 = _mm512_setzero_ps();
        __m512 accumf3 = _mm512_setzero_ps();

        for (int i = 0; i < nb; ++i) {
            const float d0 = GGML_CPU_FP16_TO_FP32(x[i].d) * y0[i].d;
            const float d1 = GGML_CPU_FP16_TO_FP32(x[i].d) * y1[i].d;
            const float d2 = GGML_CPU_FP16_TO_FP32(x[i].d) * y2[i].d;
            const float d3 = GGML_CPU_FP16_TO_FP32(x[i].d) * y3[i].d;

            const uint8_t * GGML_RESTRICT q3 = x[i].qs;
            const uint8_t * GGML_RESTRICT gas = x[i].qs + QK_K/4;
            const int8_t  * GGML_RESTRICT q8_0 = y0[i].qs;
            const int8_t  * GGML_RESTRICT q8_1 = y1[i].qs;
            const int8_t  * GGML_RESTRICT q8_2 = y2[i].qs;
            const int8_t  * GGML_RESTRICT q8_3 = y3[i].qs;

            __m512i sumi0_t0 = _mm512_setzero_si512();
            __m512i sumi1_t0 = _mm512_setzero_si512();
            __m512i sumi0_t1 = _mm512_setzero_si512();
            __m512i sumi1_t1 = _mm512_setzero_si512();
            __m512i sumi0_t2 = _mm512_setzero_si512();
            __m512i sumi1_t2 = _mm512_setzero_si512();
            __m512i sumi0_t3 = _mm512_setzero_si512();
            __m512i sumi1_t3 = _mm512_setzero_si512();

            for (int ib64 = 0; ib64 < QK_K/64; ib64 += 2) {
                // Unpack weights once for all 4 tokens:
                const __m256i q2_1_0 = _mm256_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]],
                                                        iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
                q3 += 8;
                const __m256i q2_2_0 = _mm256_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]],
                                                        iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
                q3 += 8;
                const __m512i q2_val0 = _mm512_inserti64x4(_mm512_castsi256_si512(q2_1_0), q2_2_0, 1);

                memcpy(aux32_0, gas, 8); gas += 8;
                const uint64_t mask0 = 
                    ((uint64_t)k_even_sign_mask[(aux32_0[0] >>  0) & 127] <<  0) |
                    ((uint64_t)k_even_sign_mask[(aux32_0[0] >>  7) & 127] <<  8) |
                    ((uint64_t)k_even_sign_mask[(aux32_0[0] >> 14) & 127] << 16) |
                    ((uint64_t)k_even_sign_mask[(aux32_0[0] >> 21) & 127] << 24) |
                    ((uint64_t)k_even_sign_mask[(aux32_0[1] >>  0) & 127] << 32) |
                    ((uint64_t)k_even_sign_mask[(aux32_0[1] >>  7) & 127] << 40) |
                    ((uint64_t)k_even_sign_mask[(aux32_0[1] >> 14) & 127] << 48) |
                    ((uint64_t)k_even_sign_mask[(aux32_0[1] >> 21) & 127] << 56);
                const __mmask64 neg_mask0 = (__mmask64)mask0;

                const uint16_t ls1_0 = aux32_0[0] >> 28;
                const uint16_t ls2_0 = aux32_0[1] >> 28;
                const int32_t scale1_0 = 2 * ls1_0 + 1;
                const int32_t scale2_0 = 2 * ls2_0 + 1;
                const __m512i scale_factor0 = _mm512_mask_set1_epi32(_mm512_set1_epi32(scale1_0), 0xFF00, scale2_0);

                // Token 0
                const __m512i q8_val0_t0 = _mm512_loadu_si512((const void *)q8_0); q8_0 += 64;
                const __m512i q8s0_t0 = _mm512_mask_sub_epi8(q8_val0_t0, neg_mask0, _mm512_setzero_si512(), q8_val0_t0);
                const __m512i dot0_t0 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q2_val0, q8s0_t0);
                sumi0_t0 = _mm512_add_epi32(sumi0_t0, _mm512_mullo_epi32(dot0_t0, scale_factor0));

                // Token 1
                const __m512i q8_val0_t1 = _mm512_loadu_si512((const void *)q8_1); q8_1 += 64;
                const __m512i q8s0_t1 = _mm512_mask_sub_epi8(q8_val0_t1, neg_mask0, _mm512_setzero_si512(), q8_val0_t1);
                const __m512i dot0_t1 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q2_val0, q8s0_t1);
                sumi0_t1 = _mm512_add_epi32(sumi0_t1, _mm512_mullo_epi32(dot0_t1, scale_factor0));

                // Token 2
                const __m512i q8_val0_t2 = _mm512_loadu_si512((const void *)q8_2); q8_2 += 64;
                const __m512i q8s0_t2 = _mm512_mask_sub_epi8(q8_val0_t2, neg_mask0, _mm512_setzero_si512(), q8_val0_t2);
                const __m512i dot0_t2 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q2_val0, q8s0_t2);
                sumi0_t2 = _mm512_add_epi32(sumi0_t2, _mm512_mullo_epi32(dot0_t2, scale_factor0));

                // Token 3
                const __m512i q8_val0_t3 = _mm512_loadu_si512((const void *)q8_3); q8_3 += 64;
                const __m512i q8s0_t3 = _mm512_mask_sub_epi8(q8_val0_t3, neg_mask0, _mm512_setzero_si512(), q8_val0_t3);
                const __m512i dot0_t3 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q2_val0, q8s0_t3);
                sumi0_t3 = _mm512_add_epi32(sumi0_t3, _mm512_mullo_epi32(dot0_t3, scale_factor0));

                // Branch 1 (second 64 elements)
                const __m256i q2_1_1 = _mm256_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]],
                                                        iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
                q3 += 8;
                const __m256i q2_2_1 = _mm256_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]],
                                                        iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
                q3 += 8;
                const __m512i q2_val1 = _mm512_inserti64x4(_mm512_castsi256_si512(q2_1_1), q2_2_1, 1);

                memcpy(aux32_1, gas, 8); gas += 8;
                const uint64_t mask1 = 
                    ((uint64_t)k_even_sign_mask[(aux32_1[0] >>  0) & 127] <<  0) |
                    ((uint64_t)k_even_sign_mask[(aux32_1[0] >>  7) & 127] <<  8) |
                    ((uint64_t)k_even_sign_mask[(aux32_1[0] >> 14) & 127] << 16) |
                    ((uint64_t)k_even_sign_mask[(aux32_1[0] >> 21) & 127] << 24) |
                    ((uint64_t)k_even_sign_mask[(aux32_1[1] >>  0) & 127] << 32) |
                    ((uint64_t)k_even_sign_mask[(aux32_1[1] >>  7) & 127] << 40) |
                    ((uint64_t)k_even_sign_mask[(aux32_1[1] >> 14) & 127] << 48) |
                    ((uint64_t)k_even_sign_mask[(aux32_1[1] >> 21) & 127] << 56);
                const __mmask64 neg_mask1 = (__mmask64)mask1;

                const uint16_t ls1_1 = aux32_1[0] >> 28;
                const uint16_t ls2_1 = aux32_1[1] >> 28;
                const int32_t scale1_1 = 2 * ls1_1 + 1;
                const int32_t scale2_1 = 2 * ls2_1 + 1;
                const __m512i scale_factor1 = _mm512_mask_set1_epi32(_mm512_set1_epi32(scale1_1), 0xFF00, scale2_1);

                // Token 0
                const __m512i q8_val1_t0 = _mm512_loadu_si512((const void *)q8_0); q8_0 += 64;
                const __m512i q8s1_t0 = _mm512_mask_sub_epi8(q8_val1_t0, neg_mask1, _mm512_setzero_si512(), q8_val1_t0);
                const __m512i dot1_t0 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q2_val1, q8s1_t0);
                sumi1_t0 = _mm512_add_epi32(sumi1_t0, _mm512_mullo_epi32(dot1_t0, scale_factor1));

                // Token 1
                const __m512i q8_val1_t1 = _mm512_loadu_si512((const void *)q8_1); q8_1 += 64;
                const __m512i q8s1_t1 = _mm512_mask_sub_epi8(q8_val1_t1, neg_mask1, _mm512_setzero_si512(), q8_val1_t1);
                const __m512i dot1_t1 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q2_val1, q8s1_t1);
                sumi1_t1 = _mm512_add_epi32(sumi1_t1, _mm512_mullo_epi32(dot1_t1, scale_factor1));

                // Token 2
                const __m512i q8_val1_t2 = _mm512_loadu_si512((const void *)q8_2); q8_2 += 64;
                const __m512i q8s1_t2 = _mm512_mask_sub_epi8(q8_val1_t2, neg_mask1, _mm512_setzero_si512(), q8_val1_t2);
                const __m512i dot1_t2 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q2_val1, q8s1_t2);
                sumi1_t2 = _mm512_add_epi32(sumi1_t2, _mm512_mullo_epi32(dot1_t2, scale_factor1));

                // Token 3
                const __m512i q8_val1_t3 = _mm512_loadu_si512((const void *)q8_3); q8_3 += 64;
                const __m512i q8s1_t3 = _mm512_mask_sub_epi8(q8_val1_t3, neg_mask1, _mm512_setzero_si512(), q8_val1_t3);
                const __m512i dot1_t3 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q2_val1, q8s1_t3);
                sumi1_t3 = _mm512_add_epi32(sumi1_t3, _mm512_mullo_epi32(dot1_t3, scale_factor1));
            }

            const __m512i combined_sum0 = _mm512_add_epi32(sumi0_t0, sumi1_t0);
            accumf0 = _mm512_fmadd_ps(_mm512_set1_ps(d0), _mm512_cvtepi32_ps(combined_sum0), accumf0);

            const __m512i combined_sum1 = _mm512_add_epi32(sumi0_t1, sumi1_t1);
            accumf1 = _mm512_fmadd_ps(_mm512_set1_ps(d1), _mm512_cvtepi32_ps(combined_sum1), accumf1);

            const __m512i combined_sum2 = _mm512_add_epi32(sumi0_t2, sumi1_t2);
            accumf2 = _mm512_fmadd_ps(_mm512_set1_ps(d2), _mm512_cvtepi32_ps(combined_sum2), accumf2);

            const __m512i combined_sum3 = _mm512_add_epi32(sumi0_t3, sumi1_t3);
            accumf3 = _mm512_fmadd_ps(_mm512_set1_ps(d3), _mm512_cvtepi32_ps(combined_sum3), accumf3);
        }

        const __m256 acc256_0 = _mm256_add_ps(_mm512_castps512_ps256(accumf0), _mm512_extractf32x8_ps(accumf0, 1));
        *s0 = 0.25f * hsum_float_8(acc256_0);

        const __m256 acc256_1 = _mm256_add_ps(_mm512_castps512_ps256(accumf1), _mm512_extractf32x8_ps(accumf1, 1));
        *s1 = 0.25f * hsum_float_8(acc256_1);

        const __m256 acc256_2 = _mm256_add_ps(_mm512_castps512_ps256(accumf2), _mm512_extractf32x8_ps(accumf2, 1));
        *s2 = 0.25f * hsum_float_8(acc256_2);

        const __m256 acc256_3 = _mm256_add_ps(_mm512_castps512_ps256(accumf3), _mm512_extractf32x8_ps(accumf3, 1));
        *s3 = 0.25f * hsum_float_8(acc256_3);

#else
        ggml_vec_dot_iq3_xxs_q8_K_vnni(n, s0, bs, vx, 0, (const char *)vy + 0 * by, by, 2);
        ggml_vec_dot_iq3_xxs_q8_K_vnni(n, s2, bs, vx, 0, (const char *)vy + 2 * by, by, 2);
#endif
        return;
    }

    // --- Multi-token path: nrc == 2 weight sharing ---
    record_vnni_call(1);"""

assert old_nrc2_inc in text, "old_nrc2_inc not found!"
text = text.replace(old_nrc2_inc, new_nrc4_and_nrc2, 1)

with open("llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c", "w", encoding="utf-8", newline="\n") as f:
    f.write(text)

print("SUCCESS: patched quants.c for atomic contention & nrc=4")
