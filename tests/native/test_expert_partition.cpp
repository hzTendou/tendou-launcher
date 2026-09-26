#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <vector>

static bool check(ggml_type type, int active, bool duplicate, bool broadcast = false, int pruned_count = 0, int n_tokens = 1, int heavy_expert = -1, int rows = 17) {
    auto * ctx = ggml_init({16 * 1024 * 1024, nullptr, true});
    const int columns = 256, experts = 12;
    auto * weights = ggml_new_tensor_3d(ctx, type, columns, rows, experts);
    const int inputs = broadcast ? 1 : active;
    auto * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, columns, inputs, n_tokens);
    auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, active, n_tokens);
    auto * output = ggml_mul_mat_id(ctx, weights, input, ids);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    auto buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
    auto backend = ggml_backend_cpu_init();
    std::vector<float> w(columns * rows * experts), x(columns * inputs * n_tokens);
    for (size_t i = 0; i < w.size(); ++i) w[i] = std::sin(float(i) * .017f);
    for (size_t i = 0; i < x.size(); ++i) x[i] = std::cos(float(i) * .019f);
    std::vector<unsigned char> packed(ggml_nbytes(weights));
    ggml_quantize_chunk(type, w.data(), packed.data(), 0, rows * experts, columns, nullptr);
    ggml_backend_tensor_set(weights, packed.data(), 0, packed.size());
    ggml_backend_tensor_set(input, x.data(), 0, x.size() * sizeof(float));
    std::vector<int> expert_ids(active * n_tokens);
    for (int t = 0; t < n_tokens; ++t) {
        for (int i = 0; i < active; ++i) {
            const int idx = t * active + i;
            if (i >= active - pruned_count) {
                expert_ids[idx] = -1;
            } else if (heavy_expert >= 0) {
                if (t < n_tokens * 4 / 5 && i == 0) {
                    expert_ids[idx] = heavy_expert;
                } else {
                    expert_ids[idx] = (i + t) % experts;
                }
            } else {
                expert_ids[idx] = duplicate ? 3 : ((i + t) % experts);
            }
        }
    }
    ggml_backend_tensor_set(ids, expert_ids.data(), 0, expert_ids.size() * sizeof(int));
    const size_t out_elems = (size_t)rows * active * n_tokens;
    std::vector<float> reference(out_elems), actual(out_elems);
    bool ok = true;
    for (int threads : {1, 2, 3, 8, 10, 14, 16}) {
        // A skipped output must not inherit a previous invocation's valid result.
        std::fill(actual.begin(), actual.end(), NAN);
        ggml_backend_tensor_set(output, actual.data(), 0, actual.size() * sizeof(float));
        ggml_backend_cpu_set_n_threads(backend, threads);
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return false;
        ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
        if (threads == 1) reference = actual;
        for (size_t i = 0; i < actual.size(); ++i) {
            if (!std::isfinite(actual[i]) || std::fabs(actual[i] - reference[i]) > 1e-5f) {
                std::printf("FAIL type=%s active=%d duplicate=%d threads=%d n_tok=%d row=%zu actual=%g reference=%g\n",
                    ggml_type_name(type), active, duplicate, threads, n_tokens, i, actual[i], reference[i]);
                ok = false;
                break;
            }
        }
    }
    ggml_backend_free(backend);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

int main() {
    bool ok = true;
    for (auto type : {GGML_TYPE_F32, GGML_TYPE_Q5_K, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0}) {
        ok = check(type, 10, false) && ok;
        ok = check(type, 10, false, true) && ok;
        ok = check(type, 1, false, true) && ok;
        ok = check(type, 2, true) && ok;
        // Test pruned expert configurations (active-expert load balancing)
        ok = check(type, 10, false, false, 5) && ok; // 5 out of 10 pruned
        ok = check(type, 5, false, false, 3) && ok;  // 3 out of 5 pruned (Clipgfy K=2)
        ok = check(type, 5, false, false, 2) && ok;  // 2 out of 5 pruned (Clipgfy K=3)
        // Multi-token batches with pruned experts (verifying multi-token robustness)
        ok = check(type, 10, false, false, 5, 2) && ok; // 2 tokens, 5/10 pruned
        ok = check(type, 5, false, false, 2, 4) && ok;  // 4 tokens, 2/5 pruned
        ok = check(type, 5, false, true, 3, 2) && ok;   // 2 tokens broadcast, 3/5 pruned
        // Test expert with > 64 tokens (verifying > 64 chunking boundary)
        ok = check(type, 2, true, false, 0, 70) && ok;   // 70 tokens duplicate
        ok = check(type, 10, false, false, 0, 70) && ok; // 70 tokens spread
        ok = check(type, 2, true, false, 0, 130) && ok;  // 130 tokens (2x 64 + 2 remainder)
        // Test extreme skew (heavy expert with 80% of traffic)
        ok = check(type, 8, false, false, 0, 50, 2) && ok; // 50 tokens, expert 2 heavy
        // Test row unrolling boundaries (8-row, 4-row remainder, scalar cleanup)
        ok = check(type, 5, false, false, 0, 2, -1, 4) && ok;  // 4 rows: 4-row loop only, max_t=2
        ok = check(type, 5, false, false, 0, 1, -1, 4) && ok;  // 4 rows: 4-row loop only, max_t=1
        ok = check(type, 5, false, false, 0, 3, -1, 12) && ok; // 12 rows: 8 + 4, max_t=3
        ok = check(type, 5, false, false, 0, 4, -1, 12) && ok; // 12 rows: 8 + 4, max_t=4
        ok = check(type, 5, false, false, 0, 2, -1, 21) && ok; // 21 rows: 8x2 + 4 + 1 scalar, max_t=2
        ok = check(type, 5, false, false, 0, 4, -1, 21) && ok; // 21 rows: 8x2 + 4 + 1 scalar, max_t=4
        ok = check(type, 5, false, false, 0, 1, -1, 22) && ok; // 22 rows: 8x2 + 4 + 2 scalar, max_t=1
    }
    std::puts(ok ? "PASS: every expert row matches the single-thread reference" : "FAIL: expert partition");
    return ok ? 0 : 1;
}
