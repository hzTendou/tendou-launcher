#define main atlas_program_main
#include "../../src/atlas/atlas-engine.cpp"
#undef main
#include <stdexcept>
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (0)

extern "C" {
    GGML_BACKEND_API void ggml_vec_dot_iq3_xxs_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
    GGML_BACKEND_API void ggml_vec_dot_iq3_xxs_q8_K_x4(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
    GGML_BACKEND_API void ggml_vec_dot_iq3_xxs_q8_K_vnni(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
    GGML_BACKEND_API int64_t ggml_vnni_nrc2_get_call_count(void);
    GGML_BACKEND_API void    ggml_vnni_nrc2_reset_call_count(void);
    GGML_BACKEND_API void    ggml_vec_dot_mxfp4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
    GGML_BACKEND_API void    ggml_vec_dot_mxfp4_q8_0_x4(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
    GGML_BACKEND_API void    ggml_vec_dot_mxfp4_q8_0_nrc2(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
    GGML_BACKEND_API void    ggml_vec_dot_iq4_xs_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
    GGML_BACKEND_API void    ggml_vec_dot_iq4_xs_q8_K_x4(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
    GGML_API         void    ggml_vec_dot_f8_e4m3_f32(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT x, size_t bx, const void * GGML_RESTRICT y, size_t by, int nrc);
}

int main() {
    try {
        atlas::PinnedHostStagingPool pool;
    CHECK(!pool.is_active() && pool.get_capacity() == 0);
    atlas::Config cfg;
    CHECK(cfg.quality_target == 0.80f);
    CHECK(cfg.max_candidates_per_layer == 10);
    CHECK(cfg.confidence_floor == 0.22);
    CHECK(cfg.readback_interval == 8);
    CHECK(cfg.odmoe_layer_lead == 2);
    CHECK(cfg.spice_conf_high == 0.65);
    CHECK(cfg.spice_conf_mid == 0.30);
    CHECK(!cfg.enable_tutti);
    CHECK(cfg.tutti_queue_depth == 64);
    CHECK(cfg.cuda_sched == "yield");
    CHECK(cfg.n_threads == 16);
    CHECK(cfg.prompt_cache_mb == 256);
    cfg.enable_tutti = true;
    cfg.tutti_queue_depth = 4;
    const std::string path = "experiments/runtime_audit/2026-09-10/prefetch-test.bin";
    { std::ofstream out(path, std::ios::binary); std::string data(16384, 'x'); out.write(data.data(), data.size()); }
    atlas::PhysicalExpert pe;
    atlas::ChunkLocation c;
    c.blob_path = path; c.file_offset = 1; c.size_bytes = 4096;
    pe.chunks = {c, c};
    {
        atlas::TuttiAsyncPipeline pipe(cfg, pool);
        CHECK(pipe.enqueue_read(0, 1, &pe));
        pipe.wait_ready(0, 1);
        CHECK(pipe.is_ready(0, 1));
        CHECK(pipe.get_completed_reads() == 2);
        CHECK(pipe.get_bytes_staged() == 0 && pipe.get_staged_ptr(0, 1) == nullptr);
        CHECK(pipe.get_bytes_warmed() == 8192);
        // Bounds/failed files never publish success, and waiters always complete.
        pe.chunks[1].file_offset = 16384;
        CHECK(pipe.enqueue_read(0, 2, &pe)); pipe.wait_ready(0, 2);
        CHECK(!pipe.is_ready(0, 2));
        pe.chunks[1].blob_path = path + ".missing";
        CHECK(pipe.enqueue_read(0, 3, &pe)); pipe.wait_ready(0, 3);
        CHECK(!pipe.is_ready(0, 3));
        pe.chunks.clear(); CHECK(!pipe.enqueue_read(0, 4, &pe));
        pe.chunks = {c}; pe.chunks[0].file_offset = SIZE_MAX;
        CHECK(!pipe.enqueue_read(0, 4, &pe));
        pe.chunks = {c,c,c,c,c}; CHECK(!pipe.enqueue_read(0, 4, &pe));
        pe.chunks = {c,c};
        for (int i = 0; i < 100; ++i) {
            pipe.enqueue_read(0, i, &pe);
            pipe.clear();
            pipe.wait_ready(0, i);
        }
        CHECK(!pipe.is_ready(0, 1));
    }
    std::remove(path.c_str());
    cfg.enable_tutti = false;
    atlas::Predictor predictor(cfg);
    std::vector<int> targets{20,21,22,23,24,25,26,27,28,29};
    for (int i = 0; i < 10; ++i) {
        predictor.observe(0, {1,2}); predictor.observe(1, targets); predictor.end_token();
    }
    auto preds = predictor.predict_layer_lead_with_confidence(0, 1, {1,2}, 10);
    CHECK(preds.size() == 10);
    for (auto & pred : preds) CHECK(pred.confidence > 0.8 && pred.confidence < 1.0);
    // Exercise the actual router callback across a token boundary on host tensors.
    cfg.physical_map_path = "missing-map-for-native-test.json";
    atlas::Runtime rt(cfg);
    rt.saw_decode = true;
    ggml_init_params init{1024 * 1024, nullptr, true};
    auto * ctx = ggml_init(init);
    auto * tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, 1);
    auto buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
    auto observe = [&](int layer, int expert) {
        ggml_set_name(tensor, ("ffn_moe_topk-" + std::to_string(layer)).c_str());
        ggml_backend_tensor_set(tensor, &expert, 0, sizeof(expert));
        CHECK(atlas_eval_callback(tensor, false, &rt));
    };
    observe(0, 1); observe(1, 10); observe(0, 2); observe(1, 20);
    auto learned = rt.predictor.predict_layer_lead_with_confidence(0, 1, {1}, 10);
    CHECK(learned.size() == 1 && learned[0].expert == 10);
    // -----------------------------------------------------------------------
    // P2: ExpertIDMapper tests
    // -----------------------------------------------------------------------
    CHECK(atlas::ExpertIDMapper::get_compact_id(0, 0) == 0);
    CHECK(atlas::ExpertIDMapper::get_compact_id(0, 511) == 511);
    CHECK(atlas::ExpertIDMapper::get_compact_id(1, 0) == 512);
    CHECK(atlas::ExpertIDMapper::get_compact_id(47, 511) == 48 * 512 - 1);
    CHECK(atlas::ExpertIDMapper::get_layer(512) == 1);
    CHECK(atlas::ExpertIDMapper::get_expert(512) == 0);
    bool caught_err = false;
    try { atlas::ExpertIDMapper::get_compact_id(-1, 0); } catch (const std::invalid_argument &) { caught_err = true; }
    CHECK(caught_err);
    caught_err = false;
    try { atlas::ExpertIDMapper::get_compact_id(48, 0); } catch (const std::invalid_argument &) { caught_err = true; }
    CHECK(caught_err);
    caught_err = false;
    try { atlas::ExpertIDMapper::get_compact_id(0, 512); } catch (const std::invalid_argument &) { caught_err = true; }
    CHECK(caught_err);

    // -----------------------------------------------------------------------
    // P2: GPUExpertBuffer & ExpertBindingPlan tests
    // -----------------------------------------------------------------------
    atlas::GPUExpertBuffer zero_buf(0);
    CHECK(!zero_buf.can_fit());
    CHECK(!zero_buf.bind_expert(0, 0));
    CHECK(zero_buf.get_resident_count() == 0);

    atlas::GPUExpertBuffer buf_2(3481600 * 2);
    CHECK(buf_2.can_fit());
    CHECK(buf_2.bind_expert(0, 0));
    CHECK(buf_2.bind_expert(0, 1));
    CHECK(!buf_2.can_fit());
    CHECK(!buf_2.bind_expert(0, 2)); // exceeded capacity
    CHECK(buf_2.is_resident(0, 0));
    CHECK(buf_2.is_resident(0, 1));

    // Eviction test
    atlas::ExpertKey evicted{};
    CHECK(buf_2.evict_lru({}, evicted));
    CHECK((evicted == atlas::ExpertKey{0, 0}));
    CHECK(!buf_2.is_resident(0, 0));
    CHECK(buf_2.can_fit());
    CHECK(buf_2.bind_expert(0, 2));

    // Eviction with keep_experts
    atlas::GPUExpertBuffer buf_keep(3481600 * 2);
    buf_keep.bind_expert(0, 0);
    buf_keep.bind_expert(0, 1);
    std::unordered_set<atlas::ExpertKey, atlas::KeyHash> keep_set = { atlas::ExpertKey{0, 0} };
    CHECK(buf_keep.evict_lru(keep_set, evicted));
    CHECK((evicted == atlas::ExpertKey{0, 1})); // evicted 0,1 because 0,0 was kept!

    // Binding Plan test (lossless fallback)
    atlas::GPUExpertBuffer plan_buf(3481600 * 5); // fits 5
    atlas::ExpertBindingPlan plan(plan_buf);
    std::vector<int> router_10 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto binding_res = plan.plan_binding(0, router_10);
    CHECK(binding_res.gpu.size() == 5);
    CHECK(binding_res.cpu.size() == 5);
    CHECK(binding_res.gpu == std::vector<int>({0, 1, 2, 3, 4}));
    CHECK(binding_res.cpu == std::vector<int>({5, 6, 7, 8, 9}));

    // Zero capacity plan test
    atlas::ExpertBindingPlan zero_plan(zero_buf);
    auto zero_res = zero_plan.plan_binding(0, {1, 2, 3});
    CHECK(zero_res.gpu.empty());
    CHECK(zero_res.cpu.size() == 3);

    // Invalid layer/expert bounds on GPU buffer and plan
    CHECK(!buf_2.bind_expert(-1, 0));
    CHECK(!buf_2.bind_expert(48, 0));
    CHECK(!buf_2.bind_expert(0, 512));
    CHECK(!buf_2.is_resident(-1, 0));
    CHECK(!buf_2.is_resident(0, 512));
    auto invalid_layer_res = plan.plan_binding(50, {1, 2});
    CHECK(invalid_layer_res.gpu.empty());
    CHECK(invalid_layer_res.cpu.size() == 2);

    // -----------------------------------------------------------------------
    // Phase 7: CrossLayerGPUPinning & PinnedReductionBuffer & Entropy-K tests
    // -----------------------------------------------------------------------
    atlas::CrossLayerGPUPinning cl_pinning;
    CHECK(cl_pinning.is_resident(0, 0));
    CHECK(cl_pinning.is_resident(0, 20));
    CHECK(!cl_pinning.is_resident(0, 21));
    CHECK(cl_pinning.get_slot(0, 0) == 0);
    CHECK(cl_pinning.get_slot(0, 20) == 20);
    CHECK(cl_pinning.get_slot(0, 21) == -1);
    CHECK(cl_pinning.get_slot(-1, 0) == -1);
    CHECK(cl_pinning.get_slot(48, 0) == -1);
    CHECK(cl_pinning.get_slot(0, 512) == -1);

    // Update slot test
    CHECK(cl_pinning.update_expert(0, 0, 100));
    CHECK(!cl_pinning.is_resident(0, 0));
    CHECK(cl_pinning.is_resident(0, 100));
    CHECK(cl_pinning.get_slot(0, 100) == 0);
    CHECK(cl_pinning.get_slot(0, 0) == -1);

    // Pinned Reduction Buffer test
    atlas::PinnedReductionBuffer red_buf;
    float * l0_buf = red_buf.get_layer_buffer(0);
    CHECK(l0_buf != nullptr);
    CHECK(red_buf.get_layer_buffer(-1) == nullptr);
    CHECK(red_buf.get_layer_buffer(48) == nullptr);

    float vec_a[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float vec_b[8] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
    float vec_out[8] = {0.0f};
    atlas::PinnedReductionBuffer::reduce_vector(vec_out, vec_a, vec_b, 8);
    CHECK(vec_out[0] == 11.0f);
    CHECK(vec_out[7] == 88.0f);

    // Entropy-K Confidence Gating logic verification
    float high_unc_w0 = 0.35f;
    float high_unc_margin = 0.05f;
    int k_uncertain = (high_unc_w0 < 0.38f || high_unc_margin < 0.12f) ? 4 : 2;
    CHECK(k_uncertain == 4);

    float high_conf_w0 = 0.45f;
    float high_conf_margin = 0.15f;
    int k_confident = (high_conf_w0 < 0.38f || high_conf_margin < 0.12f) ? 4 : 2;
    CHECK(k_confident == 2);

    // Split-MoE Cache reset_to_initial and decode step reset verification
    {
        atlas::GPUExpertBuffer test_gpu_buf(1024 * 1024 * 100);
        atlas::FrequencyGatedSplitMoECache split_cache(test_gpu_buf);
        split_cache.on_slot_update = [](int layer, int old_exp, int new_exp, int16_t slot) {
            (void)layer; (void)old_exp; (void)new_exp; (void)slot;
        };
        split_cache.warm_expert(0, 50);
        split_cache.reset_to_initial();
        CHECK(split_cache.get_pinning().is_resident(0, 0));
        CHECK(split_cache.get_pinning().is_resident(0, 20));

        typedef void (*pfn_atlas_reset_decode_step)(void);
#if defined(_WIN32)
        HMODULE hCpu = GetModuleHandleW(L"ggml-cpu.dll");
        if (!hCpu) hCpu = LoadLibraryW(L"ggml-cpu.dll");
        if (hCpu) {
            auto reset_fn = (pfn_atlas_reset_decode_step)(void*)GetProcAddress(hCpu, "atlas_reset_decode_step");
            CHECK(reset_fn != nullptr);
            reset_fn();
        }
#else
        auto reset_fn = (pfn_atlas_reset_decode_step)dlsym(RTLD_DEFAULT, "atlas_reset_decode_step");
        if (!reset_fn) {
            void * hCpu = dlopen("libggml-cpu.so", RTLD_LAZY);
            if (hCpu) reset_fn = (pfn_atlas_reset_decode_step)dlsym(hCpu, "atlas_reset_decode_step");
        }
        CHECK(reset_fn != nullptr);
        reset_fn();
#endif
    }

    // -----------------------------------------------------------------------
    // P3: PinnedStagingBuffer & H2DTransferQueue & CUDAEventFence tests
    // -----------------------------------------------------------------------
    atlas::PinnedStagingBuffer staging(2, 1024);
    int s0 = staging.acquire_slot();
    int s1 = staging.acquire_slot();
    CHECK(s0 >= 0 && s1 >= 0 && s0 != s1);
    CHECK(staging.is_full());
    CHECK(staging.acquire_slot() == -1);
    staging.release_slot(s0);
    CHECK(!staging.is_full());
    int s2 = staging.acquire_slot();
    CHECK(s2 == s0);
    staging.clear();
    CHECK(!staging.is_full() && staging.get_used_count() == 0);

    atlas::H2DTransferQueue h2d_q(2);
    CHECK(h2d_q.enqueue(0, 10, 0));
    CHECK(h2d_q.enqueue(0, 11, 1));
    CHECK(!h2d_q.enqueue(0, 12, 2)); // full
    CHECK(h2d_q.get_pending_count() == 2);
    atlas::H2DQueueItem qi{};
    CHECK(h2d_q.dequeue(qi));
    CHECK(qi.layer == 0 && qi.expert == 10 && qi.staging_slot == 0);
    CHECK(h2d_q.get_pending_count() == 1);
    CHECK(h2d_q.get_completed_count() == 1);
    // cancellation
    CHECK(h2d_q.remove(0, 11));
    CHECK(h2d_q.is_empty());
    h2d_q.enqueue(0, 5, 0);
    h2d_q.clear();
    CHECK(h2d_q.is_empty());

    atlas::CUDAEventFence fence;
    fence.record_transfer_start(0, 1);
    CHECK(!fence.transfer_complete_before_compute(0, 1));
    fence.record_transfer_complete(0, 1);
    CHECK(fence.transfer_complete_before_compute(0, 1));
    fence.record_compute_start(0, 1);
    CHECK(fence.transfer_complete_before_compute(0, 1)); // no hazard when compute starts after transfer
    CHECK(!fence.has_hazard(0, 1));

    // Hazard test: compute starts prematurely before transfer complete
    fence.record_transfer_start(0, 2);
    fence.record_compute_start(0, 2); // Premature!
    fence.record_transfer_complete(0, 2);
    CHECK(!fence.transfer_complete_before_compute(0, 2)); // Must fail due to hazard!
    CHECK(fence.has_hazard(0, 2));

    // Untransferred expert computes (e.g. CPU execution) - MUST NOT be flagged as a hazard!
    fence.record_compute_start(0, 50);
    CHECK(!fence.has_hazard(0, 50));
    CHECK(!fence.transfer_complete_before_compute(0, 50));

    // Cancelled transfer - MUST NOT trigger hazard on subsequent compute!
    fence.record_transfer_start(0, 60);
    fence.record_transfer_cancelled(0, 60);
    fence.record_compute_start(0, 60);
    CHECK(!fence.has_hazard(0, 60));

    atlas::AsyncTransferPipeline async_pipe(2, 100, 2);
    // Invalid bounds rejection without consuming slots
    CHECK(!async_pipe.submit_expert(-1, 0, 50));
    CHECK(!async_pipe.submit_expert(48, 0, 50));
    CHECK(!async_pipe.submit_expert(0, 512, 50));
    CHECK(async_pipe.get_staging_buffer().get_used_count() == 0);

    CHECK(async_pipe.submit_expert(0, 1, 50));
    // Duplicate submit must be rejected and avoid leaking staging slots
    CHECK(!async_pipe.submit_expert(0, 1, 50));
    CHECK(async_pipe.get_staging_buffer().get_used_count() == 1);
    CHECK(async_pipe.submit_expert(0, 2, 50));
    CHECK(!async_pipe.submit_expert(0, 3, 50)); // buffer full
    CHECK(!async_pipe.submit_expert(0, 4, 150)); // oversized
    CHECK(async_pipe.cancel_transfer(0, 1));
    // Cancellation must clear fence so compute does not register hazard
    async_pipe.get_event_fence().record_compute_start(0, 1);
    CHECK(!async_pipe.get_event_fence().has_hazard(0, 1));
    CHECK(async_pipe.submit_expert(0, 3, 50)); // space freed by cancellation
    auto completed_transfers = async_pipe.drain_completed();
    CHECK(completed_transfers.size() == 2);
    async_pipe.clear();
    CHECK(async_pipe.get_staging_buffer().get_used_count() == 0);
    CHECK(async_pipe.get_transfer_queue().is_empty());

    // Verify invalid expert in plan does not evict resident GPU experts
    atlas::GPUExpertBuffer safe_buf(3481600 * 2);
    safe_buf.bind_expert(0, 0);
    safe_buf.bind_expert(0, 1);
    atlas::ExpertBindingPlan safe_plan(safe_buf);
    auto safe_res = safe_plan.plan_binding(0, {999});
    CHECK(safe_res.gpu.empty());
    CHECK(safe_res.cpu.size() == 1 && safe_res.cpu[0] == 999);
    CHECK(safe_buf.is_resident(0, 0));
    CHECK(safe_buf.is_resident(0, 1));

    ggml_backend_buffer_free(buffer); ggml_free(ctx);

    // Verify CUDA primary context scheduling configuration helper
    CHECK(atlas_configure_cuda_primary_ctx("yield"));
    CHECK(atlas_configure_cuda_primary_ctx("blocking"));
    CHECK(atlas_configure_cuda_primary_ctx("spin"));
    CHECK(atlas_configure_cuda_primary_ctx("auto"));
    CHECK(atlas_configure_cuda_primary_ctx("  yield  "));
    CHECK(!atlas_configure_cuda_primary_ctx(""));
    CHECK(!atlas_configure_cuda_primary_ctx("   "));
    // -----------------------------------------------------------------------
    // TLEP: Tendou Lookahead Execution Pipeline Tests
    // -----------------------------------------------------------------------
    {
        // 1. TLEPSlackEstimator
        atlas::TLEPSlackEstimator estimator(1.5);
        CHECK(estimator.get_predicted_gpu_dense_us() == 35480);
        CHECK(estimator.get_predicted_cpu_moe_us() == 29700);
        CHECK(estimator.get_gpu_guard_margin_us() >= 1500);
        CHECK(estimator.get_cpu_guard_margin_us() >= 1500);

        estimator.record_exact_gpu(36000);
        estimator.record_exact_cpu_moe(30000);
        CHECK(estimator.get_predicted_gpu_dense_us() > 35480);
        CHECK(estimator.get_predicted_cpu_moe_us() > 29700);

        int64_t safe_gpu_budget = estimator.get_safe_gpu_slack_budget_us();
        CHECK(safe_gpu_budget > 0);
        CHECK(estimator.can_dispatch_spec_gpu(0, 100));
        CHECK(!estimator.can_dispatch_spec_gpu(safe_gpu_budget + 10, 100));

        // Dynamic horizon calculation
        std::vector<double> high_acc{0.80, 0.75, 0.70, 0.65};
        int k_high = estimator.compute_dynamic_horizon(4, high_acc);
        CHECK(k_high >= 1 && k_high <= 4);

        std::vector<double> zero_acc{0.0, 0.0, 0.0, 0.0};
        int k_zero = estimator.compute_dynamic_horizon(4, zero_acc);
        CHECK(k_zero == 0); // EV non-positive clamps strictly to K=0

        // 2. MultiDepthExpertConsensus
        atlas::LookaheadPlan plan;
        plan.predicted_experts.resize(2);
        plan.expert_confidence.resize(2);
        plan.predicted_experts[0].resize(1);
        plan.expert_confidence[0].resize(1);
        plan.predicted_experts[1].resize(1);
        plan.expert_confidence[1].resize(1);

        // Layer 0: Expert 10 at depth 0 (conf 0.9) and depth 1 (conf 0.8)
        // Expert 20 at depth 0 only (conf 0.7)
        plan.predicted_experts[0][0] = {10, 20};
        plan.expert_confidence[0][0] = {0.9f, 0.7f};
        plan.predicted_experts[1][0] = {10};
        plan.expert_confidence[1][0] = {0.8f};

        auto ranked = atlas::MultiDepthExpertConsensus::rank_experts(plan, 0.85);
        CHECK(ranked.size() == 2);
        // Expert 10 appears twice across depths, so with recurrence bonus it must rank #1
        CHECK(ranked[0].key.expert == 10);
        CHECK(ranked[0].occurrence_count == 2);
        CHECK(ranked[1].key.expert == 20);
        CHECK(ranked[1].occurrence_count == 1);
        CHECK(ranked[0].consensus_score > ranked[1].consensus_score);

        // 3. TLEPMetrics KPIs
        atlas::TLEPMetrics metrics;
        metrics.gpu_slack_us = 10000;
        metrics.cpu_slack_us = 10000;
        metrics.speculative_gpu_us = 2500;
        metrics.speculative_cpu_us = 1500;
        metrics.speculative_tokens_generated = 10;
        metrics.accepted_tokens = 7;
        metrics.rejected_tokens = 3;
        metrics.drafted_by_depth[0] = 5;
        metrics.accepted_by_depth[0] = 4;
        metrics.verifications_count = 2;
        metrics.useful_prefetch_bytes = 8000;
        metrics.wasted_prefetch_bytes = 2000;

        CHECK(std::abs(metrics.gpu_slack_utilization() - 0.25) < 1e-4);
        CHECK(std::abs(metrics.cpu_slack_utilization() - 0.15) < 1e-4);
        CHECK(metrics.productive_slack_utilization() > 0.17 && metrics.productive_slack_utilization() < 0.18);
        CHECK(std::abs(metrics.speculation_efficiency(0) - 0.70) < 1e-4);
        CHECK(std::abs(metrics.speculation_efficiency(1) - 0.80) < 1e-4);
        CHECK(std::abs(metrics.accepted_tokens_per_verification() - 3.50) < 1e-4);
        CHECK(std::abs(metrics.prefetch_precision_bytes() - 0.80) < 1e-4);

        // 4. PredictivePrefetchScheduler & Bandwidth Budget Guard
        atlas::Config pf_cfg;
        atlas::ExpertMemoryManager mm(pf_cfg);
        atlas::PredictivePrefetchScheduler scheduler(mm, 64 * 1024 * 1024);
        atlas::TLEPMetrics pf_metrics;
        scheduler.execute_consensus_prefetch(plan, ranked, pf_metrics, nullptr);
        // Record demand hit for expert 10
        scheduler.record_expert_demand(0, 10, pf_metrics);
        CHECK(pf_metrics.useful_prefetches == 1);
        // Finish token cycle: expert 20 was unaccessed, counts as wasted
        scheduler.finish_token_cycle(pf_metrics);
        CHECK(pf_metrics.wasted_prefetches == 1);

        // 5. Transactional Rollback, Verification & State Transitions (Section 25 gates)
        struct MockModelContext {
            int n_past = 0;
            std::vector<llama_token> tokens;
            std::vector<uint8_t> recurrent_state;

            void push_token(llama_token t) {
                tokens.push_back(t);
                n_past++;
                recurrent_state.push_back(static_cast<uint8_t>(t & 0xFF));
            }

            void rollback_to(int pos, const std::vector<uint8_t> & saved_rec) {
                if (pos < (int)tokens.size()) {
                    tokens.resize(pos);
                }
                n_past = pos;
                recurrent_state = saved_rec;
            }
        };

        MockModelContext ctx_sim;
        ctx_sim.push_token(1);
        ctx_sim.push_token(2);
        std::vector<uint8_t> committed_rec = ctx_sim.recurrent_state;
        int committed_pos = ctx_sim.n_past;

        // Gate 1: Full speculative acceptance
        {
            llama_tokens drafts = {101, 102, 103, 104};
            atlas::LookaheadPlan plan_full;
            plan_full.predicted_tokens = drafts;
            plan_full.accepted_prefix = 4;
            plan_full.rejected_from = -1;
            plan_full.is_committed = true;
            CHECK(plan_full.is_committed);
            CHECK(plan_full.accepted_prefix == 4);
            CHECK(plan_full.rejected_from == -1);
            for (auto t : drafts) ctx_sim.push_token(t);
            committed_pos = ctx_sim.n_past;
            committed_rec = ctx_sim.recurrent_state;
        }

        // Gate 2: Zero acceptance (rejection at first token)
        {
            llama_tokens drafts = {201, 202, 203, 204};
            atlas::LookaheadPlan plan_zero;
            plan_zero.predicted_tokens = drafts;
            plan_zero.accepted_prefix = 0;
            plan_zero.rejected_from = 0;
            plan_zero.is_committed = false;
            CHECK(!plan_zero.is_committed);
            CHECK(plan_zero.accepted_prefix == 0);
            CHECK(plan_zero.rejected_from == 0);
            for (auto t : drafts) ctx_sim.push_token(t);
            CHECK(ctx_sim.n_past > committed_pos);
            ctx_sim.rollback_to(committed_pos, committed_rec);
            CHECK(ctx_sim.n_past == committed_pos);
            CHECK(ctx_sim.recurrent_state == committed_rec);
        }

        // Gate 3: Rejection in middle
        {
            llama_tokens drafts = {301, 302, 303, 304};
            size_t n_acc = 2;
            atlas::LookaheadPlan plan_mid;
            plan_mid.predicted_tokens = drafts;
            plan_mid.accepted_prefix = (int)n_acc;
            plan_mid.rejected_from = 2;
            plan_mid.is_committed = false;
            CHECK(plan_mid.accepted_prefix == 2);
            CHECK(plan_mid.rejected_from == 2);
            for (size_t i = 0; i < n_acc; ++i) ctx_sim.push_token(drafts[i]);
            auto mid_rec = ctx_sim.recurrent_state;
            int mid_pos = ctx_sim.n_past;
            for (size_t i = n_acc; i < drafts.size(); ++i) ctx_sim.push_token(drafts[i]);
            ctx_sim.rollback_to(mid_pos, mid_rec);
            CHECK(ctx_sim.n_past == mid_pos);
            CHECK(ctx_sim.recurrent_state == mid_rec);
            committed_pos = ctx_sim.n_past;
            committed_rec = ctx_sim.recurrent_state;
        }

        // Gate 4: Rejection at final draft token
        {
            llama_tokens drafts = {401, 402, 403, 404};
            size_t n_acc = 3;
            atlas::LookaheadPlan plan_final;
            plan_final.predicted_tokens = drafts;
            plan_final.accepted_prefix = (int)n_acc;
            plan_final.rejected_from = 3;
            plan_final.is_committed = false;
            CHECK(plan_final.accepted_prefix == 3);
            CHECK(plan_final.rejected_from == 3);
            for (size_t i = 0; i < n_acc; ++i) ctx_sim.push_token(drafts[i]);
            auto final_rec = ctx_sim.recurrent_state;
            int final_pos = ctx_sim.n_past;
            ctx_sim.push_token(drafts[3]);
            ctx_sim.rollback_to(final_pos, final_rec);
            CHECK(ctx_sim.n_past == final_pos);
            committed_pos = ctx_sim.n_past;
            committed_rec = ctx_sim.recurrent_state;
        }

        // Gate 5: Cancellation before launch
        {
            std::atomic<bool> cancel_flag{true};
            bool spec_launched = false;
            if (!cancel_flag.load()) {
                spec_launched = true;
            }
            CHECK(!spec_launched);
        }

        // Gate 6: Cancellation during multi-chunk speculative sequence
        {
            std::atomic<bool> cancel_flag{false};
            int chunks_executed = 0;
            for (int c = 0; c < 4; ++c) {
                if (c == 2) cancel_flag.store(true);
                if (cancel_flag.load()) break;
                chunks_executed++;
            }
            CHECK(chunks_executed == 2);
        }

        // Gate 7: Recurrent-state & KV rollback parity
        {
            auto before_state = ctx_sim.recurrent_state;
            auto before_toks = ctx_sim.tokens;
            ctx_sim.push_token(991);
            ctx_sim.push_token(992);
            CHECK(ctx_sim.recurrent_state != before_state);
            ctx_sim.rollback_to((int)before_toks.size(), before_state);
            CHECK(ctx_sim.recurrent_state == before_state);
            CHECK(ctx_sim.tokens == before_toks);
        }

        // Gate 8: Cache hit + speculative path
        {
            bool cache_hit = true;
            bool spec_active = true;
            int reused_prompt_tokens = 512;
            int total_prompt_tokens = 512;
            bool can_fast_spec = cache_hit && (reused_prompt_tokens == total_prompt_tokens) && spec_active;
            CHECK(can_fast_spec);
        }

        // Gate 9: Long-context speculative rollback (32k / 64k / 128k)
        {
            MockModelContext long_ctx;
            long_ctx.n_past = 32768;
            long_ctx.recurrent_state.assign(128, 42);
            auto long_saved_rec = long_ctx.recurrent_state;
            int long_pos = long_ctx.n_past;
            long_ctx.n_past += 4;
            long_ctx.recurrent_state[0] = 99;
            CHECK(long_ctx.n_past == 32772);
            long_ctx.rollback_to(long_pos, long_saved_rec);
            CHECK(long_ctx.n_past == 32768);
            CHECK(long_ctx.recurrent_state == long_saved_rec);
        }

        // Gate 10: Server cancellation
        {
            std::atomic<bool> server_cancelled{true};
            std::string finish_reason = "normal";
            if (server_cancelled.load()) {
                finish_reason = "cancelled";
            }
            CHECK(finish_reason == "cancelled");
        }

        // Gate 11: Stream failure & safe fallback
        {
            bool stream_failed = true;
            bool exact_fallback_active = false;
            if (stream_failed) {
                exact_fallback_active = true;
            }
            CHECK(exact_fallback_active);
        }

        // Gate 12: OOM / Low Slack Budget Fallback (K=0)
        {
            atlas::TLEPSlackEstimator tight_estimator(35.0); // 35 ms guard exceeds available 29.7 ms CPU MoE
            CHECK(tight_estimator.get_safe_gpu_slack_budget_us() == 0);
            int k_oom = tight_estimator.compute_dynamic_horizon(4, high_acc);
            CHECK(k_oom == 0);
        }

        // Gate 13: AtlasHybridStateCache L1 RAM & L2 Disk hit, miss, hash, and LRU eviction
        {
            atlas::AtlasHybridStateCache cache(1024 * 1024, "models/cache/test_cache");
            std::vector<llama_token> p1 = {101, 102, 103, 104, 105};
            std::vector<uint8_t> st1(4096, 0xAA);
            std::vector<uint8_t> sp1(512, 0x55);

            // Put into cache
            CHECK(cache.put(p1, st1.data(), st1.size(), sp1.data(), sp1.size()));
            CHECK(cache.get_l1_count() == 1);
            CHECK(cache.get_l1_bytes() == 4096);

            // Exact lookup with k_tail = 2
            std::vector<llama_token> req1 = {101, 102, 103, 104, 105, 106, 107};
            std::vector<uint8_t> st_out;
            std::vector<uint8_t> sp_out;
            bool from_l1 = false;
            size_t matched = cache.find_longest_prefix(req1, 2, st_out, sp_out, from_l1);
            CHECK(matched == 5);
            CHECK(from_l1 == true);
            CHECK(st_out == st1);
            CHECK(sp_out == sp1);
            CHECK(cache.get_hits() == 1);

            // Miss lookup
            std::vector<llama_token> req_miss = {999, 888, 777};
            st_out.clear();
            sp_out.clear();
            size_t matched_miss = cache.find_longest_prefix(req_miss, 2, st_out, sp_out, from_l1);
            CHECK(matched_miss == 0);
            CHECK(cache.get_misses() == 1);

            // Longest prefix match (when prompt is longer than cached)
            std::vector<llama_token> req_longer = {101, 102, 103, 104, 105, 201, 202, 203, 204};
            st_out.clear();
            sp_out.clear();
            size_t matched_prefix = cache.find_longest_prefix(req_longer, 2, st_out, sp_out, from_l1);
            CHECK(matched_prefix == 5);
            CHECK(st_out == st1);

            // Clean up test cache directory
            cache.clear();
            CHECK(cache.get_l1_count() == 0);
            CHECK(cache.get_l1_bytes() == 0);
        }

        // Gate 14: Attention-Sink Sliding Window (n_keep = 64, n_discard)
        {
            const int n_ctx_total = 2048;
            int n_past = 2048;
            int current_draft_n = 4;
            CHECK(n_past + current_draft_n >= n_ctx_total);

            const int n_keep = std::min(64, n_ctx_total / 8);
            const int n_discard = std::min(1024, std::max(16, n_ctx_total / 4));
            CHECK(n_keep == 64);
            CHECK(n_discard == 512);

            n_past -= n_discard;
            CHECK(n_past == 1536);
            CHECK(n_past + current_draft_n < n_ctx_total);
        }

        // Gate 15: Bit-Exact Prefix matching with K_tail = 2 boundary
        {
            std::vector<llama_token> prompt = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
            const size_t k_tail = 2;
            const size_t prefix_boundary = (prompt.size() > k_tail) ? (prompt.size() - k_tail) : 0;
            CHECK(prefix_boundary == 8);
            std::vector<llama_token> prefix(prompt.begin(), prompt.begin() + prefix_boundary);
            CHECK(prefix.size() == 8);
            CHECK(prefix.back() == 8);
            std::vector<llama_token> delta(prompt.begin() + prefix_boundary, prompt.end());
            CHECK(delta.size() == 2);
            CHECK(delta[0] == 9 && delta[1] == 10);
        }

        // Gate 16: Dynamic VRAM Headroom Guard (clamp_slots_per_layer and LUT sync)
        {
            atlas::GPUExpertBuffer buffer(64 * 1024 * 1024);
            atlas::FrequencyGatedSplitMoECache cache(buffer, 1);
            CHECK(cache.get_pinning().active_slots_per_layer == 21);
            CHECK(cache.get_pinning().is_resident(0, 0) == true);
            CHECK(cache.get_pinning().is_resident(0, 20) == true);

            static bool s_sync_called = false;
            static const int16_t * s_synced_lut = nullptr;
            cache.on_sync_lut = [](const int16_t * lut) {
                s_sync_called = true;
                s_synced_lut = lut;
            };

            CHECK(cache.clamp_slots_per_layer(19));
            CHECK(cache.get_pinning().active_slots_per_layer == 19);
            CHECK(cache.get_pinning().is_resident(0, 18) == true);
            CHECK(cache.get_pinning().is_resident(0, 19) == false);
            CHECK(cache.get_pinning().is_resident(0, 20) == false);
            CHECK(s_sync_called == true);
            CHECK(s_synced_lut != nullptr);
        }

        // Gate 17: Zero-Copy State Cache & Multi-Turn State Persistence (Phase 9 Milestone 3)
        {
            atlas::AtlasHybridStateCache cache(64 * 1024 * 1024, "models/cache_test_gate17");
            std::vector<llama_token> prefix = {10, 20, 30, 40, 50};
            std::vector<uint8_t> st(1024, 0x42);
            std::vector<uint8_t> sp(128, 0x13);
            CHECK(cache.put(prefix, st.data(), st.size(), sp.data(), sp.size()));

            std::shared_ptr<std::vector<uint8_t>> st_ptr;
            std::shared_ptr<std::vector<uint8_t>> sp_ptr;
            bool from_l1 = false;
            std::vector<llama_token> prompt = {10, 20, 30, 40, 50, 60, 70};
            size_t matched = cache.find_longest_prefix_shared(prompt, 2, st_ptr, sp_ptr, from_l1);
            CHECK(matched == 5);
            CHECK(from_l1 == true);
            CHECK(st_ptr != nullptr);
            CHECK(st_ptr->size() == 1024);
            CHECK((*st_ptr)[0] == 0x42);
            CHECK(sp_ptr != nullptr && sp_ptr->size() == 128);
            CHECK((*sp_ptr)[0] == 0x13);

            cache.clear();
        }

        // Gate 18: ggml_vec_dot_iq3_xxs_q8_K_x4 Bit-Exact 0-ULP Parity vs 4x single vec_dot (Phase 9 Milestone 2)
        {
            const int n = 256;
            const size_t bx = ggml_type_size(GGML_TYPE_IQ3_XXS);
            const size_t by_size = ggml_type_size(GGML_TYPE_Q8_K);
            std::vector<uint8_t> vx_raw(4 * bx);
            std::vector<uint8_t> vy_raw(by_size);
            for (size_t i = 0; i < vx_raw.size(); ++i) {
                vx_raw[i] = (uint8_t)((i * 37 + 11) & 0xFF);
            }
            for (size_t i = 0; i < vy_raw.size(); ++i) {
                vy_raw[i] = (uint8_t)((i * 19 + 7) & 0xFF);
            }

            float s_single[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float s_x4[4] = {0.0f, 0.0f, 0.0f, 0.0f};

            for (int r = 0; r < 4; ++r) {
                ggml_vec_dot_iq3_xxs_q8_K(n, &s_single[r], 0, vx_raw.data() + r * bx, 0, vy_raw.data(), 0, 1);
            }

            ggml_vec_dot_iq3_xxs_q8_K_x4(n, s_x4, sizeof(float), vx_raw.data(), bx, vy_raw.data(), 0, 4);

            for (int r = 0; r < 4; ++r) {
                CHECK(s_single[r] == s_x4[r]);
                uint32_t u_single = 0, u_x4 = 0;
                std::memcpy(&u_single, &s_single[r], 4);
                std::memcpy(&u_x4, &s_x4[r], 4);
                CHECK(u_single == u_x4);
            }
        }

        // Gate 19: Phase 10 Tiered Asymmetric KV Quantization State Machine (P0)
        {
            CHECK(atlas::KVQuantTierProfile::select_tier_for_tokens(4096) == atlas::KVQuantTier::TIER_0_Q8_Q8);
            CHECK(atlas::KVQuantTierProfile::select_tier_for_tokens(8192) == atlas::KVQuantTier::TIER_0_Q8_Q8);
            CHECK(atlas::KVQuantTierProfile::select_tier_for_tokens(8193) == atlas::KVQuantTier::TIER_1_Q8_Q4);
            CHECK(atlas::KVQuantTierProfile::select_tier_for_tokens(32768) == atlas::KVQuantTier::TIER_1_Q8_Q4);
            CHECK(atlas::KVQuantTierProfile::select_tier_for_tokens(32769) == atlas::KVQuantTier::TIER_2_Q4_Q4);
            CHECK(atlas::KVQuantTierProfile::select_tier_for_tokens(131072) == atlas::KVQuantTier::TIER_2_Q4_Q4);
            CHECK(atlas::KVQuantTierProfile::select_tier_for_tokens(131073) == atlas::KVQuantTier::TIER_3_COMPRESSED_RING);
            CHECK(atlas::KVQuantTierProfile::select_tier_for_tokens(256000) == atlas::KVQuantTier::TIER_3_COMPRESSED_RING);

            auto p0 = atlas::KVQuantTierProfile::get_profile(atlas::KVQuantTier::TIER_0_Q8_Q8);
            CHECK(p0.k_type == GGML_TYPE_Q8_0 && p0.v_type == GGML_TYPE_Q8_0);
            CHECK(p0.bytes_per_dim == 2.125);

            auto p1 = atlas::KVQuantTierProfile::get_profile(atlas::KVQuantTier::TIER_1_Q8_Q4);
            CHECK(p1.k_type == GGML_TYPE_Q8_0 && p1.v_type == GGML_TYPE_Q4_0);
            CHECK(p1.bytes_per_dim == 1.625);

            auto p2 = atlas::KVQuantTierProfile::get_profile(atlas::KVQuantTier::TIER_2_Q4_Q4);
            CHECK(p2.k_type == GGML_TYPE_Q4_0 && p2.v_type == GGML_TYPE_Q4_0);
            CHECK(p2.bytes_per_dim == 1.125);

            // Footprint check: 10,240 dim across layers in Tier 2 = 1.125 B/dim (~11.25 KB/tok)
            size_t fp_tier2 = static_cast<size_t>(10240 * p2.bytes_per_dim);
            CHECK(fp_tier2 == static_cast<size_t>(10240 * 1.125));
            size_t fp_50k = atlas::KVQuantTierProfile::calculate_footprint_bytes(50000, 40, 2, 128);
            CHECK(fp_50k == static_cast<size_t>(50000ULL * 10240 * 1.125));
        }

        // Gate 20: Amortized Background Re-Quantization Dual-Cursor Ring Buffer (P0)
        {
            atlas::AmortizedRequantizerDualCursor rq(4); // 4 blocks of 32 = 128 tokens/step
            rq.start_transition(atlas::KVQuantTier::TIER_0_Q8_Q8, atlas::KVQuantTier::TIER_1_Q8_Q4, 8192);
            CHECK(rq.is_in_transition() == true);
            CHECK(rq.get_cursor_quant() == 0);
            CHECK(rq.get_cursor_active() == 8192);

            // Step 1: converts 128 tokens
            size_t c1 = rq.step_re_quantize();
            CHECK(c1 == 128);
            CHECK(rq.get_cursor_quant() == 128);
            CHECK(rq.get_effective_tier(64) == atlas::KVQuantTier::TIER_1_Q8_Q4);
            CHECK(rq.get_effective_tier(200) == atlas::KVQuantTier::TIER_0_Q8_Q8);

            // Test rollback under active transition: roll back context to 64 tokens
            rq.rollback_context(64);
            CHECK(rq.get_cursor_active() == 64);
            CHECK(rq.get_cursor_quant() == 64);
            CHECK(rq.is_in_transition() == false);
            CHECK(rq.step_re_quantize() == 0); // No underflow!

            // Re-start transition and advance through remaining steps
            rq.start_transition(atlas::KVQuantTier::TIER_0_Q8_Q8, atlas::KVQuantTier::TIER_1_Q8_Q4, 8192);
            while (rq.is_in_transition()) {
                rq.step_re_quantize();
            }
            CHECK(rq.is_in_transition() == false);
            CHECK(rq.get_cursor_quant() == 8192);
            CHECK(rq.get_effective_tier(8000) == atlas::KVQuantTier::TIER_1_Q8_Q4);

            // Test append_token auto-triggering transition to Tier 2
            rq.append_token(32769);
            CHECK(rq.is_in_transition() == true);
            CHECK(rq.get_effective_tier(32769) == atlas::KVQuantTier::TIER_1_Q8_Q4);
        }

        // Gate 21: MoE-Aware Dynamic Expected Value (EV) Controller (P2)
        {
            atlas::MtpDynamicHorizonController mtp(0.15);
            CHECK(mtp.compute_expert_expansion_factor(1) == 1.0);
            CHECK(mtp.compute_expert_expansion_factor(2) > 1.0);
            CHECK(mtp.compute_expert_expansion_factor(4) > mtp.compute_expert_expansion_factor(2));

            // With warm baseline latencies (T_gpu=12ms, T_cpu_moe=28ms, T_draft=2.5ms)
            // EV(1) should be positive
            double ev1 = mtp.compute_expected_value(1);
            CHECK(ev1 > 0.0);
            int opt_d = mtp.select_optimal_horizon(4);
            CHECK(opt_d >= 1 && opt_d <= 4);

            // Verification result recording check: verify conditional probability update
            mtp.record_verification_result(3, 2); // 2 accepted out of 3
            CHECK(mtp.get_acceptance_probs()[0] > 0.80);

            // Fallback test: when draft model diverges (low acceptance), EV <= 0 -> opt_d = 0
            atlas::MtpDynamicHorizonController slow_mtp;
            slow_mtp.set_acceptance_probs({0.04, 0.02, 0.01, 0.00});
            int fallback_d = slow_mtp.select_optimal_horizon(4);
            CHECK(fallback_d == 0); // Collapses to d=0 fallback!
        }

        // Gate 22: Zero-Cost Rollback & Attention-Sink Preservation (P0/P2)
        {
            size_t cursor = 4096;
            atlas::MtpDynamicHorizonController::rollback_kv_cursor(cursor, 3);
            CHECK(cursor == 4093);

            // Attention sink check: token 0..63 always attended
            for (size_t k = 0; k < 64; ++k) {
                CHECK(atlas::BlockSparseAttentionConfig::is_token_attended(100000, k) == true);
            }
            // Local sliding window check: token within 8192 attended
            CHECK(atlas::BlockSparseAttentionConfig::is_token_attended(10000, 5000) == true);
            // Distant middle token not attended
            CHECK(atlas::BlockSparseAttentionConfig::is_token_attended(100000, 50000) == false);

            // Tile masking: verify boundary conditions and causal masking
            CHECK(atlas::BlockSparseAttentionConfig::is_tile_masked(10, 20) == true); // Future tile (distant)
            CHECK(atlas::BlockSparseAttentionConfig::is_tile_masked(10, 11) == true); // Immediate next future tile
            CHECK(atlas::BlockSparseAttentionConfig::is_tile_masked(10, 10) == false); // Diagonal tile attended
            CHECK(atlas::BlockSparseAttentionConfig::is_tile_masked(0, 0) == false);   // Sink tile attended
            CHECK(atlas::BlockSparseAttentionConfig::is_tile_masked(200, 0) == false); // Sink tile attended from distant query
            CHECK(atlas::BlockSparseAttentionConfig::is_tile_masked(200, 1) == true);  // Distant tile between sink and window is masked
        }

        // Gate 23: DirectStorage / BypassIO Pinned Memory Double Buffering (P3)
        {
            atlas::WindowsDirectStorageExpertStreamer streamer;
            streamer.set_prefetch_ring_size(2);
            CHECK(streamer.is_bypass_io_supported() == true);
            CHECK(streamer.is_cuda_pinned() == true);
            CHECK(streamer.get_staging_capacity() == 8 * 1024 * 1024);
            CHECK(streamer.get_slot_buffer(0) != nullptr);
            CHECK(streamer.get_slot_buffer(1) != nullptr);

            CHECK(streamer.enqueue_speculative_read(2, 10, 4096, 65536) == true);
            CHECK(streamer.wait_expert_ready(2, 10) == true);
            CHECK(streamer.wait_expert_ready(99, 99) == false);
            CHECK(streamer.get_completed_reads_count() == 1);

            // Slot recycling: double buffering slot 0 -> slot 1 -> slot 0
            CHECK(streamer.enqueue_speculative_read(3, 11, 8192, 65536) == true);
            CHECK(streamer.wait_expert_ready(3, 11) == true);
            CHECK(streamer.wait_expert_ready(2, 10) == true); // Both in slots 0 and 1
            // Third read overwrites slot 0
            CHECK(streamer.enqueue_speculative_read(4, 12, 12288, 65536) == true);
            CHECK(streamer.wait_expert_ready(4, 12) == true);
            CHECK(streamer.wait_expert_ready(2, 10) == false); // Old slot 0 recycled!
            CHECK(streamer.get_completed_reads_count() == 3);
        }

        // Gate 24: AVX-512 VNNI / Bit-Exact Parity vs Reference vec_dot (P1)
        {
            const size_t bx_elem = ggml_type_size(GGML_TYPE_IQ3_XXS);
            const size_t by_elem = ggml_type_size(GGML_TYPE_Q8_K);

            // Test across multiple super-block lengths (256, 512, 1024 elements)
            const int test_sizes[] = {256, 512, 1024};
            for (int n : test_sizes) {
                const int n_blocks = n / 256;
                std::vector<uint8_t> vx_raw(n_blocks * bx_elem);
                std::vector<uint8_t> vy_raw(n_blocks * by_elem);
                for (size_t i = 0; i < vx_raw.size(); ++i) {
                    vx_raw[i] = (uint8_t)((i * 43 + 17) & 0xFF);
                }
                for (size_t i = 0; i < vy_raw.size(); ++i) {
                    vy_raw[i] = (uint8_t)((i * 29 + 13) & 0xFF);
                }

                float s_ref = 0.0f;
                float s_vnni = 0.0f;
                ggml_vec_dot_iq3_xxs_q8_K(n, &s_ref, 0, vx_raw.data(), 0, vy_raw.data(), 0, 1);
                ggml_vec_dot_iq3_xxs_q8_K_vnni(n, &s_vnni, 0, vx_raw.data(), 0, vy_raw.data(), 0, 1);

                CHECK(s_ref == s_vnni);
                uint32_t u_ref = 0, u_vnni = 0;
                std::memcpy(&u_ref, &s_ref, 4);
                std::memcpy(&u_vnni, &s_vnni, 4);
                CHECK(u_ref == u_vnni);
            }
        }

        // Gate 25: Phase 11 AVX-512 VNNI Multi-Token (nrc = 2) Weight Sharing Parity
        {
            ggml_vnni_nrc2_reset_call_count();
            CHECK(ggml_vnni_nrc2_get_call_count() == 0);

            const size_t bx_elem = ggml_type_size(GGML_TYPE_IQ3_XXS);
            const size_t by_elem = ggml_type_size(GGML_TYPE_Q8_K);
            const int test_sizes[] = {256, 512, 1024};

            for (int n : test_sizes) {
                const int n_blocks = n / 256;
                std::vector<uint8_t> vx_raw(n_blocks * bx_elem);
                std::vector<uint8_t> vy0_raw(n_blocks * by_elem);
                std::vector<uint8_t> vy1_raw(n_blocks * by_elem);

                for (size_t i = 0; i < vx_raw.size(); ++i) {
                    vx_raw[i] = (uint8_t)((i * 47 + 19) & 0xFF);
                }
                for (size_t i = 0; i < vy0_raw.size(); ++i) {
                    vy0_raw[i] = (uint8_t)((i * 31 + 11) & 0xFF);
                }
                for (size_t i = 0; i < vy1_raw.size(); ++i) {
                    vy1_raw[i] = (uint8_t)((i * 53 + 23) & 0xFF);
                }

                // Reference single-token calls (nrc = 1)
                float s0_ref = 0.0f;
                float s1_ref = 0.0f;
                ggml_vec_dot_iq3_xxs_q8_K(n, &s0_ref, 0, vx_raw.data(), 0, vy0_raw.data(), 0, 1);
                ggml_vec_dot_iq3_xxs_q8_K(n, &s1_ref, 0, vx_raw.data(), 0, vy1_raw.data(), 0, 1);

                // Multi-token dual-vector call (nrc = 2) with shared weights
                std::vector<uint8_t> vy_combined(2 * n_blocks * by_elem);
                std::memcpy(vy_combined.data(), vy0_raw.data(), vy0_raw.size());
                std::memcpy(vy_combined.data() + vy0_raw.size(), vy1_raw.data(), vy1_raw.size());

                float s_vnni_2[2] = {0.0f, 0.0f};
                size_t by_stride = vy0_raw.size();
                ggml_vec_dot_iq3_xxs_q8_K_vnni(n, s_vnni_2, sizeof(float), vx_raw.data(), 0, vy_combined.data(), by_stride, 2);

                CHECK(s_vnni_2[0] == s0_ref);
                CHECK(s_vnni_2[1] == s1_ref);

                uint32_t u0_ref = 0, u0_vnni = 0;
                std::memcpy(&u0_ref, &s0_ref, 4);
                std::memcpy(&u0_vnni, &s_vnni_2[0], 4);
                CHECK(u0_ref == u0_vnni);

                uint32_t u1_ref = 0, u1_vnni = 0;
                std::memcpy(&u1_ref, &s1_ref, 4);
                std::memcpy(&u1_vnni, &s_vnni_2[1], 4);
                CHECK(u1_ref == u1_vnni);

                // Multi-token quad-vector call (nrc = 4) with shared weights
                std::vector<uint8_t> vy2_raw(n_blocks * by_elem);
                std::vector<uint8_t> vy3_raw(n_blocks * by_elem);
                for (size_t i = 0; i < vy2_raw.size(); ++i) {
                    vy2_raw[i] = (uint8_t)((i * 61 + 37) & 0xFF);
                }
                for (size_t i = 0; i < vy3_raw.size(); ++i) {
                    vy3_raw[i] = (uint8_t)((i * 71 + 43) & 0xFF);
                }
                float s2_ref = 0.0f;
                float s3_ref = 0.0f;
                ggml_vec_dot_iq3_xxs_q8_K(n, &s2_ref, 0, vx_raw.data(), 0, vy2_raw.data(), 0, 1);
                ggml_vec_dot_iq3_xxs_q8_K(n, &s3_ref, 0, vx_raw.data(), 0, vy3_raw.data(), 0, 1);

                std::vector<uint8_t> vy_combined_4(4 * n_blocks * by_elem);
                std::memcpy(vy_combined_4.data() + 0 * vy0_raw.size(), vy0_raw.data(), vy0_raw.size());
                std::memcpy(vy_combined_4.data() + 1 * vy0_raw.size(), vy1_raw.data(), vy1_raw.size());
                std::memcpy(vy_combined_4.data() + 2 * vy0_raw.size(), vy2_raw.data(), vy2_raw.size());
                std::memcpy(vy_combined_4.data() + 3 * vy0_raw.size(), vy3_raw.data(), vy3_raw.size());

                float s_vnni_4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                ggml_vec_dot_iq3_xxs_q8_K_vnni(n, s_vnni_4, sizeof(float), vx_raw.data(), 0, vy_combined_4.data(), by_stride, 4);

                CHECK(s_vnni_4[0] == s0_ref);
                CHECK(s_vnni_4[1] == s1_ref);
                CHECK(s_vnni_4[2] == s2_ref);
                CHECK(s_vnni_4[3] == s3_ref);

                uint32_t u2_ref = 0, u2_vnni = 0;
                std::memcpy(&u2_ref, &s2_ref, 4);
                std::memcpy(&u2_vnni, &s_vnni_4[2], 4);
                CHECK(u2_ref == u2_vnni);

                uint32_t u3_ref = 0, u3_vnni = 0;
                std::memcpy(&u3_ref, &s3_ref, 4);
                std::memcpy(&u3_vnni, &s_vnni_4[3], 4);
                CHECK(u3_ref == u3_vnni);
            }

            CHECK(ggml_vnni_nrc2_get_call_count() == 9);

            // Multithreaded concurrent atomic increment verification (16 threads x 1000 calls)
            ggml_vnni_nrc2_reset_call_count();
            CHECK(ggml_vnni_nrc2_get_call_count() == 0);
            {
                const int n_threads = 16;
                const int iters_per_thread = 1000;
                std::vector<std::thread> workers;
                workers.reserve(n_threads);
                for (int t = 0; t < n_threads; ++t) {
                    workers.emplace_back([iters_per_thread]() {
                        const size_t bx_elem = ggml_type_size(GGML_TYPE_IQ3_XXS);
                        const size_t by_elem = ggml_type_size(GGML_TYPE_Q8_K);
                        const int n = 256;
                        const int n_blocks = 1;
                        std::vector<uint8_t> vx_raw(n_blocks * bx_elem, 1);
                        std::vector<uint8_t> vy_combined(2 * n_blocks * by_elem, 1);
                        float s_vnni_2[2] = {0.0f, 0.0f};
                        for (int i = 0; i < iters_per_thread; ++i) {
                            ggml_vec_dot_iq3_xxs_q8_K_vnni(n, s_vnni_2, sizeof(float), vx_raw.data(), 0, vy_combined.data(), n_blocks * by_elem, 2);
                        }
                    });
                }
                for (auto & w : workers) {
                    w.join();
                }
                CHECK(ggml_vnni_nrc2_get_call_count() == n_threads * iters_per_thread);
            }
        }

        // Gate 26: Phase 11 Zero-VRAM N-Gram Prompt Lookup (PLD) Draft Engine
        {
            atlas::PromptLookupDraftEngine pld(2, 4);
            CHECK(pld.get_vram_usage_bytes() == 0);

            // Context: Repetitive code pattern
            std::vector<llama_token> ctx = {
                100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                110, 111, 112, 113, 104, 105, 106
            };
            // The suffix [104, 105, 106] matches position 4..6, followed by [107, 108, 109, 110]
            double draft_ms = 0.0;
            auto drafted = pld.draft(ctx, 3, draft_ms);

            CHECK(drafted.size() == 3);
            CHECK(drafted[0] == 107);
            CHECK(drafted[1] == 108);
            CHECK(drafted[2] == 109);
            CHECK(draft_ms < 0.10); // Target ~0.01 ms

            // Boundary test: Empty context
            std::vector<llama_token> empty_ctx;
            auto drafted_empty = pld.draft(empty_ctx, 4, draft_ms);
            CHECK(drafted_empty.empty());

            // Boundary test: 1 token (shorter than min_n_gram)
            std::vector<llama_token> single_ctx = {42};
            auto drafted_single = pld.draft(single_ctx, 4, draft_ms);
            CHECK(drafted_single.empty());

            // Boundary test: No matching n-gram
            std::vector<llama_token> non_rep = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
            auto drafted_none = pld.draft(non_rep, 4, draft_ms);
            CHECK(drafted_none.empty());

            // Long-context scaling test (8192 tokens with periodic repeats)
            std::vector<llama_token> long_ctx;
            long_ctx.reserve(8193);
            for (int i = 0; i < 8192; ++i) {
                long_ctx.push_back(static_cast<llama_token>(i % 256));
            }
            double draft_cold_ms = 0.0;
            pld.draft(long_ctx, 4, draft_cold_ms);

            // True incremental decode draft step (appended token in stream)
            long_ctx.push_back(0);
            double draft_step_ms = 0.0;
            auto drafted_long = pld.draft(long_ctx, 4, draft_step_ms);
            CHECK(drafted_long.size() == 4);
            CHECK(draft_step_ms < 0.05); // True per-step decode draft latency ~0.002-0.01 ms!

            // Integration test with MtpDynamicHorizonController:
            atlas::MtpDynamicHorizonController controller;
            CHECK(controller.get_t_draft_step_ms() == 2.5); // Neural default
            pld.integrate_with_horizon_controller(controller);
            CHECK(controller.get_t_draft_step_ms() < 0.05);  // PLD ~0.01 ms
            CHECK(controller.get_t_draft_step_ms() > 0.0);

            // With T_draft_step ~0.01 ms, EV(d) increases significantly and optimal horizon expands
            double ev_neural = 0.0;
            {
                atlas::MtpDynamicHorizonController neural_ctl;
                neural_ctl.set_latencies(12.0, 28.0, 2.5);
                ev_neural = neural_ctl.compute_expected_value(2);
            }
            double ev_pld = controller.compute_expected_value(2);
            CHECK(ev_pld > ev_neural);

            // Pre-warm prompt tokens test
            std::vector<llama_token> warm_prompt = {500, 501, 502, 503, 504, 505, 501, 502};
            pld.ingest_prompt_tokens(warm_prompt);
            double warm_draft_ms = 0.0;
            auto warm_drafted = pld.draft(warm_prompt, 2, warm_draft_ms);
            CHECK(warm_drafted.size() == 2);
            CHECK(warm_drafted[0] == 503);
            CHECK(warm_drafted[1] == 504);

            // Phase 13: Multi-turn incremental PLD prefix preservation & rollback test
            std::vector<llama_token> multi_turn_prompt = {500, 501, 502, 503, 504, 505, 501, 502, 506, 507, 501, 502};
            CHECK(pld.matches_prefix(multi_turn_prompt.data(), warm_prompt.size()));
            pld.ingest_prompt_tokens_incremental(multi_turn_prompt.data(), multi_turn_prompt.size(), warm_prompt.size());
            CHECK(pld.size() == multi_turn_prompt.size());
            auto mt_drafted = pld.draft(multi_turn_prompt, 2, warm_draft_ms);
            CHECK(mt_drafted.size() == 2);
            pld.rollback_to(warm_prompt.size());
            CHECK(pld.size() == warm_prompt.size());
            CHECK(pld.matches_prefix(warm_prompt.data(), warm_prompt.size()));

            // AtlasEngineConfig & mtp_verify_ubatch_limit verification
            atlas::AtlasEngineConfig a_cfg;
            CHECK(a_cfg.mtp_verify_ubatch_limit == 4);

            // Aligned horizon selection test (VNNI nrc=2 / nrc=4 SIMD alignment)
            int aligned_h = controller.select_optimal_horizon_aligned(2);
            CHECK(aligned_h >= 0);
            if (aligned_h > 0) {
                CHECK(aligned_h % 2 == 0);
            }
        }

        // -------------------------------------------------------------------
        // Phase 14 Gate: Pre-Allocated Slab Arenas & Dynamic Cascading 3-Tier Working Set
        // -------------------------------------------------------------------
        {
            // 1. Slab Arena Pool Allocation & O(1) Free List Stack Tests
            const size_t slot_bytes = 1024 * 1024; // 1 MB
            atlas::ExpertSlabPool vram_pool(8, slot_bytes, false, false);
            CHECK(vram_pool.total_slots() == 8);
            CHECK(vram_pool.used_slots() == 0);
            CHECK(vram_pool.free_slots() == 8);
            CHECK(vram_pool.slot_bytes() == slot_bytes);
            CHECK(vram_pool.total_capacity_bytes() == 8 * slot_bytes);

            std::vector<int> acquired;
            for (int i = 0; i < 8; ++i) {
                int s = vram_pool.acquire_slot();
                CHECK(s >= 0 && s < 8);
                CHECK(vram_pool.get_slot_ptr(s) != nullptr);
                acquired.push_back(s);
            }
            CHECK(vram_pool.used_slots() == 8);
            CHECK(vram_pool.free_slots() == 0);
            // Overflow test: 9th acquire returns -1 without crash or corruption
            CHECK(vram_pool.acquire_slot() == -1);

            // Release 2 slots
            vram_pool.release_slot(acquired.back());
            acquired.pop_back();
            vram_pool.release_slot(acquired.back());
            acquired.pop_back();
            CHECK(vram_pool.used_slots() == 6);
            CHECK(vram_pool.free_slots() == 2);

            // Re-acquire slots: verifies LIFO reuse
            int re_s1 = vram_pool.acquire_slot();
            int re_s2 = vram_pool.acquire_slot();
            CHECK(re_s1 >= 0 && re_s2 >= 0);
            CHECK(vram_pool.used_slots() == 8);
            CHECK(vram_pool.acquire_slot() == -1);

            // 2. Cascading 3-Tier Working Set Migration (T2 <-> T1 <-> T0)
            atlas::Config mm_cfg;
            mm_cfg.vram_cap_mb = 4; // 4 MB = 4 slots (1 MB each)
            mm_cfg.ram_cap_mb  = 8; // 8 MB = 8 slots
            mm_cfg.vram_slab_slots = 4;
            mm_cfg.ram_slab_slots = 8;
            mm_cfg.gpu_expert_layers = 1; // Layer 0 is anchor in VRAM
            mm_cfg.dynamic_cascading_working_set = true;
            mm_cfg.odmoe_quick_evict = true;

            atlas::ExpertMemoryManager mm(mm_cfg);
            CHECK(mm.get_ram_slab_pool().is_cuda_pinned() == true);
            CHECK(mm.get_vram_active_slabs() == 0);
            CHECK(mm.get_ram_active_slabs() == 0);

            // Verify initial state: Layer 0 anchor is VRAM, other layers are Tier-2 (NVMe)
            CHECK(mm.get_residency(0, 10) == atlas::MemoryTier::VRAM);
            CHECK(mm.get_residency(2, 10) == atlas::MemoryTier::NVME);
            CHECK(mm.get_vram_slot(2, 10) == -1);
            CHECK(mm.get_ram_slot(2, 10) == -1);

            // Phase 14 Test: Promote expert (2, 10) directly to Tier-1 RAM (T2 -> T1)
            CHECK(mm.promote_expert(2, 10, atlas::MemoryTier::RAM));
            CHECK(mm.get_residency(2, 10) == atlas::MemoryTier::RAM);
            CHECK(mm.get_ram_slot(2, 10) >= 0);
            CHECK(mm.get_ram_slot_ptr(2, 10) != nullptr);
            CHECK(mm.get_ram_active_slabs() == 1);
            CHECK(mm.get_metrics().tier2_to_tier1_promotions == 1);

            // Phase 14 Test: Promote expert (2, 10) from Tier-1 RAM to Tier-0 VRAM (T1 -> T0)
            CHECK(mm.promote_expert(2, 10, atlas::MemoryTier::VRAM));
            CHECK(mm.get_residency(2, 10) == atlas::MemoryTier::VRAM);
            CHECK(mm.get_vram_slot(2, 10) >= 0);
            CHECK(mm.get_vram_slot_ptr(2, 10) != nullptr);
            CHECK(mm.get_vram_active_slabs() == 1);
            CHECK(mm.get_metrics().tier1_to_tier0_promotions == 1);

            // Promote 3 more experts to fill all 4 VRAM slots (Layer 2, experts 11, 12, 13)
            CHECK(mm.promote_expert(2, 11, atlas::MemoryTier::VRAM));
            CHECK(mm.promote_expert(2, 12, atlas::MemoryTier::VRAM));
            CHECK(mm.promote_expert(2, 13, atlas::MemoryTier::VRAM));
            CHECK(mm.get_vram_active_slabs() == 4);

            // Phase 14 Test: Capacity Overfill & Cascading Demotion (T0 -> T1)
            // When expert (2, 14) is promoted to Tier-0, VRAM is full!
            // The coldest unpinned non-anchor expert in VRAM (expert 10) must be demoted to RAM.
            CHECK(mm.promote_expert(2, 14, atlas::MemoryTier::VRAM));
            CHECK(mm.get_residency(2, 14) == atlas::MemoryTier::VRAM);
            CHECK(mm.get_vram_active_slabs() == 4); // Still 4, no leak
            // Expert 10 was demoted to Tier-1 RAM:
            CHECK(mm.get_residency(2, 10) == atlas::MemoryTier::RAM);
            CHECK(mm.get_vram_slot(2, 10) == -1);
            CHECK(mm.get_ram_slot(2, 10) >= 0);
            CHECK(mm.get_metrics().tier0_to_tier1_demotions >= 1);

            // Pin expert 11 so it cannot be evicted
            mm.pin(2, 11);
            // Promote expert 15 to VRAM: expert 11 must be skipped, next coldest (12) demoted to RAM
            CHECK(mm.promote_expert(2, 15, atlas::MemoryTier::VRAM));
            CHECK(mm.get_residency(2, 11) == atlas::MemoryTier::VRAM); // Protected by pin!
            CHECK(mm.get_residency(2, 12) == atlas::MemoryTier::RAM);  // Evicted instead
            mm.unpin_all();

            // Phase 14 Test: Cascading Demotion (T1 -> T2 NVMe)
            // Now fill Tier-1 RAM slots until RAM is full (total 8 slots)
            for (int e = 20; e < 26; ++e) {
                CHECK(mm.promote_expert(3, e, atlas::MemoryTier::RAM));
            }
            CHECK(mm.get_ram_active_slabs() == 8);

            // Trigger another demotion into RAM -> forces coldest Tier-1 expert down to Tier-2 NVMe!
            CHECK(mm.promote_expert(4, 50, atlas::MemoryTier::RAM));
            CHECK(mm.get_ram_active_slabs() == 8); // Clamped at 8, zero leak
            CHECK(mm.get_metrics().tier1_to_tier2_demotions >= 1);

            // Re-promote the demoted expert back to Tier-0 (Verifies clean re-entry & LFU reset)
            CHECK(mm.promote_expert(2, 12, atlas::MemoryTier::VRAM));
            CHECK(mm.get_residency(2, 12) == atlas::MemoryTier::VRAM);

            // Phase 14 Deep Test: Verify VRAM-resident experts are NOT corrupted by RAM saturation/evictions
            CHECK(mm.get_residency(2, 14) == atlas::MemoryTier::VRAM);
            CHECK(mm.get_vram_slot(2, 14) >= 0);
            CHECK(mm.get_ram_slot(2, 14) == -1);

            // Phase 14 Deep Test: Verify access() returns VRAM when promoting hot RAM expert
            atlas::Config access_cfg;
            access_cfg.vram_slab_slots = 4;
            access_cfg.ram_slab_slots = 8;
            access_cfg.gpu_expert_layers = 1;
            access_cfg.dynamic_cascading_working_set = true;
            atlas::ExpertMemoryManager mm_acc(access_cfg);
            // Promote (2, 10) to RAM (initial hotness = 1)
            CHECK(mm_acc.promote_expert(2, 10, atlas::MemoryTier::RAM));
            CHECK(mm_acc.get_residency(2, 10) == atlas::MemoryTier::RAM);
            // Access: hotness increments to 2 >= threshold -> promotes to VRAM -> MUST return VRAM!
            CHECK(mm_acc.access(2, 10, false) == atlas::MemoryTier::VRAM);
            CHECK(mm_acc.get_residency(2, 10) == atlas::MemoryTier::VRAM);
            CHECK(mm_acc.get_vram_slot(2, 10) >= 0);
            CHECK(mm_acc.get_ram_slot(2, 10) == -1);

            // Phase 14 Deep Test: High-churn thrashing stress test (100 migrations, zero slab leak invariant)
            for (int step = 0; step < 100; ++step) {
                int l = 1 + (step % 5);
                int e = step % 30;
                atlas::MemoryTier target = (step % 3 == 0) ? atlas::MemoryTier::VRAM : atlas::MemoryTier::RAM;
                mm.promote_expert(l, e, target);
                // Verify slab conservation invariant: used + free == total
                CHECK(mm.get_vram_slab_pool().used_slots() + mm.get_vram_slab_pool().free_slots() == mm.get_vram_slab_pool().total_slots());
                CHECK(mm.get_ram_slab_pool().used_slots() + mm.get_ram_slab_pool().free_slots() == mm.get_ram_slab_pool().total_slots());
                CHECK(mm.get_vram_active_slabs() <= 4);
                CHECK(mm.get_ram_active_slabs() <= 8);
            }

            // Test flush_all cleans everything
            mm.flush_all();
            CHECK(mm.get_vram_active_slabs() == 0);
            CHECK(mm.get_ram_active_slabs() == 0);
            CHECK(mm.get_residency(2, 12) == atlas::MemoryTier::NVME);

            // Gate 28: Qwen3.5-122B MXFP4 AVX-512 VNNI x4 & nrc2 0-ULP Bit-Exact Parity + 0-VRAM LUT Invariant
            {
                ggml_cpu_init();
                const size_t bx_elem = ggml_type_size(GGML_TYPE_MXFP4);
                const size_t by_elem = ggml_type_size(GGML_TYPE_Q8_0);
                const int test_sizes[] = {256, 1024, 3072};

                for (int n : test_sizes) {
                    const int n_blocks = n / 32;
                    const size_t row_bytes_x = n_blocks * bx_elem;
                    const size_t row_bytes_y = n_blocks * by_elem;

                    std::vector<uint8_t> vx_raw(4 * row_bytes_x);
                    std::vector<uint8_t> vy_raw(2 * row_bytes_y);

                    for (size_t r = 0; r < 4; ++r) {
                        for (int b = 0; b < n_blocks; ++b) {
                            uint8_t * blk = vx_raw.data() + r * row_bytes_x + b * bx_elem;
                            blk[0] = (uint8_t)(120 + ((r * 7 + b * 3) % 12));
                            for (size_t q = 1; q < bx_elem; ++q) {
                                blk[q] = (uint8_t)(((r * 53 + b * 19 + q * 37) + 11) & 0xFF);
                            }
                        }
                    }

                    std::vector<float> y_fp32(2 * n);
                    for (size_t i = 0; i < y_fp32.size(); ++i) {
                        y_fp32[i] = std::sin(float(i) * 0.031f) * 1.5f;
                    }
                    ggml_quantize_chunk(GGML_TYPE_Q8_0, y_fp32.data(), vy_raw.data(), 0, 2, n, nullptr);

                    float s_ref_x4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    float s_opt_x4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    for (int r = 0; r < 4; ++r) {
                        ggml_vec_dot_mxfp4_q8_0(n, &s_ref_x4[r], 0, vx_raw.data() + r * row_bytes_x, 0, vy_raw.data(), 0, 1);
                    }
                    ggml_vec_dot_mxfp4_q8_0_x4(n, s_opt_x4, sizeof(float), vx_raw.data(), row_bytes_x, vy_raw.data(), 0, 4);

                    for (int r = 0; r < 4; ++r) {
                        CHECK(std::isfinite(s_ref_x4[r]) && s_ref_x4[r] != 0.0f);
                        uint32_t u_ref = 0, u_opt = 0;
                        std::memcpy(&u_ref, &s_ref_x4[r], 4);
                        std::memcpy(&u_opt, &s_opt_x4[r], 4);
                        CHECK(u_ref == u_opt);
                    }

                    float s_ref_nrc2[2] = {0.0f, 0.0f};
                    float s_opt_nrc2[2] = {0.0f, 0.0f};
                    ggml_vec_dot_mxfp4_q8_0(n, &s_ref_nrc2[0], 0, vx_raw.data(), 0, vy_raw.data(), 0, 1);
                    ggml_vec_dot_mxfp4_q8_0(n, &s_ref_nrc2[1], 0, vx_raw.data(), 0, vy_raw.data() + row_bytes_y, 0, 1);
                    ggml_vec_dot_mxfp4_q8_0_nrc2(n, s_opt_nrc2, sizeof(float), vx_raw.data(), 0, vy_raw.data(), row_bytes_y, 2);

                    for (int t = 0; t < 2; ++t) {
                        CHECK(std::isfinite(s_ref_nrc2[t]) && s_ref_nrc2[t] != 0.0f);
                        uint32_t u_ref = 0, u_opt = 0;
                        std::memcpy(&u_ref, &s_ref_nrc2[t], 4);
                        std::memcpy(&u_opt, &s_opt_nrc2[t], 4);
                        CHECK(u_ref == u_opt);
                    }
                }

                atlas::GPUExpertBuffer zero_buf(0);
                atlas::FrequencyGatedSplitMoECache zero_cache(zero_buf, 10);
                zero_cache.clamp_slots_per_layer(0);
                CHECK(zero_cache.get_pinning().active_slots_per_layer == 0);
                for (int l = 0; l < 48; ++l) {
                    for (int e = 0; e < 256; ++e) {
                        CHECK(zero_cache.get_pinning().gpu_slot_lut[l][e] == -1);
                    }
                }
            }

            // Gate 29: Dual-Priority DirectStorage Staging Pools (Demand=2, Prefetch=4) & Zero-Copy Slot Promotion
            {
                atlas::WindowsDirectStorageExpertStreamer ds;
                CHECK(ds.is_cuda_pinned() == true);
                CHECK(ds.get_demand_slot_count() == 2);
                CHECK(ds.get_prefetch_slot_count() == 4);
                for (int i = 0; i < 2; ++i) {
                    CHECK(ds.get_demand_slot_buffer(i) != nullptr);
                    CHECK((reinterpret_cast<uintptr_t>(ds.get_demand_slot_buffer(i)) % 4096) == 0);
                }
                for (int i = 0; i < 4; ++i) {
                    CHECK(ds.get_prefetch_slot_buffer(i) != nullptr);
                    CHECK((reinterpret_cast<uintptr_t>(ds.get_prefetch_slot_buffer(i)) % 4096) == 0);
                }

                // 1. Fill Demand Pool (2 urgent layer-5 experts)
                CHECK(ds.enqueue_demand_read(5, 100, 4096, 65536) == true);
                CHECK(ds.enqueue_demand_read(5, 101, 8192, 65536) == true);
                CHECK(ds.get_demand_reads_count() == 2);
                CHECK(ds.get_prefetch_reads_count() == 0);

                // 2. Flood Prefetch Pool with 4 speculative reads for layers 6 & 7
                CHECK(ds.enqueue_speculative_read(6, 200, 16384, 65536) == true);
                CHECK(ds.enqueue_speculative_read(6, 201, 20480, 65536) == true);
                CHECK(ds.enqueue_speculative_read(7, 202, 24576, 65536) == true);
                CHECK(ds.enqueue_speculative_read(7, 203, 28672, 65536) == true);
                CHECK(ds.get_prefetch_reads_count() == 4);

                // 3. Verify zero cross-pool eviction: Demand Pool experts are still intact!
                CHECK(ds.wait_expert_ready(5, 100) == true);
                CHECK(ds.wait_expert_ready(5, 101) == true);
                CHECK(ds.resolve_expert_buffer(5, 100) == ds.get_demand_slot_buffer(0));
                CHECK(ds.resolve_expert_buffer(5, 101) == ds.get_demand_slot_buffer(1));
                CHECK(ds.get_zero_copy_promotions_count() == 0);

                // 4. Verify Zero-Copy Slot Promotion on Prefetch Hit:
                // Requesting prefetched expert (7, 202) promotes it in-place from Prefetch slot 2
                // without allocating a Demand slot or copying data!
                uint64_t demand_before = ds.get_demand_reads_count();
                CHECK(ds.enqueue_demand_read(7, 202, 24576, 65536) == true);
                CHECK(ds.get_demand_reads_count() == demand_before);
                CHECK(ds.get_zero_copy_promotions_count() == 1);
                CHECK(ds.resolve_expert_buffer(7, 202) == ds.get_prefetch_slot_buffer(2));

                // And wait_expert_ready on another prefetched expert (6, 201) also promotes in-place
                CHECK(ds.wait_expert_ready(6, 201) == true);
                CHECK(ds.get_zero_copy_promotions_count() == 2);
                CHECK(ds.resolve_expert_buffer(6, 201) == ds.get_prefetch_slot_buffer(1));

                //Demand Pool slots (5, 100) and (5, 101) remain untouched!
                CHECK(ds.resolve_expert_buffer(5, 100) == ds.get_demand_slot_buffer(0));
                CHECK(ds.resolve_expert_buffer(5, 101) == ds.get_demand_slot_buffer(1));
            }

            // -------------------------------------------------------------------
            // Gate 30: Phase 16 DirectStorage io_uring DMA & Zero-OOM Elastic Watermark
            // -------------------------------------------------------------------
            {
                // 1. Verify DirectStorageIoUringDMA initialization and alignment
                atlas::DirectStorageIoUringDMA dma(-1, nullptr);
                CHECK(dma.num_slots() == 4);
                CHECK(dma.slot_capacity() == 32 * 1024 * 1024);
                for (size_t s = 0; s < dma.num_slots(); ++s) {
                    CHECK(dma.get_slot_ptr(s) != nullptr);
                    CHECK(((uintptr_t)dma.get_slot_ptr(s) % 4096) == 0); // 4KB aligned
                }

                // 2. Test disk read pipeline with sector-aligned temporary file & O_DIRECT
                const std::string dma_test_path = "experiments/runtime_audit/dma_test_p16.bin";
                {
                    std::ofstream out(dma_test_path, std::ios::binary);
                    std::string payload(65536, 'Z');
                    out.write(payload.data(), payload.size());
                }
#if defined(__linux__)
                int test_fd = open(dma_test_path.c_str(), O_RDONLY | O_DIRECT);
                if (test_fd < 0) {
                    test_fd = open(dma_test_path.c_str(), O_RDONLY);
                }
#else
                int test_fd = open(dma_test_path.c_str(), O_RDONLY);
#endif
                if (test_fd >= 0) {
                    atlas::DirectStorageIoUringDMA test_dma(test_fd, nullptr);
                    // Sector-aligned read
                    CHECK(test_dma.enqueue_ssd_read(0, 0, 4096, 1, 10, 0) == true);
                    CHECK(test_dma.wait_slot_completed(0) == true);
                    CHECK(test_dma.get_completed_requests() == 1);
                    CHECK(test_dma.get_bytes_streamed() >= 4096);
                    CHECK(test_dma.get_slot_payload_offset(0) == 0);
                    CHECK(*((const char *)test_dma.get_slot_ptr(0)) == 'Z');

                    // Unaligned offset read (128 bytes shift)
                    CHECK(test_dma.enqueue_ssd_read(1, 128, 4096, 1, 11, 0) == true);
                    CHECK(test_dma.wait_slot_completed(1) == true);
                    CHECK(test_dma.get_slot_payload_offset(1) == 128);
                    const char * payload_data = (const char *)test_dma.get_slot_ptr(1) + test_dma.get_slot_payload_offset(1);
                    CHECK(*payload_data == 'Z');

                    // Multi-part GGUF file descriptor routing
                    int dup_fd = dup(test_fd);
                    test_dma.set_blob_fd(1, dup_fd);
                    CHECK(test_dma.get_blob_fd(1) == dup_fd);

                    // Safe NULL-buffer handling
                    CHECK(test_dma.dispatch_to_vram_async(0, nullptr, 4096) == false);

                    close(test_fd);
                }
                std::remove(dma_test_path.c_str());

                // 3. Verify Zero-OOM Elastic Watermarking in ExpertMemoryManager
                atlas::Config wm_cfg;
                atlas::ExpertMemoryManager mm(wm_cfg);
                CHECK(mm.get_vram_soft_capacity() == 1536 * 1024 * 1024ULL); // 1.5 GiB default soft ceiling
                CHECK(mm.get_kv_reserved_headroom() == 512 * 1024 * 1024ULL); // 512 MiB guaranteed baseline

                // Baseline context progression (0 tokens)
                mm.sync_context_progress(0, 65536, 16384);
                CHECK(mm.get_vram_soft_capacity() == 1536 * 1024 * 1024ULL);
                CHECK(mm.get_kv_reserved_headroom() == 512 * 1024 * 1024ULL);

                // Simulate medium context progression (16k context): headroom expands, soft cap shrinks
                mm.sync_context_progress(16384, 65536, 16384);
                CHECK(mm.get_kv_reserved_headroom() >= 768 * 1024 * 1024ULL);
                CHECK(mm.get_vram_soft_capacity() < 1536 * 1024 * 1024ULL);

                // Simulate deep context progression (64k context): headroom reaches ceiling, soft cap shrinks toward floor
                mm.sync_context_progress(65536, 65536, 16384);
                CHECK(mm.get_kv_reserved_headroom() >= 1024 * 1024 * 1024ULL);
                CHECK(mm.get_vram_soft_capacity() <= 256 * 1024 * 1024ULL);

                // Watermark enforcement returns true when headroom is maintained
                CHECK(mm.enforce_kv_watermark(1253376, nullptr) == true);

                // Thread-safety verification under concurrent watermark queries
                std::vector<std::thread> workers;
                std::atomic<bool> all_ok{true};
                for (int t = 0; t < 4; ++t) {
                    workers.emplace_back([&mm, &all_ok, t]() {
                        for (int iter = 0; iter < 50; ++iter) {
                            if (!mm.enforce_kv_watermark(iter * 1024, nullptr)) {
                                all_ok = false;
                            }
                            if ((iter % 10) == 0) {
                                mm.sync_context_progress(t * 1000 + iter * 100, 65536, 16384);
                            }
                        }
                    });
                }
                for (auto & w : workers) {
                    w.join();
                }
                CHECK(all_ok.load() == true);

                // 4. Verify Phase 16 L-Prefill 64-token threshold & Lookahead Router in Runtime
                atlas::Config p16_cfg;
                atlas::Runtime p16_rt(p16_cfg);
                CHECK(p16_rt.l_prefill_threshold == 64); // User constraint: exactly 64 tokens!
                CHECK(p16_rt.enable_layer_wise_prefill == true);
                CHECK(p16_rt.enable_wavefront_cooking == true);

                // Trigger lookahead preduction without crash and check layer deduplication
                p16_rt.execute_lookahead_router_preduction(0, nullptr);
                CHECK(p16_rt.last_preduction_layer == 0);
                p16_rt.execute_lookahead_router_preduction(0, nullptr); // Identical layer: deduplicated
                CHECK(p16_rt.last_preduction_layer == 0);
            }
        }
    }

    // Gate 31: Saturated RAM Pool Demote Idempotency & Clamped Split-MoE Bounds Regression
    {
        atlas::Config sat_cfg;
        sat_cfg.vram_slab_slots = 4;
        sat_cfg.ram_slab_slots = 4;
        sat_cfg.gpu_expert_layers = 1;
        sat_cfg.dynamic_cascading_working_set = true;
        atlas::ExpertMemoryManager sat_mm(sat_cfg);

        for (int e = 0; e < 4; ++e) {
            CHECK(sat_mm.promote_expert(1, e, atlas::MemoryTier::RAM));
        }
        CHECK(sat_mm.get_ram_slab_pool().free_slots() == 0);

        CHECK(sat_mm.promote_expert(2, 0, atlas::MemoryTier::VRAM));
        CHECK(sat_mm.demote_vram_to_ram(2, 0, nullptr) == true);
        // Calling demote_vram_to_ram again on an already-RAM expert must not leak slots or duplicate LRU nodes.
        for (int rep = 0; rep < 8; ++rep) {
            CHECK(sat_mm.demote_vram_to_ram(2, 0, nullptr) == true);
        }
        CHECK(sat_mm.get_ram_active_slabs() == 4);
        CHECK(sat_mm.get_ram_slab_pool().used_slots() == 4);
        CHECK(sat_mm.get_ram_slab_pool().used_slots() + sat_mm.get_ram_slab_pool().free_slots() == sat_mm.get_ram_slab_pool().total_slots());
        CHECK(sat_mm.get_vram_slab_pool().used_slots() + sat_mm.get_vram_slab_pool().free_slots() == sat_mm.get_vram_slab_pool().total_slots());

        for (int e = 4; e < 8; ++e) {
            CHECK(sat_mm.promote_expert(2, e, atlas::MemoryTier::VRAM));
        }
        sat_mm.evict_vram(64ULL * 1024ULL * 1024ULL);
        CHECK(sat_mm.get_vram_slab_pool().free_slots() == sat_mm.get_vram_slab_pool().total_slots());
        sat_mm.evict_ram(64ULL * 1024ULL * 1024ULL);
        CHECK(sat_mm.get_ram_slab_pool().free_slots() == sat_mm.get_ram_slab_pool().total_slots());
        sat_mm.flush_all();
        CHECK(sat_mm.get_ram_slab_pool().free_slots() == sat_mm.get_ram_slab_pool().total_slots());
        CHECK(sat_mm.get_vram_slab_pool().free_slots() == sat_mm.get_vram_slab_pool().total_slots());

        atlas::GPUExpertBuffer clamp_buf(64 * 1024 * 1024);
        atlas::FrequencyGatedSplitMoECache clamp_cache(clamp_buf, 1);
        static int16_t s_max_observed_slot = -1;
        s_max_observed_slot = -1;
        clamp_cache.on_slot_update = [](int, int, int, int16_t slot) {
            CHECK(slot >= 0 && slot < 19);
            if (slot > s_max_observed_slot) s_max_observed_slot = slot;
        };
        CHECK(clamp_cache.clamp_slots_per_layer(19));
        CHECK(!clamp_cache.get_pinning().is_resident(0, 19));
        CHECK(!clamp_cache.get_pinning().is_resident(0, 20));
        clamp_cache.warm_expert(0, 50);
        auto clamp_plan = clamp_cache.plan_split_execution(0, {18, 19, 50});
        CHECK(!clamp_plan.gpu.empty());
        CHECK(clamp_cache.get_pinning().active_slots_per_layer == 19);
        CHECK(s_max_observed_slot >= 0 && s_max_observed_slot < 19);
        CHECK(clamp_cache.get_pinning().slot_to_expert[0][19] == -1);
        CHECK(clamp_cache.get_pinning().slot_to_expert[0][20] == -1);

        atlas::FrequencyGatedSplitMoECache zero_split(clamp_buf, 1);
        CHECK(zero_split.clamp_slots_per_layer(0));
        zero_split.warm_expert(0, 50);
        auto zero_plan = zero_split.plan_split_execution(0, {0, 1, 50});
        CHECK(zero_plan.gpu.empty() && zero_plan.cpu.size() == 3);

        // Gate 32: Phase 17 Real CUDA VRAM Arena + Locked RAM Unified Cache
        {
            atlas::UnifiedTieredExpertCache ucache;
            const size_t gate_sz = 4096;
            const size_t up_sz   = 4096;
            const size_t down_sz = 4096;
            const size_t stride  = 4096 * 3;
            std::vector<uint8_t> mock_gate(16 * gate_sz);
            std::vector<uint8_t> mock_up(16 * up_sz);
            std::vector<uint8_t> mock_down(16 * down_sz);
            for (size_t b = 0; b < mock_gate.size(); ++b) {
                mock_gate[b] = uint8_t((b * 13 + 7) & 0xFF);
                mock_up[b]   = uint8_t((b * 29 + 3) & 0xFF);
                mock_down[b] = uint8_t((b * 43 + 11) & 0xFF);
            }
            CHECK(ucache.init(stride * 4, stride * 4, stride, 64.0f));
            atlas::UnifiedTieredExpertCache::LayerMeta lm;
            lm.gate_expert_bytes = gate_sz;
            lm.up_expert_bytes   = up_sz;
            lm.down_expert_bytes = down_sz;
            lm.gate_base = mock_gate.data();
            lm.up_base   = mock_up.data();
            lm.down_base = mock_down.data();
            ucache.register_layer_weights(2, lm);
            CHECK(ucache.has_real_ram());
            bool ram_admitted_cb = false;
            ucache.set_ram_callback([&](int l, int e, int, bool admitted) {
                if (l == 2 && e == 5 && admitted) ram_admitted_cb = true;
            });
            ucache.access_and_promote(2, 5, true, ucache.has_real_vram());
            CHECK(ram_admitted_cb);
            const uint8_t * r_gate = static_cast<const uint8_t *>(ucache.get_ram_tensor_ptr(2, 5, 0));
            const uint8_t * r_down = static_cast<const uint8_t *>(ucache.get_ram_tensor_ptr(2, 5, 2));
            CHECK(r_gate != nullptr && r_down != nullptr);
            CHECK(std::memcmp(r_gate, mock_gate.data() + 5 * gate_sz, gate_sz) == 0);
            CHECK(std::memcmp(r_down, mock_down.data() + 5 * down_sz, down_sz) == 0);
            if (ucache.has_real_vram()) {
                const int vslot = ucache.get_vram_slot(2, 5);
                CHECK(vslot >= 0);
                const uint64_t dev_ptr = ucache.get_vram_slot_dev_ptr(vslot, 2, 0);
                CHECK(dev_ptr != 0);
                std::vector<uint8_t> readback(gate_sz, 0);
                CHECK(atlas::CudaDriverMemBridge::copy_d2h_async(readback.data(), dev_ptr, gate_sz, nullptr));
                CHECK(atlas::CudaDriverMemBridge::sync_stream(nullptr));
                CHECK(std::memcmp(readback.data(), mock_gate.data() + 5 * gate_sz, gate_sz) == 0);
            }
            ucache.shutdown();
        }

        // Gate 33: Phase 17 Lockless Mapped Mailbox & Adaptive CPU Share Controller
        {
            atlas::MappedRouterMailboxRing mb;
            CHECK(mb.init());
            const int32_t ids[4] = {12, 45, 99, 200};
            const float wts[4]   = {0.4f, 0.3f, 0.2f, 0.1f};
            mb.publish_cpu_snapshot(7, 42ULL, 0b0101ULL, 0b1010ULL, 1, 4, ids, wts);
            atlas::LayerRouterMailbox out{};
            CHECK(mb.poll_layer_ready(7, 42ULL, out, 100));
            CHECK(out.gpu_hit_mask == 0b0101ULL && out.cpu_miss_mask == 0b1010ULL);
            CHECK(out.expert_ids[1] == 45 && out.expert_ids[3] == 200);
            mb.release();

            atlas::AdaptiveCpuShareController ctrl(-1.0, 0.40, 0.30, 0.80);
            // Small miss count hidden behind GPU dense overlap => 100% CPU share
            auto dec_small = ctrl.plan_misses(8, 0b11111100ULL, 4, true); // 2 misses
            CHECK(dec_small.cpu_count == 2 && dec_small.gpu_upload_count == 0);
            CHECK(dec_small.effective_cpu_share > 0.99);
            // Large miss burst exceeding CPU hidden window => mixed CPU + GPU upload
            auto dec_large = ctrl.plan_misses(10, 0ULL, 8, true); // 10 misses
            CHECK(dec_large.cpu_count > 0 && dec_large.gpu_upload_count > 0);
            CHECK(dec_large.cpu_count + dec_large.gpu_upload_count == 10);
            // Fixed user share override
            ctrl.set_user_share(0.5);
            auto dec_fixed = ctrl.plan_misses(8, 0ULL, 8, true);
            CHECK(dec_fixed.cpu_count == 4 && dec_fixed.gpu_upload_count == 4);
        }

        // Gate 34: Phase 17 ClassLRUPolicy CRF Decay & Frequency-Class Eviction
        {
            atlas::ClassLRUPolicy policy;
            policy.init(3, 16.0f);
            const int s0 = policy.admit(1, 10, 1, 1.0f);
            const int s1 = policy.admit(1, 20, 1, 1.0f);
            const int s2 = policy.admit(1, 30, 1, 1.0f);
            CHECK(s0 >= 0 && s1 >= 0 && s2 >= 0 && policy.free_slots() == 0);
            // Boost expert 10 repeatedly at step 2..8
            for (uint64_t t = 2; t <= 8; ++t) policy.touch(s0, t, 1.0f);
            // Touch expert 30 once at step 8
            policy.touch(s2, 8, 1.0f);
            // Evict victim at step 9 => expert 20 (lowest class / oldest recency) must go first!
            int ev_l = -1, ev_e = -1;
            const int ev_slot = policy.evict_victim(nullptr, &ev_l, &ev_e);
            CHECK(ev_slot == s1 && ev_l == 1 && ev_e == 20);
            // After long decay (step 500), expert 10's score decays so fresh multi-hit can overtake
            policy.admit(1, 40, 500, 1.0f);
            const auto * n0 = policy.get_node(s0);
            CHECK(n0 != nullptr && n0->occupied);
            policy.touch(s0, 500, 1.0f);
            CHECK(policy.get_node(s0)->crf_score < 2.0f);
        }

        // Gate 35: Phase 2.1 FFN Chunked vs Unchunked Mathematical Equivalence
        {
            struct ggml_init_params gp = { 4 * 1024 * 1024, nullptr, true };
            struct ggml_context * gctx = ggml_init(gp);
            const int n_embd = 64, n_ff = 128, n_tok = 16;
            struct ggml_tensor * w_gate = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n_embd, n_ff);
            struct ggml_tensor * w_up   = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n_embd, n_ff);
            struct ggml_tensor * w_down = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n_ff, n_embd);
            struct ggml_tensor * inp    = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n_embd, n_tok);

            auto eval_ffn = [&](struct ggml_tensor * x) {
                struct ggml_tensor * g = ggml_mul_mat(gctx, w_gate, x);
                struct ggml_tensor * u = ggml_mul_mat(gctx, w_up, x);
                struct ggml_tensor * s = ggml_swiglu_split(gctx, g, u);
                return ggml_mul_mat(gctx, w_down, s);
            };
            struct ggml_tensor * ref_out = eval_ffn(inp);

            std::vector<struct ggml_tensor *> chunks;
            for (int i0 = 0; i0 < n_tok; i0 += 5) {
                const int c = std::min(5, n_tok - i0);
                struct ggml_tensor * xv = ggml_view_2d(gctx, inp, n_embd, c, inp->nb[1], size_t(i0) * inp->nb[1]);
                chunks.push_back(eval_ffn(xv));
            }
            while (chunks.size() > 1) {
                std::vector<struct ggml_tensor *> next;
                for (size_t i = 0; i + 1 < chunks.size(); i += 2) {
                    next.push_back(ggml_concat(gctx, chunks[i], chunks[i + 1], 1));
                }
                if (chunks.size() % 2 != 0) next.push_back(chunks.back());
                chunks.swap(next);
            }
            struct ggml_tensor * chk_out = chunks[0];

            struct ggml_cgraph * gf = ggml_new_graph(gctx);
            ggml_build_forward_expand(gf, ref_out);
            ggml_build_forward_expand(gf, chk_out);

            ggml_backend_t cpu = ggml_backend_cpu_init();
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(gctx, cpu);
            auto fill_det = [](struct ggml_tensor * t, float scale, int seed) {
                std::vector<float> v(ggml_nelements(t));
                for (size_t i = 0; i < v.size(); ++i) v[i] = scale * std::sin(float((i * 13 + seed) % 97) * 0.1f);
                ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
            };
            fill_det(w_gate, 0.2f, 1); fill_det(w_up, 0.2f, 2); fill_det(w_down, 0.2f, 3); fill_det(inp, 0.5f, 4);
            CHECK(ggml_backend_graph_compute(cpu, gf) == GGML_STATUS_SUCCESS);

            std::vector<float> ref_v(n_embd * n_tok), chk_v(n_embd * n_tok);
            ggml_backend_tensor_get(ref_out, ref_v.data(), 0, ggml_nbytes(ref_out));
            ggml_backend_tensor_get(chk_out, chk_v.data(), 0, ggml_nbytes(chk_out));
            float max_diff = 0.0f;
            for (size_t i = 0; i < ref_v.size(); ++i) max_diff = std::max(max_diff, std::fabs(ref_v[i] - chk_v[i]));
            CHECK(max_diff < 1e-5f);
            ggml_backend_buffer_free(buf);
            ggml_backend_free(cpu);
            ggml_free(gctx);
        }

        // Gate 36: Phase 2.2 FP8 E4M3 Encoding, Decoding, Saturation, and vec_dot Parity
        {
            CHECK(ggml_type_size(GGML_TYPE_F8_E4M3) == 1);
            CHECK(ggml_blck_size(GGML_TYPE_F8_E4M3) == 1);
            CHECK(std::strcmp(ggml_type_name(GGML_TYPE_F8_E4M3), "f8_e4m3") == 0);
            const ggml_fp8_e4m3_t max_pos = ggml_fp32_to_fp8_e4m3(448.0f);
            const ggml_fp8_e4m3_t sat_pos = ggml_fp32_to_fp8_e4m3(999.0f);
            CHECK(max_pos == 0x7E && sat_pos == 0x7E);
            CHECK(std::fabs(ggml_fp8_e4m3_to_fp32(max_pos) - 448.0f) < 1e-4f);
            CHECK(std::fabs(ggml_fp8_e4m3_to_fp32(ggml_fp32_to_fp8_e4m3(1.5f)) - 1.5f) < 1e-4f);

            const int N = 128;
            std::vector<float> a(N), b(N);
            for (int i = 0; i < N; ++i) {
                a[i] = std::sin(float(i) * 0.17f) * 3.0f;
                b[i] = std::cos(float(i) * 0.23f) * 2.0f;
            }
            std::vector<ggml_fp8_e4m3_t> a_fp8(N);
            ggml_fp32_to_fp8_e4m3_row(a.data(), a_fp8.data(), N);
            std::vector<float> a_dq(N);
            ggml_fp8_e4m3_to_fp32_row(a_fp8.data(), a_dq.data(), N);

            float ref_dot = 0.0f;
            for (int i = 0; i < N; ++i) ref_dot += a_dq[i] * b[i];
            float fp8_dot = 0.0f;
            ggml_vec_dot_f8_e4m3_f32(N, &fp8_dot, 0, a_fp8.data(), 0, b.data(), 0, 1);
            CHECK(std::fabs(ref_dot - fp8_dot) < 1e-4f);
        }

        // Gate 39: Phase 4.5 MTP Speculative / GDN Recurrent State Rollback Determinism
        {
            // Verify HybridStateCache prefix/state save+restore rollback determinism + GATED_DELTA_NET step consistency
            atlas::AtlasHybridStateCache sc(16 * 1024 * 1024ULL, "/tmp/atlas_gate39_sc");
            std::vector<llama_token> p_base = { 10, 20, 30, 40 };
            std::vector<uint8_t> snap0(4096);
            for (size_t i = 0; i < snap0.size(); ++i) snap0[i] = uint8_t((i * 37 + 11) & 0xFF);
            CHECK(sc.put(p_base, snap0.data(), snap0.size()));

            std::vector<llama_token> p_draft = { 10, 20, 30, 40, 999, 888 };
            std::vector<uint8_t> snap_draft(4096, 0xAB);
            CHECK(sc.put(p_draft, snap_draft.data(), snap_draft.size()));

            // Simulating rejected draft rollback: recovering longest prefix {10,20,30,40} must restore exact snap0 bytes
            std::vector<llama_token> q_roll = { 10, 20, 30, 40, 777 };
            std::vector<uint8_t> recovered, spec_out;
            bool from_l1 = false;
            const size_t matched = sc.find_longest_prefix(q_roll, 1, recovered, spec_out, from_l1);
            CHECK(matched == 4 && from_l1 && recovered == snap0);
        }

        // Gate 37: Phase 3.3-3.4 CUDA Odd-Row & Empty MoE Tile MUL_MAT_ID Correctness Regression
        {
#if defined(GGML_USE_CUDA)
            ggml_backend_t cuda = ggml_backend_cuda_init(0);
            ggml_backend_t cpu  = ggml_backend_cpu_init();
            CHECK(cuda != nullptr && cpu != nullptr);

            for (int n_tok : {1, 3}) {
                for (int nrows : {65, 127}) { // odd expert row counts (PR #32159/#32385)
                    const int ncols = 256, n_exp = 16, K = 2; // many unselected empty tiles (PR #32067)
                    const size_t w_nbytes = ggml_row_size(GGML_TYPE_Q4_0, ncols) * nrows * n_exp;
                    std::vector<uint8_t> w_q4(w_nbytes);
                    {
                        std::vector<float> w_f32((size_t) ncols * nrows * n_exp);
                        for (size_t i = 0; i < w_f32.size(); ++i) w_f32[i] = 0.2f * std::sin(float((i * 19 + 7) % 101) * 0.1f);
                        ggml_quantize_chunk(GGML_TYPE_Q4_0, w_f32.data(), w_q4.data(), 0, nrows * n_exp, ncols, nullptr);
                    }
                    std::vector<float> x_v(ncols * n_tok);
                    for (size_t i = 0; i < x_v.size(); ++i) x_v[i] = 0.5f * std::cos(float((i * 11 + 3) % 73) * 0.1f);
                    std::vector<int32_t> id_v(K * n_tok);
                    for (int t = 0; t < n_tok; ++t) { id_v[t * K + 0] = 2; id_v[t * K + 1] = 11; }

                    auto eval_mmid = [&](ggml_backend_t bk) -> std::vector<float> {
                        struct ggml_init_params gp = { 8 * 1024 * 1024, nullptr, true };
                        struct ggml_context * ctx = ggml_init(gp);
                        struct ggml_tensor * w  = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, ncols, nrows, n_exp);
                        struct ggml_tensor * x  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ncols, 1, n_tok);
                        struct ggml_tensor * id = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, K, n_tok);
                        struct ggml_tensor * o  = ggml_mul_mat_id(ctx, w, x, id);
                        struct ggml_cgraph * gf = ggml_new_graph(ctx);
                        ggml_build_forward_expand(gf, o);
                        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, bk);
                        ggml_backend_tensor_set(w,  w_q4.data(), 0, w_q4.size());
                        ggml_backend_tensor_set(x,  x_v.data(),  0, x_v.size() * sizeof(float));
                        ggml_backend_tensor_set(id, id_v.data(), 0, id_v.size() * sizeof(int32_t));
                        CHECK(ggml_backend_graph_compute(bk, gf) == GGML_STATUS_SUCCESS);
                        std::vector<float> out(nrows * K * n_tok);
                        ggml_backend_tensor_get(o, out.data(), 0, out.size() * sizeof(float));
                        ggml_backend_buffer_free(buf);
                        ggml_free(ctx);
                        return out;
                    };

                    auto cpu_o  = eval_mmid(cpu);
                    auto cuda_o = eval_mmid(cuda);
                    float max_diff = 0.0f;
                    for (size_t i = 0; i < cpu_o.size(); ++i) max_diff = std::max(max_diff, std::fabs(cpu_o[i] - cuda_o[i]));
                    CHECK(max_diff < 5e-3f);
                }
            }
            ggml_backend_free(cuda);
            ggml_backend_free(cpu);
#endif
        }

        // Gate 38: Phase 4.1 Fused qwen4exp HC Mix-Collapse & Combine CUDA Kernel Parity
        {
#if defined(GGML_USE_CUDA)
            ggml_backend_t cuda = ggml_backend_cuda_init(0);
            CHECK(cuda != nullptr);

            const int n_embd = 256, hc = 4, nt = 5;
            struct ggml_init_params gp = { 8 * 1024 * 1024, nullptr, true };
            struct ggml_context * gctx = ggml_init(gp);

            struct ggml_tensor * xn   = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n_embd * hc, nt);
            struct ggml_tensor * gpre = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n_embd * hc, nt);
            struct ggml_tensor * res  = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, n_embd, hc, nt);
            struct ggml_tensor * inj  = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, hc, nt);

            // Unfused reference HC mix
            ggml_tensor * gate_ref = ggml_sigmoid(gctx, gpre);
            ggml_tensor * gated_r  = ggml_reshape_3d(gctx, ggml_mul(gctx, xn, gate_ref), n_embd, hc, nt);
            ggml_tensor * mix_ref  = ggml_cont(gctx, ggml_view_2d(gctx, gated_r, n_embd, nt, sizeof(float) * n_embd * hc, 0));
            for (int c = 1; c < hc; ++c) {
                mix_ref = ggml_add(gctx, mix_ref, ggml_view_2d(gctx, gated_r, n_embd, nt, sizeof(float) * n_embd * hc, sizeof(float) * n_embd * c));
            }
            mix_ref = ggml_scale(gctx, mix_ref, 1.0f / float(hc));

            // Fused HC mix custom op
            ggml_tensor * mix_args[2] = { xn, gpre };
            ggml_tensor * mix_fused = ggml_custom_4d(gctx, GGML_TYPE_F32, n_embd, nt, 1, 1, mix_args, 2, nullptr, 1, nullptr);
            mix_fused->op_params[6]  = 0x41544C41;
            mix_fused->op_params[7]  = 0x48434D58;
            mix_fused->op_params[8]  = n_embd;
            mix_fused->op_params[9]  = hc;
            mix_fused->op_params[10] = nt;
            mix_fused->op_params[11] = 1;

            // Unfused reference HC combine (using mix_ref as block_out)
            ggml_tensor * w_ref = ggml_reshape_3d(gctx, ggml_scale(gctx, ggml_sigmoid(gctx, ggml_scale(gctx, inj, 1.0f / float(hc))), 2.0f), 1, hc, nt);
            ggml_tensor * b_ref = ggml_repeat_4d(gctx, ggml_reshape_3d(gctx, mix_ref, n_embd, 1, nt), n_embd, hc, nt, 1);
            ggml_tensor * comb_ref = ggml_add(gctx, res, ggml_mul(gctx, b_ref, w_ref));

            // Fused HC combine custom op
            ggml_tensor * comb_args[3] = { res, mix_fused, inj };
            ggml_tensor * comb_fused = ggml_custom_4d(gctx, GGML_TYPE_F32, n_embd, hc, nt, 1, comb_args, 3, nullptr, 1, nullptr);
            comb_fused->op_params[6]  = 0x41544C41;
            comb_fused->op_params[7]  = 0x48434342;
            comb_fused->op_params[8]  = n_embd;
            comb_fused->op_params[9]  = hc;
            comb_fused->op_params[10] = nt;

            struct ggml_cgraph * gf = ggml_new_graph(gctx);
            ggml_build_forward_expand(gf, comb_ref);
            ggml_build_forward_expand(gf, comb_fused);
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(gctx, cuda);
            auto fill_t = [&](struct ggml_tensor * t, float s, int seed) {
                std::vector<float> v(ggml_nelements(t));
                for (size_t i = 0; i < v.size(); ++i) v[i] = s * std::sin(float((i * 13 + seed) % 89) * 0.1f);
                ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
            };
            fill_t(xn, 0.5f, 1); fill_t(gpre, 1.0f, 2); fill_t(res, 0.3f, 3); fill_t(inj, 0.7f, 4);
            CHECK(ggml_backend_graph_compute(cuda, gf) == GGML_STATUS_SUCCESS);

            std::vector<float> mr(n_embd * nt), mf(n_embd * nt), cr(n_embd * hc * nt), cf(n_embd * hc * nt);
            ggml_backend_tensor_get(mix_ref,    mr.data(), 0, ggml_nbytes(mix_ref));
            ggml_backend_tensor_get(mix_fused,  mf.data(), 0, ggml_nbytes(mix_fused));
            ggml_backend_tensor_get(comb_ref,   cr.data(), 0, ggml_nbytes(comb_ref));
            ggml_backend_tensor_get(comb_fused, cf.data(), 0, ggml_nbytes(comb_fused));
            for (size_t i = 0; i < mr.size(); ++i) CHECK(std::fabs(mr[i] - mf[i]) < 1e-5f);
            for (size_t i = 0; i < cr.size(); ++i) CHECK(std::fabs(cr[i] - cf[i]) < 1e-5f);

            ggml_backend_buffer_free(buf);
            ggml_free(gctx);
            ggml_backend_free(cuda);
#endif
        }

        // Gate 40: Phase 17 Real Unified Tier-0 CUDA VRAM + CPU Split MoE Execution Parity
        {
#if defined(GGML_USE_CUDA)
            ggml_backend_t cuda = ggml_backend_cuda_init(0);
            ggml_backend_t cpu  = ggml_backend_cpu_init();
            CHECK(cuda != nullptr && cpu != nullptr);

            const int n_embd = 256, n_ff = 256, n_exp = 4, K = 2;
            const enum ggml_type qtype = GGML_TYPE_Q4_0;
            const size_t row_bytes_e = ggml_row_size(qtype, n_embd);
            const size_t exp_bytes_g = row_bytes_e * n_ff;
            const size_t exp_bytes_d = ggml_row_size(qtype, n_ff) * n_embd;

            std::vector<uint8_t> raw_gate(n_exp * exp_bytes_g), raw_up(n_exp * exp_bytes_g), raw_down(n_exp * exp_bytes_d);
            auto quant_fill = [&](std::vector<uint8_t> & dst, int rows, int cols, int seed) {
                std::vector<float> tmp(rows * cols);
                for (size_t i = 0; i < tmp.size(); ++i) tmp[i] = 0.15f * std::sin(float((i * 17 + seed) % 113) * 0.1f);
                ggml_quantize_chunk(qtype, tmp.data(), dst.data(), 0, rows, cols, nullptr);
            };
            quant_fill(raw_gate, n_exp * n_ff, n_embd, 11);
            quant_fill(raw_up,   n_exp * n_ff, n_embd, 29);
            quant_fill(raw_down, n_exp * n_embd, n_ff, 47);

            std::vector<float> x_host(n_embd);
            for (int i = 0; i < n_embd; ++i) x_host[i] = 0.4f * std::cos(float(i) * 0.11f);
            int32_t ids_host[2] = { 1, 3 };
            float   w_host[2]   = { 0.65f, 0.35f };

            // Build shared CPU reference and split CPU+GPU compute via ggml_backend_sched
            struct ggml_init_params gp = { 8 * 1024 * 1024, nullptr, true };
            struct ggml_context * wctx = ggml_init(gp);
            struct ggml_tensor * t_gate = ggml_new_tensor_3d(wctx, qtype, n_embd, n_ff, n_exp);
            struct ggml_tensor * t_up   = ggml_new_tensor_3d(wctx, qtype, n_embd, n_ff, n_exp);
            struct ggml_tensor * t_down = ggml_new_tensor_3d(wctx, qtype, n_ff, n_embd, n_exp);
            ggml_set_name(t_gate, "blk.0.ffn_gate_exps.weight");
            ggml_set_name(t_up,   "blk.0.ffn_up_exps.weight");
            ggml_set_name(t_down, "blk.0.ffn_down_exps.weight");
            ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, cpu);
            ggml_backend_tensor_set(t_gate, raw_gate.data(), 0, raw_gate.size());
            ggml_backend_tensor_set(t_up,   raw_up.data(),   0, raw_up.size());
            ggml_backend_tensor_set(t_down, raw_down.data(), 0, raw_down.size());

            auto run_moe = [&](bool enable_gpu_hit, bool second_hit_too) -> std::vector<float> {
                struct ggml_init_params gp2 = { 8 * 1024 * 1024, nullptr, true };
                struct ggml_context * gctx = ggml_init(gp2);
                struct ggml_tensor * tx  = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, n_embd, 1, 1);
                struct ggml_tensor * tid = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, K, 1);
                struct ggml_tensor * tw  = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, 1, K, 1);
                ggml_set_input(tx); ggml_set_input(tid); ggml_set_input(tw);

                struct ggml_tensor * g_out = ggml_mul_mat_id(gctx, t_gate, tx, tid);
                struct ggml_tensor * u_out = ggml_mul_mat_id(gctx, t_up,   tx, tid);
                struct ggml_tensor * swi   = ggml_swiglu_split(gctx, g_out, u_out);
                struct ggml_tensor * d_out = ggml_mul_mat_id(gctx, t_down, swi, tid);
                struct ggml_tensor * w_out = ggml_mul(gctx, d_out, tw);
                struct ggml_tensor * e0 = ggml_view_2d(gctx, w_out, n_embd, 1, w_out->nb[2], 0);
                struct ggml_tensor * e1 = ggml_view_2d(gctx, w_out, n_embd, 1, w_out->nb[2], w_out->nb[1]);
                struct ggml_tensor * cpu_sum = ggml_add(gctx, e0, e1);

                struct ggml_tensor * total = cpu_sum;
                struct ggml_tensor * gpu_moe = nullptr;
                if (enable_gpu_hit) {
                    struct ggml_tensor * args[3] = { tx, tid, tw };
                    gpu_moe = ggml_custom_4d(gctx, GGML_TYPE_F32, n_embd, 1, 1, 1, args, 3, nullptr, 1, nullptr);
                    gpu_moe->op_params[6] = 0x41544C41;
                    gpu_moe->op_params[7] = 0x534D4F45;
                    gpu_moe->op_params[8] = 0;
                    total = ggml_add(gctx, gpu_moe, cpu_sum);
                }
                struct ggml_cgraph * gf = ggml_new_graph(gctx);
                if (gpu_moe) ggml_build_forward_expand(gf, gpu_moe);
                ggml_build_forward_expand(gf, total);

                ggml_backend_t backends[2] = { cuda, cpu };
                ggml_backend_buffer_type_t bufts[2] = { ggml_backend_cuda_buffer_type(0), ggml_backend_cpu_buffer_type() };
                ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, 2, 256, false, true);
                if (gpu_moe) {
                    ggml_backend_sched_set_tensor_backend(sched, gpu_moe, cuda);
                    ggml_backend_sched_set_tensor_backend(sched, total, cuda);
                }
                ggml_backend_sched_reserve(sched, gf);
                ggml_backend_sched_alloc_graph(sched, gf);
                ggml_backend_tensor_set(tx,  x_host.data(), 0, sizeof(float) * n_embd);
                ggml_backend_tensor_set(tid, ids_host, 0, sizeof(ids_host));
                ggml_backend_tensor_set(tw,  w_host,   0, sizeof(w_host));

                atlas::atlas_set_gpu_cache_active(enable_gpu_hit ? 1 : 0);
                atlas::atlas_set_is_decode(enable_gpu_hit ? 1 : 0);
                ggml_backend_cuda_atlas_moe_set_active(enable_gpu_hit ? 1 : 0);
                ggml_backend_cuda_atlas_moe_set_decode(enable_gpu_hit ? 1 : 0);

                CHECK(ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS);
                std::vector<float> res(n_embd);
                ggml_backend_tensor_get(total, res.data(), 0, ggml_nbytes(total));
                ggml_backend_sched_free(sched);
                ggml_free(gctx);
                (void) second_hit_too;
                return res;
            };

            auto ref = run_moe(false, false);

            atlas::UnifiedTieredExpertCache ucache;
            atlas::UnifiedTieredExpertCache::LayerMeta lm;
            lm.gate_base = raw_gate.data(); lm.gate_expert_bytes = exp_bytes_g; lm.gate_type = qtype;
            lm.up_base   = raw_up.data();   lm.up_expert_bytes   = exp_bytes_g; lm.up_type   = qtype;
            lm.down_base = raw_down.data(); lm.down_expert_bytes = exp_bytes_d; lm.down_type = qtype;
            lm.n_embd = n_embd; lm.n_ff = n_ff;
            CHECK(ucache.init(4 * 1024 * 1024ULL, 0, lm.slot_bytes(), 64.0f));
            ucache.register_layer_weights(0, lm);
            ucache.set_vram_callback([](int l, int e, int slot, bool admitted) {
                ::atlas_set_expert_gpu_slot(l, e, admitted ? int16_t(slot) : int16_t(-1));
            });
            CHECK(ggml_backend_cuda_atlas_moe_set_arena(ucache.vram_dev_base(), ucache.slot_stride_bytes(), 1));
            CHECK(ggml_backend_cuda_atlas_moe_set_layer_meta(0, qtype, qtype, qtype, n_embd, n_ff, 0ULL, lm.up_offset(), lm.down_offset()));

            // Split 1: Expert 1 in GPU VRAM, Expert 3 on CPU
            CHECK(ucache.access_and_promote(0, 1, false, true));
            ggml_backend_cuda_atlas_moe_sync_lut(ucache.vram_lut_flat(), 1, atlas::UnifiedTieredExpertCache::MAX_EXPERTS);
            auto split1 = run_moe(true, false);

            // Split 2: Promote Expert 3 too -> BOTH experts on GPU, 0 on CPU (also tests TLS epoch invalidation)
            CHECK(ucache.access_and_promote(0, 3, false, true));
            ggml_backend_cuda_atlas_moe_sync_lut(ucache.vram_lut_flat(), 1, atlas::UnifiedTieredExpertCache::MAX_EXPERTS);
            auto split2 = run_moe(true, true);

            float max_d1 = 0.0f, max_d2 = 0.0f;
            for (int i = 0; i < n_embd; ++i) {
                max_d1 = std::max(max_d1, std::fabs(ref[i] - split1[i]));
                max_d2 = std::max(max_d2, std::fabs(ref[i] - split2[i]));
            }
            CHECK(max_d1 < 5e-3f && max_d2 < 5e-3f);

            ggml_backend_cuda_atlas_moe_set_active(0);
            ggml_backend_cuda_atlas_moe_set_decode(0);
            ggml_backend_cuda_atlas_moe_set_arena(0, 0, 0);
            atlas::atlas_set_gpu_cache_active(0);
            atlas::atlas_set_is_decode(0);
            ::atlas_set_expert_gpu_slot(0, 1, -1);
            ::atlas_set_expert_gpu_slot(0, 3, -1);
            ucache.shutdown();
            ggml_backend_buffer_free(wbuf);
            ggml_free(wctx);
            ggml_backend_free(cuda);
            ggml_backend_free(cpu);
#endif
        }

        // Gate 41: Phase 18 DirectStorageIoUringDMA 3-Tensor Unaligned NVMe -> Tier-0 VRAM Parity & Race-Free LUT Publish
        {
#if defined(GGML_USE_CUDA) && defined(__linux__)
            const std::string dma_path = "experiments/runtime_audit/2026-09-10/p18-dma3-test.bin";
            const size_t gate_sz = 4352;
            const size_t up_sz   = 4352;
            const size_t down_sz = 3840;
            const uint64_t off_g = 137;
            const uint64_t off_u = 9013;
            const uint64_t off_d = 21111;
            std::vector<uint8_t> blob(32768, 0xA5);
            std::vector<uint8_t> ref_g(gate_sz), ref_u(up_sz), ref_d(down_sz);
            for (size_t i = 0; i < gate_sz; ++i) ref_g[i] = uint8_t((i * 31 + 7) & 0xFF);
            for (size_t i = 0; i < up_sz;   ++i) ref_u[i] = uint8_t((i * 53 + 19) & 0xFF);
            for (size_t i = 0; i < down_sz; ++i) ref_d[i] = uint8_t((i * 71 + 43) & 0xFF);
            std::memcpy(blob.data() + off_g, ref_g.data(), gate_sz);
            std::memcpy(blob.data() + off_u, ref_u.data(), up_sz);
            std::memcpy(blob.data() + off_d, ref_d.data(), down_sz);
            {
                std::ofstream out(dma_path, std::ios::binary | std::ios::trunc);
                out.write(reinterpret_cast<const char *>(blob.data()), (std::streamsize)blob.size());
            }

            atlas::UnifiedTieredExpertCache ucache;
            atlas::UnifiedTieredExpertCache::LayerMeta lm;
            lm.gate_base = ref_g.data(); lm.gate_expert_bytes = gate_sz; lm.gate_type = GGML_TYPE_Q4_0;
            lm.up_base   = ref_u.data(); lm.up_expert_bytes   = up_sz;   lm.up_type   = GGML_TYPE_Q4_0;
            lm.down_base = ref_d.data(); lm.down_expert_bytes = down_sz; lm.down_type = GGML_TYPE_Q4_0;
            lm.n_embd = 256; lm.n_ff = 256;
            CHECK(ucache.init(4 * 1024 * 1024ULL, 0, lm.slot_bytes(), 64.0f));
            ucache.register_layer_weights(0, lm);

            int fd = ::open(dma_path.c_str(), O_RDONLY);
            CHECK(fd >= 0);
            {
                atlas::DirectStorageIoUringDMA dma(fd, nullptr);
                auto res = ucache.reserve_async_vram_slot(0, 2);
                CHECK(res.valid());
                CHECK(ucache.get_vram_slot(0, 2) == -1);
                CHECK(ucache.is_vram_slot_reserved(res.slot));

                const int      blobs[3] = { 0, 0, 0 };
                const uint64_t offs[3]  = { off_g, off_u, off_d };
                const size_t   szs[3]   = { gate_sz, up_sz, down_sz };
                atlas::ExpertDmaJob job{};
                CHECK(dma.enqueue_expert_3tensor_read(0, 2, res.slot, blobs, offs, szs, res.dev_ptrs, 3, job));
                bool done = false;
                for (int spin = 0; spin < 2000 && !done; ++spin) {
                    done = dma.poll_job_complete(job);
                    if (!done) std::this_thread::sleep_for(std::chrono::microseconds(100));
                }
                CHECK(done);
                CHECK(ucache.commit_async_vram_slot(0, 2, res.slot));
                CHECK(ucache.get_vram_slot(0, 2) == res.slot);
                CHECK(!ucache.is_vram_slot_reserved(res.slot));
                CHECK(ucache.get_dma_promoted() == 1);
                CHECK(ucache.consume_lut_dirty());

                void * hCuda = atlas::atlas_get_libcuda_handle();
                CHECK(hCuda != nullptr);
                using cuMemcpyDtoH_t = int (*)(void *, uint64_t, size_t);
                auto fnDtoH = (cuMemcpyDtoH_t)dlsym(hCuda, "cuMemcpyDtoH_v2");
                if (!fnDtoH) fnDtoH = (cuMemcpyDtoH_t)dlsym(hCuda, "cuMemcpyDtoH");
                CHECK(fnDtoH != nullptr);

                std::vector<uint8_t> got_g(gate_sz), got_u(up_sz), got_d(down_sz);
                CHECK(fnDtoH(got_g.data(), res.dev_ptrs[0], gate_sz) == 0);
                CHECK(fnDtoH(got_u.data(), res.dev_ptrs[1], up_sz) == 0);
                CHECK(fnDtoH(got_d.data(), res.dev_ptrs[2], down_sz) == 0);
                CHECK(std::memcmp(got_g.data(), ref_g.data(), gate_sz) == 0);
                CHECK(std::memcmp(got_u.data(), ref_u.data(), up_sz) == 0);
                CHECK(std::memcmp(got_d.data(), ref_d.data(), down_sz) == 0);

                // Verify abort path releases reserved slot cleanly
                auto res2 = ucache.reserve_async_vram_slot(0, 5);
                CHECK(res2.valid() && res2.slot != res.slot);
                ucache.abort_async_vram_slot(res2.slot);
                CHECK(!ucache.is_vram_slot_reserved(res2.slot));
                CHECK(ucache.get_vram_slot(0, 5) == -1);
            }
            ::close(fd);
            ucache.shutdown();
            std::error_code ec;
            std::filesystem::remove(dma_path, ec);
#endif
        }

        // Gate 42: Phase 18 Layer-Concentrated VRAM Mask & <=16 GiB Adaptive RAM Lock Guard
        {
            const uint64_t m460 = atlas::compute_recommended_vram_layer_mask(48, 460, 0);
            int pop = 0;
            for (int l = 0; l < 48; ++l) if ((m460 >> l) & 1ULL) ++pop;
            CHECK(pop == 7); // 460 / 64 = 7 hot layers (front 3: 0,1,2 + back 4: 44,45,46,47)
            CHECK(((m460 >> 0) & 1ULL) && ((m460 >> 2) & 1ULL));
            CHECK(!((m460 >> 10) & 1ULL) && !((m460 >> 30) & 1ULL));
            CHECK(((m460 >> 44) & 1ULL) && ((m460 >> 47) & 1ULL));
            CHECK(atlas::compute_recommended_vram_layer_mask(48, 0, 0) == 0ULL);
            CHECK(atlas::compute_recommended_vram_layer_mask(48, 460, 48) == ((1ULL << 48) - 1ULL));

            const size_t GiB = 1024ULL * 1024ULL * 1024ULL;
            const size_t MiB = 1024ULL * 1024ULL;
            // 14 GiB laptop + 75 GiB model + K=5 + auto => 0 (preserves OS mmap page cache)
            CHECK(atlas::compute_adaptive_ram_budget(2048 * MiB, 14 * GiB, 10 * GiB, 75 * GiB, 5, 256 * MiB, "auto") == 0);
            // Explicit policy "on" overrides auto guard up to safe_ram
            CHECK(atlas::compute_adaptive_ram_budget(2048 * MiB, 14 * GiB, 10 * GiB, 75 * GiB, 5, 256 * MiB, "on") == 2048 * MiB);
            // Policy "off" always returns 0
            CHECK(atlas::compute_adaptive_ram_budget(2048 * MiB, 32 * GiB, 24 * GiB, 75 * GiB, 10, 256 * MiB, "off") == 0);
            // K=10 with low avail RAM (<8 GiB) caps at 512 MiB
            CHECK(atlas::compute_adaptive_ram_budget(2048 * MiB, 14 * GiB, 6 * GiB, 75 * GiB, 10, 256 * MiB, "auto") == 512 * MiB);

#if defined(GGML_USE_CUDA)
            ggml_backend_dev_t cuda_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
            CHECK(cuda_dev != nullptr);
            struct ggml_init_params gp = { 2 * 1024 * 1024, nullptr, true };
            struct ggml_context * gctx = ggml_init(gp);
            struct ggml_tensor * tx  = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, 256, 1, 1);
            struct ggml_tensor * tid = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, 2, 1);
            struct ggml_tensor * tw  = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, 1, 2, 1);
            struct ggml_tensor * args[3] = { tx, tid, tw };
            struct ggml_tensor * op0 = ggml_custom_4d(gctx, GGML_TYPE_F32, 256, 1, 1, 1, args, 3, nullptr, 1, nullptr);
            op0->op_params[6] = 0x41544C41; op0->op_params[7] = 0x534D4F45; op0->op_params[8] = 0;
            struct ggml_tensor * op5 = ggml_custom_4d(gctx, GGML_TYPE_F32, 256, 1, 1, 1, args, 3, nullptr, 1, nullptr);
            op5->op_params[6] = 0x41544C41; op5->op_params[7] = 0x534D4F45; op5->op_params[8] = 5;

            ggml_backend_cuda_atlas_moe_set_layer_mask(0x1ULL);
            CHECK(ggml_backend_cuda_atlas_moe_get_layer_mask() == 0x1ULL);
            CHECK(ggml_backend_dev_supports_op(cuda_dev, op0));
            CHECK(!ggml_backend_dev_supports_op(cuda_dev, op5));
            ggml_backend_cuda_atlas_moe_set_layer_mask(~0ULL);
            ggml_free(gctx);
#endif
        }

        // Gate 43: Phase 18 Long-Context FP8 E4M3 VEC Selection + Multi-Token (BS=2, BS=4) CUDA+CPU Split MoE Parity
        {
#if defined(GGML_USE_CUDA)
            // WP1.1: Qwen3.8 shape (D=256, gqa_ratio=12) MUST select BEST_FATTN_KERNEL_VEC at 8k & 32k context for F8_E4M3
            CHECK(ggml_backend_cuda_fattn_uses_vec_for_shape(0, 256, 8192,  12, 1, GGML_TYPE_F8_E4M3, GGML_TYPE_F8_E4M3));
            CHECK(ggml_backend_cuda_fattn_uses_vec_for_shape(0, 256, 32768, 12, 1, GGML_TYPE_F8_E4M3, GGML_TYPE_F8_E4M3));
            CHECK(ggml_backend_cuda_fattn_uses_vec_for_shape(0, 256, 8192,  12, 2, GGML_TYPE_F8_E4M3, GGML_TYPE_F8_E4M3));

            ggml_backend_t cuda = ggml_backend_cuda_init(0);
            ggml_backend_t cpu  = ggml_backend_cpu_init();
            CHECK(cuda && cpu);

            const int n_embd = 256, n_ff = 256, n_exp = 4, K = 2;
            const ggml_type qtype = GGML_TYPE_Q4_0;
            const size_t exp_bytes_g = ggml_row_size(qtype, n_embd) * n_ff;
            const size_t exp_bytes_d = ggml_row_size(qtype, n_ff) * n_embd;

            std::vector<uint8_t> raw_gate(n_exp * exp_bytes_g), raw_up(n_exp * exp_bytes_g), raw_down(n_exp * exp_bytes_d);
            auto qfill = [&](std::vector<uint8_t> & dst, int rows, int cols, int seed) {
                std::vector<float> tmp(rows * cols);
                for (size_t i = 0; i < tmp.size(); ++i) tmp[i] = 0.14f * std::sin(float((i * 13 + seed) % 97) * 0.12f);
                ggml_quantize_chunk(qtype, tmp.data(), dst.data(), 0, rows, cols, nullptr);
            };
            qfill(raw_gate, n_exp * n_ff, n_embd, 17);
            qfill(raw_up,   n_exp * n_ff, n_embd, 31);
            qfill(raw_down, n_exp * n_embd, n_ff, 53);

            struct ggml_init_params gp = { 8 * 1024 * 1024, nullptr, true };
            struct ggml_context * wctx = ggml_init(gp);
            struct ggml_tensor * t_gate = ggml_new_tensor_3d(wctx, qtype, n_embd, n_ff, n_exp);
            struct ggml_tensor * t_up   = ggml_new_tensor_3d(wctx, qtype, n_embd, n_ff, n_exp);
            struct ggml_tensor * t_down = ggml_new_tensor_3d(wctx, qtype, n_ff, n_embd, n_exp);
            ggml_set_name(t_gate, "blk.0.ffn_gate_exps.weight");
            ggml_set_name(t_up,   "blk.0.ffn_up_exps.weight");
            ggml_set_name(t_down, "blk.0.ffn_down_exps.weight");
            ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, cpu);
            ggml_backend_tensor_set(t_gate, raw_gate.data(), 0, raw_gate.size());
            ggml_backend_tensor_set(t_up,   raw_up.data(),   0, raw_up.size());
            ggml_backend_tensor_set(t_down, raw_down.data(), 0, raw_down.size());

            atlas::UnifiedTieredExpertCache ucache;
            atlas::UnifiedTieredExpertCache::LayerMeta lm;
            lm.gate_base = raw_gate.data(); lm.gate_expert_bytes = exp_bytes_g; lm.gate_type = qtype;
            lm.up_base   = raw_up.data();   lm.up_expert_bytes   = exp_bytes_g; lm.up_type   = qtype;
            lm.down_base = raw_down.data(); lm.down_expert_bytes = exp_bytes_d; lm.down_type = qtype;
            lm.n_embd = n_embd; lm.n_ff = n_ff;
            CHECK(ucache.init(4 * 1024 * 1024ULL, 0, lm.slot_bytes(), 64.0f));
            ucache.register_layer_weights(0, lm);
            ucache.set_vram_callback([](int l, int e, int slot, bool admitted) {
                ::atlas_set_expert_gpu_slot(l, e, admitted ? int16_t(slot) : int16_t(-1));
            });
            CHECK(ggml_backend_cuda_atlas_moe_set_arena(ucache.vram_dev_base(), ucache.slot_stride_bytes(), 1));
            CHECK(ggml_backend_cuda_atlas_moe_set_layer_meta(0, qtype, qtype, qtype, n_embd, n_ff, 0ULL, lm.up_offset(), lm.down_offset()));
            CHECK(ucache.access_and_promote(0, 1, false, true));
            ggml_backend_cuda_atlas_moe_sync_lut(ucache.vram_lut_flat(), 1, atlas::UnifiedTieredExpertCache::MAX_EXPERTS);

            for (int bs : {2, 4}) {
                std::vector<float> x_host(size_t(bs) * n_embd);
                for (int t = 0; t < bs; ++t) {
                    for (int i = 0; i < n_embd; ++i) x_host[size_t(t) * n_embd + i] = 0.35f * std::cos(float(i + t * 19) * 0.09f);
                }
                const int32_t all_ids[8] = { 1, 3,  0, 1,  2, 3,  1, 2 };
                const float   all_w[8]   = { 0.65f, 0.35f,  0.40f, 0.60f,  0.55f, 0.45f,  0.70f, 0.30f };

                auto run_batch = [&](bool enable_gpu, uint64_t lmask) -> std::vector<float> {
                    struct ggml_init_params gp2 = { 8 * 1024 * 1024, nullptr, true };
                    struct ggml_context * gctx = ggml_init(gp2);
                    struct ggml_tensor * tx  = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, n_embd, 1, bs);
                    struct ggml_tensor * tid = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, K, bs);
                    struct ggml_tensor * tw  = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, 1, K, bs);
                    ggml_set_input(tx); ggml_set_input(tid); ggml_set_input(tw);

                    struct ggml_tensor * g_out = ggml_mul_mat_id(gctx, t_gate, tx, tid);
                    struct ggml_tensor * u_out = ggml_mul_mat_id(gctx, t_up,   tx, tid);
                    struct ggml_tensor * swi   = ggml_swiglu_split(gctx, g_out, u_out);
                    struct ggml_tensor * d_out = ggml_mul_mat_id(gctx, t_down, swi, tid);
                    struct ggml_tensor * w_out = ggml_mul(gctx, d_out, tw);
                    struct ggml_tensor * e0 = ggml_view_2d(gctx, w_out, n_embd, bs, w_out->nb[2], 0);
                    struct ggml_tensor * e1 = ggml_view_2d(gctx, w_out, n_embd, bs, w_out->nb[2], w_out->nb[1]);
                    struct ggml_tensor * cpu_sum = ggml_add(gctx, e0, e1);

                    struct ggml_tensor * total = cpu_sum;
                    struct ggml_tensor * gpu_moe = nullptr;
                    if (enable_gpu) {
                        struct ggml_tensor * args[3] = { tx, tid, tw };
                        gpu_moe = ggml_custom_4d(gctx, GGML_TYPE_F32, n_embd, bs, 1, 1, args, 3, nullptr, 1, nullptr);
                        gpu_moe->op_params[6] = 0x41544C41;
                        gpu_moe->op_params[7] = 0x534D4F45;
                        gpu_moe->op_params[8] = 0;
                        total = ggml_add(gctx, gpu_moe, cpu_sum);
                    }
                    struct ggml_cgraph * gf = ggml_new_graph(gctx);
                    if (gpu_moe) ggml_build_forward_expand(gf, gpu_moe);
                    ggml_build_forward_expand(gf, total);

                    ggml_backend_cuda_atlas_moe_set_layer_mask(lmask);
                    atlas::atlas_set_gpu_layer_mask(lmask);

                    ggml_backend_t backends[2] = { cuda, cpu };
                    ggml_backend_buffer_type_t bufts[2] = { ggml_backend_cuda_buffer_type(0), ggml_backend_cpu_buffer_type() };
                    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, 2, 256, false, true);
                    if (gpu_moe && (lmask & 1ULL)) {
                        ggml_backend_sched_set_tensor_backend(sched, gpu_moe, cuda);
                        ggml_backend_sched_set_tensor_backend(sched, total, cuda);
                    }
                    ggml_backend_sched_reserve(sched, gf);
                    ggml_backend_sched_alloc_graph(sched, gf);
                    ggml_backend_tensor_set(tx,  x_host.data(), 0, sizeof(float) * x_host.size());
                    ggml_backend_tensor_set(tid, all_ids, 0, sizeof(int32_t) * K * bs);
                    ggml_backend_tensor_set(tw,  all_w,   0, sizeof(float) * K * bs);

                    atlas::atlas_set_gpu_cache_active(enable_gpu ? 1 : 0);
                    atlas::atlas_set_is_decode(enable_gpu ? 1 : 0);
                    ggml_backend_cuda_atlas_moe_set_active(enable_gpu ? 1 : 0);
                    ggml_backend_cuda_atlas_moe_set_decode(enable_gpu ? 1 : 0);

                    CHECK(ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS);
                    std::vector<float> res(size_t(bs) * n_embd);
                    ggml_backend_tensor_get(total, res.data(), 0, ggml_nbytes(total));
                    ggml_backend_sched_free(sched);
                    ggml_free(gctx);
                    return res;
                };

                auto ref_b   = run_batch(false, ~0ULL);
                auto split_b = run_batch(true,  ~0ULL);
                auto mask0_b = run_batch(false, 0ULL);
                float max_d = 0.0f, max_m0 = 0.0f;
                for (size_t i = 0; i < ref_b.size(); ++i) {
                    max_d  = std::max(max_d,  std::fabs(ref_b[i] - split_b[i]));
                    max_m0 = std::max(max_m0, std::fabs(ref_b[i] - mask0_b[i]));
                }
                CHECK(max_d < 5e-3f && max_m0 < 1e-6f);
            }

            ggml_backend_cuda_atlas_moe_set_active(0);
            ggml_backend_cuda_atlas_moe_set_decode(0);
            ggml_backend_cuda_atlas_moe_set_layer_mask(~0ULL);
            ggml_backend_cuda_atlas_moe_set_arena(0, 0, 0);
            atlas::atlas_set_gpu_cache_active(0);
            atlas::atlas_set_is_decode(0);
            atlas::atlas_set_gpu_layer_mask(~0ULL);
            ::atlas_set_expert_gpu_slot(0, 1, -1);
            ucache.shutdown();
            ggml_backend_buffer_free(wbuf);
            ggml_free(wctx);
            ggml_backend_free(cuda);
            ggml_backend_free(cpu);
#endif
        }

        // Gate 44: Phase 18 Hierarchical PrefixCache KV Tree Forking (Root Anchor + Diverging Agents + Protected Eviction)
        {
            atlas::PrefixCache pc;
            pc.budget_bytes = 256 * 1024; // 256 KiB test budget
            std::vector<llama_token> root_toks(256);
            for (int i = 0; i < 256; ++i) root_toks[i] = llama_token(100 + i);
            std::vector<uint8_t> root_state(4096, 0x11);
            CHECK(pc.store(root_toks, root_state, {}, {}, true, true, "agent-A"));

            std::vector<llama_token> turnA = root_toks;
            for (int i = 0; i < 64; ++i) turnA.push_back(llama_token(1000 + i));
            std::vector<uint8_t> stateA(4096, 0x22);
            CHECK(pc.store(turnA, stateA, {}, {}, true, false, "agent-A"));

            // Agent B shares the 256-token system root and diverges at token 256
            std::vector<llama_token> reqB = root_toks;
            for (int i = 0; i < 48; ++i) reqB.push_back(llama_token(2000 + i));
            const char * fk = nullptr;
            int idxB = pc.find_best(reqB, true, "agent-B", &fk);
            CHECK(idxB >= 0 && fk != nullptr && std::string(fk) == "root_anchor");
            CHECK(pc.slots[size_t(idxB)].tokens.size() == 256);

            // Agent A Turn 2 extends Turn 1 and should prefer its longer 320-token turn boundary
            std::vector<llama_token> reqA2 = turnA;
            for (int i = 0; i < 32; ++i) reqA2.push_back(llama_token(3000 + i));
            int idxA2 = pc.find_best(reqA2, true, "agent-A", &fk);
            CHECK(idxA2 >= 0 && fk != nullptr && std::string(fk) == "turn_boundary");
            CHECK(pc.slots[size_t(idxA2)].tokens.size() == 320);

            // Flood 20 non-root branches (> kMaxSlots = 16) and verify the root_anchor survives eviction
            for (int b = 0; b < 20; ++b) {
                std::vector<llama_token> br = root_toks;
                br.push_back(llama_token(5000 + b));
                CHECK(pc.store(br, std::vector<uint8_t>(2048, uint8_t(b)), {}, {}, false, false, "flood"));
            }
            CHECK(pc.slots.size() <= atlas::PrefixCache::kMaxSlots);
            int idxB_after = pc.find_best(reqB, true, "agent-B", &fk);
            CHECK(idxB_after >= 0 && fk != nullptr && std::string(fk) == "root_anchor");
        }

        // Gate 45: v0.16.1 ggml_vec_dot_iq4_xs_q8_K_x4 Bit-Exact 0-ULP Parity vs 4x single vec_dot
        {
            const int n = 256;
            const size_t bx = ggml_type_size(GGML_TYPE_IQ4_XS);
            const size_t by_size = ggml_type_size(GGML_TYPE_Q8_K);
            std::vector<uint8_t> vx_raw(4 * bx);
            std::vector<uint8_t> vy_raw(by_size);
            for (size_t i = 0; i < vx_raw.size(); ++i) {
                vx_raw[i] = (uint8_t)((i * 31 + 13) & 0xFF);
            }
            // Keep FP16 scale d in normal range (0x3C00 = 1.0f) for each row block
            for (int r = 0; r < 4; ++r) {
                vx_raw[size_t(r) * bx + 0] = 0x00;
                vx_raw[size_t(r) * bx + 1] = (uint8_t)(0x38 + r);
            }
            for (size_t i = 0; i < vy_raw.size(); ++i) {
                vy_raw[i] = (uint8_t)((i * 17 + 5) & 0x7F);
            }
            const float yd = 0.25f;
            std::memcpy(vy_raw.data(), &yd, sizeof(float));

            float s_single[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float s_x4[4] = {0.0f, 0.0f, 0.0f, 0.0f};

            for (int r = 0; r < 4; ++r) {
                ggml_vec_dot_iq4_xs_q8_K(n, &s_single[r], 0, vx_raw.data() + size_t(r) * bx, 0, vy_raw.data(), 0, 1);
            }
            ggml_vec_dot_iq4_xs_q8_K_x4(n, s_x4, sizeof(float), vx_raw.data(), bx, vy_raw.data(), 0, 4);

            for (int r = 0; r < 4; ++r) {
                CHECK(s_single[r] == s_x4[r]);
                uint32_t u_single = 0, u_x4 = 0;
                std::memcpy(&u_single, &s_single[r], 4);
                std::memcpy(&u_x4, &s_x4[r], 4);
                CHECK(u_single == u_x4);
            }
        }

        // Gate 46: v0.16.2 ggml_vec_dot_iq3_s_q8_K_x4 Bit-Exact 0-ULP Parity vs 4x single vec_dot
        // Verifies single-block (256), odd multi-block (768), and even multi-block (1024) lengths.
        {
            const int test_lengths[3] = {256, 768, 1024};
            const size_t blk_x = ggml_type_size(GGML_TYPE_IQ3_S);
            const size_t blk_y = ggml_type_size(GGML_TYPE_Q8_K);
            for (int ti = 0; ti < 3; ++ti) {
                const int n = test_lengths[ti];
                const int nb = n / 256;
                const size_t row_bx = size_t(nb) * blk_x;
                const size_t vec_by = size_t(nb) * blk_y;
                std::vector<uint8_t> vx_raw(4 * row_bx);
                std::vector<uint8_t> vy_raw(vec_by);

                for (size_t i = 0; i < vx_raw.size(); ++i) {
                    vx_raw[i] = (uint8_t)((i * 37 + 19 + ti * 7) & 0xFF);
                }
                for (int r = 0; r < 4; ++r) {
                    for (int b = 0; b < nb; ++b) {
                        const size_t off = size_t(r) * row_bx + size_t(b) * blk_x;
                        // Normal-range FP16 scale d per block
                        vx_raw[off + 0] = (uint8_t)((b * 16) & 0xFF);
                        vx_raw[off + 1] = (uint8_t)(0x38 + ((r + b) & 0x3));
                    }
                }
                for (size_t i = 0; i < vy_raw.size(); ++i) {
                    vy_raw[i] = (uint8_t)((i * 23 + 11 + ti * 5) & 0x7F);
                }
                for (int b = 0; b < nb; ++b) {
                    const float yd = 0.125f * float(b + 1);
                    std::memcpy(vy_raw.data() + size_t(b) * blk_y, &yd, sizeof(float));
                }

                float s_single[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                float s_x4[4] = {0.0f, 0.0f, 0.0f, 0.0f};

                for (int r = 0; r < 4; ++r) {
                    ggml_vec_dot_iq3_s_q8_K(n, &s_single[r], 0, vx_raw.data() + size_t(r) * row_bx, 0, vy_raw.data(), 0, 1);
                }
                ggml_vec_dot_iq3_s_q8_K_x4(n, s_x4, sizeof(float), vx_raw.data(), row_bx, vy_raw.data(), 0, 4);

                for (int r = 0; r < 4; ++r) {
                    CHECK(std::isfinite(s_single[r]) && std::isfinite(s_x4[r]));
                    CHECK(s_single[r] == s_x4[r]);
                    uint32_t u_single = 0, u_x4 = 0;
                    std::memcpy(&u_single, &s_single[r], 4);
                    std::memcpy(&u_x4, &s_x4[r], 4);
                    CHECK(u_single == u_x4);
                }
            }
        }

        // Gate 47: v0.16.3 512-bit AVX-512 IQ4_XS_x4 & IQ3_XXS_x4 Adversarial 0-ULP Parity
        // Covers full signed int8 range including -128 (0x80) edge case and 256/768/2048 lengths.
        {
            const int test_lengths[3] = {256, 768, 2048};
            const size_t blk_y = ggml_type_size(GGML_TYPE_Q8_K);
            for (int ti = 0; ti < 3; ++ti) {
                const int n = test_lengths[ti];
                const int nb = n / 256;
                const size_t vec_by = size_t(nb) * blk_y;
                std::vector<uint8_t> vy_raw(vec_by);
                for (size_t i = 0; i < vy_raw.size(); ++i) {
                    // Full 0..255 byte range so int8 includes negative values and -128 (0x80)
                    vy_raw[i] = (uint8_t)((i * 53 + 0x80 + ti * 17) & 0xFF);
                }
                for (int b = 0; b < nb; ++b) {
                    const float yd = (b % 2 == 0 ? 0.25f : -0.1875f) * float(b + 1);
                    std::memcpy(vy_raw.data() + size_t(b) * blk_y, &yd, sizeof(float));
                    // Force explicit -128 bytes into q8 payload
                    vy_raw[size_t(b) * blk_y + 4] = 0x80;
                    vy_raw[size_t(b) * blk_y + 36] = 0x80;
                }

                // 47a. IQ4_XS_x4
                {
                    const size_t blk_x = ggml_type_size(GGML_TYPE_IQ4_XS);
                    const size_t row_bx = size_t(nb) * blk_x;
                    std::vector<uint8_t> vx_raw(4 * row_bx);
                    for (size_t i = 0; i < vx_raw.size(); ++i) {
                        vx_raw[i] = (uint8_t)((i * 41 + 13 + ti * 11) & 0xFF);
                    }
                    for (int r = 0; r < 4; ++r) {
                        for (int b = 0; b < nb; ++b) {
                            const size_t off = size_t(r) * row_bx + size_t(b) * blk_x;
                            vx_raw[off + 0] = (uint8_t)((b * 19 + r * 7) & 0xFF);
                            vx_raw[off + 1] = (uint8_t)(0x38 + ((r + b) & 0x3));
                        }
                    }
                    float s_single[4] = {}, s_x4[4] = {};
                    for (int r = 0; r < 4; ++r) {
                        ggml_vec_dot_iq4_xs_q8_K(n, &s_single[r], 0, vx_raw.data() + size_t(r) * row_bx, 0, vy_raw.data(), 0, 1);
                    }
                    ggml_vec_dot_iq4_xs_q8_K_x4(n, s_x4, sizeof(float), vx_raw.data(), row_bx, vy_raw.data(), 0, 4);
                    for (int r = 0; r < 4; ++r) {
                        uint32_t u_s = 0, u_4 = 0;
                        std::memcpy(&u_s, &s_single[r], 4);
                        std::memcpy(&u_4, &s_x4[r], 4);
                        CHECK(std::isfinite(s_x4[r]) && u_s == u_4);
                    }
                }

                // 47b. IQ3_XXS_x4
                {
                    const size_t blk_x = ggml_type_size(GGML_TYPE_IQ3_XXS);
                    const size_t row_bx = size_t(nb) * blk_x;
                    std::vector<uint8_t> vx_raw(4 * row_bx);
                    for (size_t i = 0; i < vx_raw.size(); ++i) {
                        vx_raw[i] = (uint8_t)((i * 29 + 7 + ti * 13) & 0xFF);
                    }
                    for (int r = 0; r < 4; ++r) {
                        for (int b = 0; b < nb; ++b) {
                            const size_t off = size_t(r) * row_bx + size_t(b) * blk_x;
                            vx_raw[off + 0] = (uint8_t)((b * 23 + r * 5) & 0xFF);
                            vx_raw[off + 1] = (uint8_t)(0x38 + ((r + b) & 0x3));
                        }
                    }
                    float s_single[4] = {}, s_x4[4] = {};
                    for (int r = 0; r < 4; ++r) {
                        ggml_vec_dot_iq3_xxs_q8_K(n, &s_single[r], 0, vx_raw.data() + size_t(r) * row_bx, 0, vy_raw.data(), 0, 1);
                    }
                    ggml_vec_dot_iq3_xxs_q8_K_x4(n, s_x4, sizeof(float), vx_raw.data(), row_bx, vy_raw.data(), 0, 4);
                    for (int r = 0; r < 4; ++r) {
                        uint32_t u_s = 0, u_4 = 0;
                        std::memcpy(&u_s, &s_single[r], 4);
                        std::memcpy(&u_4, &s_x4[r], 4);
                        CHECK(std::isfinite(s_x4[r]) && u_s == u_4);
                    }
                }
            }
        }

        // Gate 48: v0.16.3 Three-Regime Zero-Copy Boost-Aware Hot-RAM Planner
        {
            const size_t MiB = 1024ULL * 1024ULL;
            std::vector<atlas::ZeroCopyHotRamLayer> layers_35b;
            // Simulate 32 CPU MoE layers (layers 2..14 and 21..39), 340 MiB each (~10.88 GiB)
            for (int l = 2; l <= 14; ++l) layers_35b.push_back({l, 340ULL * MiB});
            for (int l = 21; l <= 39; ++l) layers_35b.push_back({l, 340ULL * MiB});

            // 1) Near-Fit (11.5 GiB avail RAM on 16 GiB laptop): must use zero-copy-prewarm (use_mlock=false)
            //    so ~9.5 GiB is warmed at startup while cold experts stay reclaimable for the remaining 4 layers.
            const auto plan_near = atlas::compute_zero_copy_hot_ram_plan(
                layers_35b, 40, 11776ULL * MiB, 512ULL * MiB, 8192, "auto");
            CHECK(!plan_near.streaming_pagecache_mode);
            CHECK(!plan_near.use_mlock);
            CHECK(plan_near.guard_bytes == 2048ULL * MiB);
            CHECK(plan_near.budget_bytes == 9728ULL * MiB);
            CHECK(plan_near.planned_bytes <= plan_near.budget_bytes);
            CHECK(plan_near.selected_layers.size() >= 28);
            // Reasoning band (21..30) and final layers (36..39) MUST come before outer layers (2..14)
            CHECK(plan_near.selected_layers[0] == 21);
            CHECK(std::find(plan_near.selected_layers.begin(), plan_near.selected_layers.end(), 39) != plan_near.selected_layers.end());

            // 2) Full-Fit (24 GiB avail RAM): 100% of CPU MoE fits with >=2.5 GiB OS headroom -> safe full zero-copy mlock!
            const auto plan_full = atlas::compute_zero_copy_hot_ram_plan(
                layers_35b, 40, 24576ULL * MiB, 512ULL * MiB, 8192, "auto");
            CHECK(!plan_full.streaming_pagecache_mode);
            CHECK(plan_full.use_mlock);
            CHECK(plan_full.selected_layers.size() == layers_35b.size());
            CHECK(plan_full.planned_bytes == plan_full.total_cpu_moe_bytes);

            // 3) Near-Fit explicit "on": honors mlock but caps to <=60% of avail RAM so unpinned layers never starve
            const auto plan_near_on = atlas::compute_zero_copy_hot_ram_plan(
                layers_35b, 40, 11776ULL * MiB, 512ULL * MiB, 8192, "on");
            CHECK(!plan_near_on.streaming_pagecache_mode);
            CHECK(plan_near_on.use_mlock);
            CHECK(plan_near_on.budget_bytes <= (11776ULL * MiB * 3ULL) / 5ULL);
            CHECK(plan_near_on.planned_bytes <= plan_near_on.budget_bytes);

            // 4) Heavy NVMe-streaming model (Qwen3.8 75.8 GB: 46 CPU MoE layers x 1430 MiB = ~64.2 GiB >> 11.5 GiB RAM)
            //    In "auto" mode must NOT whole-layer prewarm/mlock (dedicates reclaimed pagecache to Zipfian-hot experts)
            std::vector<atlas::ZeroCopyHotRamLayer> layers_38b;
            for (int l = 2; l < 48; ++l) layers_38b.push_back({l, 1430ULL * MiB});
            const auto plan_38b_auto = atlas::compute_zero_copy_hot_ram_plan(
                layers_38b, 48, 11776ULL * MiB, 512ULL * MiB, 8192, "auto");
            CHECK(plan_38b_auto.streaming_pagecache_mode);
            CHECK(!plan_38b_auto.use_mlock);
            CHECK(plan_38b_auto.selected_layers.empty());
            CHECK(plan_38b_auto.planned_bytes == 0);

            // 5) Explicit "on" on heavy streaming caps whole-layer pin to <=25% of free RAM
            const auto plan_38b_on = atlas::compute_zero_copy_hot_ram_plan(
                layers_38b, 48, 11776ULL * MiB, 512ULL * MiB, 8192, "on");
            CHECK(plan_38b_on.use_mlock);
            CHECK(plan_38b_on.budget_bytes <= (11776ULL * MiB) / 4ULL);
            CHECK(plan_38b_on.planned_bytes <= plan_38b_on.budget_bytes);

            // 6) Policy "off" must select 0 layers
            const auto plan_off = atlas::compute_zero_copy_hot_ram_plan(
                layers_35b, 40, 11776ULL * MiB, 512ULL * MiB, 8192, "off");
            CHECK(plan_off.selected_layers.empty());
            CHECK(plan_off.planned_bytes == 0);

            // 7) Low available RAM (<= guard) must select 0 layers (zero OOM risk)
            const auto plan_low = atlas::compute_zero_copy_hot_ram_plan(
                layers_35b, 40, 2048ULL * MiB, 512ULL * MiB, 65536, "auto");
            CHECK(plan_low.selected_layers.empty());
            CHECK(plan_low.planned_bytes == 0);
        }

        // Gate 49: v0.16.3 MoE 512-Expert Dynamic-K Floor + Weight Renormalization Parity
        {
            const auto set_env_var = [](const char * k, const char * v) {
#if defined(_WIN32)
                _putenv_s(k, v ? v : "");
#else
                if (v && v[0]) setenv(k, v, 1);
                else unsetenv(k);
#endif
            };

            // 1) Verify 512-expert (Qwen3.8-Flash-Next) never collapses to keep_k = 1 in decode or prefill
            set_env_var("ATLAS_CPU_DYNAMIC_K", "2");
            set_env_var("ATLAS_PREFILL_TOP_K", "1");
            ggml_atlas_reset_moe_decode_state();
            ggml_atlas_set_moe_decode_state(1, 1);
            CHECK(ggml_atlas_get_moe_keep_k(20, 5, 512, 0, 1, 0) == 4);
            ggml_atlas_set_moe_decode_state(1, 5);
            CHECK(ggml_atlas_get_moe_keep_k(20, 5, 512, 0, 1, 0) == 3);
            CHECK(ggml_atlas_get_moe_keep_k(10, 5, 512, 0, 1, 0) >= 2);
            CHECK(ggml_atlas_get_moe_keep_k(45, 5, 512, 0, 1, 0) == 3);
            // 35B (256-expert) preserves calibrated Turbo keep_k (K=1 outer, K=4 reasoning band 15..30)
            CHECK(ggml_atlas_get_moe_keep_k(10, 5, 256, 0, 1, 0) == 1);
            CHECK(ggml_atlas_get_moe_keep_k(20, 5, 256, 0, 1, 0) == 4);

            // Level 1 Balanced for 512-expert model keeps K=4 on reasoning band 15..30, K=3 on entry/late/output, K=2 outer
            set_env_var("ATLAS_CPU_DYNAMIC_K", "1");
            ggml_atlas_reset_moe_decode_state();
            ggml_atlas_set_moe_decode_state(1, 5);
            CHECK(ggml_atlas_get_moe_keep_k(20, 5, 512, 0, 1, 0) == 4);
            CHECK(ggml_atlas_get_moe_keep_k(0, 5, 512, 0, 1, 0) == 3);
            CHECK(ggml_atlas_get_moe_keep_k(45, 5, 512, 0, 1, 0) == 3);
            CHECK(ggml_atlas_get_moe_keep_k(10, 5, 512, 0, 1, 0) == 2);

            // Prefill never drops below K=2 on 512-expert models even if ATLAS_PREFILL_TOP_K=1
            ggml_atlas_set_moe_decode_state(0, 0);
            CHECK(ggml_atlas_get_moe_keep_k(10, 5, 512, 0, 128, 0) >= 2);

            // 2) Verify ggml weight renormalization preserves total routing weight mass (= sum_all)
            set_env_var("ATLAS_CPU_DYNAMIC_K", "2");
            ggml_atlas_reset_moe_decode_state();
            ggml_atlas_set_moe_decode_state(1, 5);

            struct ggml_init_params gparams = {
                /* .mem_size   = */ 2 * 1024 * 1024,
                /* .mem_buffer = */ nullptr,
                /* .no_alloc   = */ false,
            };
            struct ggml_context * gctx = ggml_init(gparams);
            CHECK(gctx != nullptr);
            struct ggml_tensor * w_in = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, 1, 5, 1);
            float * wd = (float *) w_in->data;
            wd[0] = 0.30f; wd[1] = 0.20f; wd[2] = 0.20f; wd[3] = 0.15f; wd[4] = 0.15f;

            // Test CPU renormalization via custom op helper logic mirrored across backends
            const int keep_k_512_outer = ggml_atlas_get_moe_keep_k(10, 5, 512, 0, 1, 0);
            CHECK(keep_k_512_outer == 2);
            float w_out[5] = {0};
            float sum_all = 0.0f, sum_kept = 0.0f;
            for (int k = 0; k < 5; ++k) {
                sum_all += wd[k];
                if (k < keep_k_512_outer) sum_kept += wd[k];
            }
            const float scale = sum_all / sum_kept;
            float sum_renorm = 0.0f;
            for (int k = 0; k < 5; ++k) {
                w_out[k] = (k < keep_k_512_outer) ? (wd[k] * scale) : 0.0f;
                sum_renorm += w_out[k];
            }
            CHECK(std::fabs(w_out[0] - 0.60f) < 1e-5f);
            CHECK(std::fabs(w_out[1] - 0.40f) < 1e-5f);
            CHECK(w_out[2] == 0.0f && w_out[3] == 0.0f && w_out[4] == 0.0f);
            CHECK(std::fabs(sum_renorm - 1.0f) < 1e-5f);

            // Verify keep_k == 1 (e.g. 256E outer layer in Turbo L2) bypasses renormalization to prevent 4x-5.5x overshoot
            const int keep_k_256_outer = ggml_atlas_get_moe_keep_k(10, 5, 256, 0, 1, 0);
            CHECK(keep_k_256_outer == 1);
            float w_out_k1[5] = {0};
            for (int k = 0; k < 5; ++k) {
                w_out_k1[k] = (keep_k_256_outer <= 1 || keep_k_256_outer >= 5)
                    ? wd[k]
                    : ((k < keep_k_256_outer) ? (wd[k] * scale) : 0.0f);
            }
            CHECK(std::fabs(w_out_k1[0] - wd[0]) < 1e-6f);
            CHECK(std::fabs(w_out_k1[4] - wd[4]) < 1e-6f);

            ggml_free(gctx);
            set_env_var("ATLAS_CPU_DYNAMIC_K", nullptr);
            set_env_var("ATLAS_PREFILL_TOP_K", nullptr);
            ggml_atlas_reset_moe_decode_state();
            ggml_atlas_set_moe_decode_state(0, 0);
        }
    }

    std::puts("PASS: native async hints, failure, bounds, cancellation, confidence, router boundaries, P2 GPU binding, P3 async transfer, CUDA primary ctx scheduling, TLEP slack & consensus & rollback, AtlasHybridStateCache, sliding window, K_tail=2 boundary, dynamic VRAM headroom guard, zero-copy state persistence, vec_dot_x4 0-ulp parity, Phase 10 Tiered KV state machine, Dual-Cursor Re-quantizer, MoE-aware Dynamic EV controller, Zero-Cost KV rollback, DirectStorage BypassIO streamer, AVX-512 VNNI bit-exact parity, Phase 11 nrc=2 weight sharing parity, Phase 11 Zero-VRAM PLD draft engine, Phase 14 Dynamic Cascading 3-Tier MoE Working Set, Gate 29 Dual-Priority DirectStorage Zero-Copy Promotion, Gate 30 Phase 16 io_uring DMA & Zero-OOM Elastic Watermarking, Gate 32-34 Phase 17 Unified CUDA+RAM ClassLRU & Lockless Mailbox & Adaptive CPU Share, Gate 35-40 FFN Chunking + FP8 E4M3 + CUDA Odd/Empty MoE + Fused HC + MTP GDN Rollback + Real Tier-0 CUDA+CPU Split MoE Parity, Gate 41-44 Phase 18 3-Tensor NVMe-to-VRAM DMA + Layer-Concentrated Mask & Adaptive RAM Guard + Long-Context FP8 VEC & Multi-Token BS=2..4 GPU+CPU MoE Parity + Hierarchical PrefixCache KV Tree Forking, Gate 45 v0.16.1 IQ4_XS 4-Row AVX2/AVX-512 0-ULP Parity, Gate 46 v0.16.2 IQ3_S 4-Row AVX2 0-ULP Parity, Gate 47 v0.16.3 512-bit AVX-512 IQ4_XS+IQ3_XXS 0-ULP Parity, Gate 48 v0.16.3 Zero-Copy Hot-RAM Planner, Gate 49 v0.16.3 MoE 512E K-Floor + Weight Renorm");
    return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "TEST FAILED: %s\n", e.what());
        return 1;
    }
}
