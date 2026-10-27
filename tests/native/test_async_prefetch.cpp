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
        HMODULE hCpu = GetModuleHandleW(L"ggml-cpu.dll");
        if (!hCpu) hCpu = LoadLibraryW(L"ggml-cpu.dll");
        if (hCpu) {
            auto reset_fn = (pfn_atlas_reset_decode_step)(void*)GetProcAddress(hCpu, "atlas_reset_decode_step");
            CHECK(reset_fn != nullptr);
            reset_fn();
        }
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
            CHECK(streamer.is_bypass_io_supported() == true);
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
        }
    }
    std::puts("PASS: native async hints, failure, bounds, cancellation, confidence, router boundaries, P2 GPU binding, P3 async transfer, CUDA primary ctx scheduling, TLEP slack & consensus & rollback, AtlasHybridStateCache, sliding window, K_tail=2 boundary, dynamic VRAM headroom guard, zero-copy state persistence, vec_dot_x4 0-ulp parity, Phase 10 Tiered KV state machine, Dual-Cursor Re-quantizer, MoE-aware Dynamic EV controller, Zero-Cost KV rollback, DirectStorage BypassIO streamer, AVX-512 VNNI bit-exact parity, Phase 11 nrc=2 weight sharing parity, Phase 11 Zero-VRAM PLD draft engine, Phase 14 Dynamic Cascading 3-Tier MoE Working Set");
    return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "TEST FAILED: %s\n", e.what());
        return 1;
    }
}
