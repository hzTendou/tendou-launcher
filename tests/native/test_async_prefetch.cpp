#define main atlas_program_main
#include "../../src/atlas/atlas-engine.cpp"
#undef main
#include <stdexcept>
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (0)
int main() {
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
    CHECK(!atlas_configure_cuda_primary_ctx("invalid_sched_mode"));

    std::puts("PASS: native async hints, failure, bounds, cancellation, confidence, router boundaries, P2 GPU binding, P3 async transfer, CUDA primary ctx scheduling");
}
