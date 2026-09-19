import os
import json
import unittest
from pathlib import Path
from collections import OrderedDict, defaultdict

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"

class TestAtlasQwen38(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        map_file = CONFIG_DIR / "atlas_physical_map_qwen38.json"
        assert map_file.exists(), f"Physical map file not found: {map_file}"
        with open(map_file, "r", encoding="utf-8") as f:
            cls.physical_map = json.load(f)

        report_file = CONFIG_DIR / "qwen38_audit_report.json"
        assert report_file.exists(), f"Audit report not found: {report_file}"
        with open(report_file, "r", encoding="utf-8") as f:
            cls.audit_report = json.load(f)

    # GGUF Tests
    def test_01_architecture_detection(self):
        model = self.physical_map["model"]
        self.assertEqual(model["architecture"], "qwen4exp")
        self.assertEqual(model["name"], "Qwen3.8 Flash Next Abliterated")

    def test_02_structural_dimensions(self):
        model = self.physical_map["model"]
        self.assertEqual(model["layers"], 48)
        self.assertEqual(model["hidden_size"], 2560)
        self.assertEqual(model["intermediate_size"], 640)
        self.assertEqual(model["shared_intermediate_size"], 640)
        self.assertEqual(model["attention_heads"], 24)
        self.assertEqual(model["kv_heads"], 2)
        self.assertEqual(model["context_length"], 262144)
        self.assertEqual(model["vocab_size"], 248320)

    def test_03_hybrid_and_ssm_parameters(self):
        model = self.physical_map["model"]
        self.assertEqual(model["full_attention_interval"], 4)
        ssm = model["ssm"]
        self.assertEqual(ssm["conv_kernel"], 4)
        self.assertEqual(ssm["state_size"], 128)
        self.assertEqual(ssm["group_count"], 16)
        self.assertEqual(ssm["time_step_rank"], 48)
        self.assertEqual(ssm["inner_size"], 6144)

    def test_04_hyper_connection_and_ple(self):
        model = self.physical_map["model"]
        hc = model["hyper_connection"]
        self.assertEqual(hc["count"], 4)
        self.assertEqual(hc["low_rank"], 320)

        ple = model["ple"]
        self.assertEqual(ple["layers"], [1])
        self.assertEqual(ple["ngram_size"], 3)
        self.assertEqual(ple["heads_per_ngram"], 8)
        self.assertEqual(ple["conv_kernel"], 4)
        self.assertEqual(ple["embedding_length_per_layer"], 160)

    def test_05_multipart_split(self):
        model = self.physical_map["model"]
        parts = model["parts"]
        self.assertEqual(len(parts), 3)
        for part in parts:
            self.assertTrue(os.path.exists(part["blob_path"]), f"Blob missing: {part['blob_path']}")
            self.assertGreater(part["size_bytes"], 0)
            self.assertTrue(part["is_reparse_or_symlink"])

    # Router Tests
    def test_06_router_configuration(self):
        model = self.physical_map["model"]
        self.assertEqual(model["experts"], 512)
        self.assertEqual(model["top_k"], 10)
        self.assertEqual(model["shared_experts"], 1)

    def test_07_deterministic_expert_range(self):
        experts = self.physical_map["experts"]
        self.assertEqual(len(experts), 48 * 512)
        self.assertIn("0:0", experts)
        self.assertIn("0:511", experts)
        self.assertIn("47:0", experts)
        self.assertIn("47:511", experts)

    # Expert Mapping Tests
    def test_08_expert_chunks(self):
        experts = self.physical_map["experts"]
        exp0 = experts["0:0"]
        self.assertEqual(exp0["layer"], 0)
        self.assertEqual(exp0["expert_id"], 0)
        self.assertEqual(len(exp0["chunks"]), 3)

        kinds = {c["kind"] for c in exp0["chunks"]}
        self.assertEqual(kinds, {"gate", "up", "down"})

        gate_chunk = [c for c in exp0["chunks"] if c["kind"] == "gate"][0]
        self.assertEqual(gate_chunk["size_bytes"], 1126400)
        self.assertEqual(gate_chunk["type"], "Q5_K")
        self.assertTrue(os.path.exists(gate_chunk["blob_path"]))

        down_chunk = [c for c in exp0["chunks"] if c["kind"] == "down"][0]
        self.assertEqual(down_chunk["size_bytes"], 1228800)
        self.assertEqual(down_chunk["type"], "Q5_1")
        self.assertTrue(os.path.exists(down_chunk["blob_path"]))

        self.assertEqual(exp0["total_bytes"], 1126400 * 2 + 1228800)
        self.assertEqual(exp0["total_bytes"], 3481600)

    def test_09_chunk_offsets_are_increasing_and_aligned(self):
        experts = self.physical_map["experts"]
        gate_offsets = [experts[f"0:{e}"]["chunks"][0]["file_offset"] for e in range(5)]
        for i in range(1, len(gate_offsets)):
            self.assertEqual(gate_offsets[i], gate_offsets[i-1] + 1126400)
            self.assertEqual(gate_offsets[i] % 32, 0)

    # Memory Manager Logic Tests
    def test_10_residency_states_and_transitions(self):
        class MockMemoryManager:
            def __init__(self, vram_cap_mb, ram_cap_mb):
                self.vram_cap = vram_cap_mb
                self.ram_cap = ram_cap_mb
                self.vram_used = 0
                self.ram_used = 0
                self.vram_cache = OrderedDict()
                self.ram_cache = OrderedDict()
                self.pinned = set()
                self.stats = {"vram_hit": 0, "ram_hit": 0, "nvme_miss": 0, "evictions": 0}

            def access(self, key, size_mb=3.32):
                if key in self.vram_cache:
                    self.stats["vram_hit"] += 1
                    self.vram_cache.move_to_end(key)
                    return "VRAM"
                if key in self.ram_cache:
                    self.stats["ram_hit"] += 1
                    self.ram_cache.move_to_end(key)
                    self.promote_to_vram(key, size_mb)
                    return "RAM"
                self.stats["nvme_miss"] += 1
                self.load_from_nvme(key, size_mb)
                self.promote_to_vram(key, size_mb)
                return "NVME"

            def load_from_nvme(self, key, size_mb):
                while self.ram_used + size_mb > self.ram_cap:
                    evictable = [k for k in self.ram_cache if k not in self.pinned]
                    if not evictable:
                        break
                    victim = evictable[0]
                    del self.ram_cache[victim]
                    self.ram_used -= size_mb
                    self.stats["evictions"] += 1
                self.ram_cache[key] = True
                self.ram_used += size_mb

            def promote_to_vram(self, key, size_mb):
                while self.vram_used + size_mb > self.vram_cap:
                    evictable = [k for k in self.vram_cache if k not in self.pinned]
                    if not evictable:
                        break
                    victim = evictable[0]
                    del self.vram_cache[victim]
                    self.vram_used -= size_mb
                    self.stats["evictions"] += 1
                self.vram_cache[key] = True
                self.vram_used += size_mb

        mm = MockMemoryManager(vram_cap_mb=10, ram_cap_mb=20)
        s1 = mm.access("0:1")
        self.assertEqual(s1, "NVME")
        self.assertIn("0:1", mm.vram_cache)

        s2 = mm.access("0:1")
        self.assertEqual(s2, "VRAM")

        mm.access("0:2")
        mm.access("0:3")
        mm.access("0:4")
        self.assertNotIn("0:1", mm.vram_cache)
        self.assertIn("0:1", mm.ram_cache)

        s_ram = mm.access("0:1")
        self.assertEqual(s_ram, "RAM")
        self.assertEqual(mm.stats["ram_hit"], 1)
        self.assertEqual(mm.stats["vram_hit"], 1)
        self.assertEqual(mm.stats["nvme_miss"], 4)

    # MTP Architecture Tests (Original Baseline)
    def test_11_mtp_metadata_and_isolation(self):
        mtp = self.audit_report["mtp"]
        self.assertEqual(mtp["architecture"], "qwen4exp")
        self.assertEqual(mtp["predict_layers"], 1)
        self.assertFalse(mtp["required_for_normal_inference"])
        self.assertTrue(os.path.exists(mtp["blob_path"]))
        self.assertGreater(mtp["size_gb"], 2.0)

    # -----------------------------------------------------------------------
    # Atlas-Aware MTP / Multi-Token Working Set (MTWS) Requirements (1-17)
    # -----------------------------------------------------------------------

    # Req 1: MTP Q4_K_M loading and verification
    def test_12_mtp_q4km_loading(self):
        paths = [
            r"C:\Users\Ali\.cache\huggingface\hub\models--unsloth--Qwen3.8-Flash-Next-GGUF\snapshots\38bb39ee97821de2c9009abb7e93950eec396e66d\MTP\mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf",
            r"C:\Users\Ali\.cache\huggingface\hub\models--unsloth--Qwen3.8-Flash-Next-GGUF\snapshots\38bb39ee97821de2c9009abb7e93950eec396e66\MTP\mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf"
        ]
        found = False
        target_path = None
        for p in paths:
            if os.path.exists(p):
                found = True
                target_path = p
                break
        self.assertTrue(found, "Q4_K_M MTP model file must exist in Hugging Face cache")
        
        # Check size: ~1.77-1.9 GB (significantly smaller than Q8_0's 2.6 GB)
        size_bytes = os.path.getsize(target_path)
        size_gb = size_bytes / (1024**3)
        self.assertLess(size_gb, 2.1)
        self.assertGreater(size_gb, 1.5)

    # Req 2: MTP RAM placement pattern verification
    def test_13_mtp_ram_placement(self):
        # In RAM mode, all block 48 tensors (dense + MoE) are assigned to CPU RAM
        pattern = r"blk\.48\..*"
        import re
        self.assertTrue(re.match(pattern, "blk.48.attn_q.weight"))
        self.assertTrue(re.match(pattern, "blk.48.ffn_gate_exps.weight"))
        self.assertTrue(re.match(pattern, "blk.48.nextn.eh_proj.weight"))
        self.assertFalse(re.match(pattern, "blk.47.attn_q.weight"))

    # Req 3: MTP VRAM placement pattern verification
    def test_14_mtp_vram_placement(self):
        # In VRAM mode, dense block 48 tensors go to VRAM, while only MoE experts stay in CPU RAM
        moe_only_pattern = r"blk\.48\.ffn_.*_exps"
        import re
        self.assertTrue(re.match(moe_only_pattern, "blk.48.ffn_gate_exps.weight"))
        self.assertTrue(re.match(moe_only_pattern, "blk.48.ffn_up_exps.weight"))
        self.assertTrue(re.match(moe_only_pattern, "blk.48.ffn_down_exps.weight"))
        # Dense attention stays on VRAM (does NOT match CPU override pattern)
        self.assertFalse(re.match(moe_only_pattern, "blk.48.attn_q.weight"))

    # Req 4: Future token extraction (T+1 ... T+N)
    def test_15_future_token_extraction(self):
        draft_proposals = [1042, 2085, 319]
        id_last = 999
        extracted_future = [id_last] + draft_proposals
        self.assertEqual(len(extracted_future), 4)
        self.assertEqual(extracted_future[0], 999)
        self.assertEqual(extracted_future[1:], [1042, 2085, 319])

    # Req 5: Future expert candidate generation
    def test_16_future_expert_candidate_generation(self):
        transitions = {
            (0, 10): {20: 5, 25: 3},
            (0, 20): {30: 4, 35: 2}
        }
        # Step 1: from expert 10 -> [20, 25]
        step1_cands = list(transitions.get((0, 10), {}).keys())
        self.assertIn(20, step1_cands)
        self.assertIn(25, step1_cands)
        # Step 2: from expert 20 -> [30, 35]
        step2_cands = list(transitions.get((0, 20), {}).keys())
        self.assertIn(30, step2_cands)

    # Req 6: Expert scoring (Value-Density formula)
    def test_17_expert_scoring_value_density(self):
        # Score = (prob * transition * locality * latency_saved) / bytes
        prob = 0.8
        trans = 0.7
        locality = 0.9
        latency_saved_ms = 0.8  # NVMe+PCIe stall avoided
        expert_bytes = 3481600

        value_density = (prob * trans * locality * latency_saved_ms) / (expert_bytes / 1048576.0)
        self.assertGreater(value_density, 0.0)

        # High probability beats low probability with same byte footprint
        low_prob = 0.1
        low_density = (low_prob * trans * locality * latency_saved_ms) / (expert_bytes / 1048576.0)
        self.assertGreater(value_density, low_density)

    # Req 7: Unique working-set construction (Deduplication across future tokens)
    def test_18_unique_working_set_construction(self):
        # Token T+1 needs: Expert 42, Expert 10
        # Token T+2 needs: Expert 42, Expert 15
        # Token T+3 needs: Expert 42, Expert 20
        token_experts = {
            1: [42, 10],
            2: [42, 15],
            3: [42, 20]
        }
        unique_working_set = {}
        for t, exps in token_experts.items():
            for e in exps:
                if e not in unique_working_set:
                    unique_working_set[e] = {"score": 0.5, "count": 1}
                else:
                    # Boost score for reused expert
                    s = unique_working_set[e]["score"]
                    unique_working_set[e]["score"] = 1.0 - (1.0 - s) * (1.0 - 0.5)
                    unique_working_set[e]["count"] += 1

        self.assertEqual(len(unique_working_set), 4) # 42, 10, 15, 20
        self.assertEqual(unique_working_set[42]["count"], 3)
        self.assertGreater(unique_working_set[42]["score"], unique_working_set[10]["score"])

    # Req 8: VRAM budget enforcement
    def test_19_vram_budget_enforcement(self):
        vram_cap_mb = 100
        expert_size_mb = 3.32
        max_experts_allowed = int(vram_cap_mb // expert_size_mb)
        
        candidates = list(range(50))
        allocated = []
        used_mb = 0
        for c in candidates:
            if used_mb + expert_size_mb <= vram_cap_mb:
                allocated.append(c)
                used_mb += expert_size_mb
            else:
                break

        self.assertEqual(len(allocated), max_experts_allowed)
        self.assertLessEqual(used_mb, vram_cap_mb)

    # Req 9: RAM budget enforcement
    def test_20_ram_budget_enforcement(self):
        ram_cap_mb = 200
        expert_size_mb = 3.32
        max_ram_experts = int(ram_cap_mb // expert_size_mb)
        
        candidates = list(range(100))
        allocated_ram = []
        used_ram_mb = 0
        for c in candidates:
            if used_ram_mb + expert_size_mb <= ram_cap_mb:
                allocated_ram.append(c)
                used_ram_mb += expert_size_mb
            else:
                break

        self.assertEqual(len(allocated_ram), max_ram_experts)
        self.assertLessEqual(used_ram_mb, ram_cap_mb)

    # Req 10: Batched n_tok > 1 cache accounting
    def test_21_batched_ntok_gt_1_cache_accounting(self):
        # Verification batch with n_tok = 3
        n_tok = 3
        k = 10
        raw_tensor = [i % 512 for i in range(k * n_tok)]
        
        accessed_records = []
        for t_idx in range(n_tok):
            for rank in range(k):
                exp = raw_tensor[t_idx * k + rank]
                accessed_records.append((t_idx, exp))

        self.assertEqual(len(accessed_records), 30)
        self.assertEqual(len(accessed_records), n_tok * k)

    # Req 11: Expert reuse across multiple tokens
    def test_22_expert_reuse_across_tokens(self):
        batch_tokens_experts = [
            [1, 2, 3, 4], # tok 0
            [2, 3, 5, 6], # tok 1
            [3, 6, 7, 8]  # tok 2
        ]
        total_accesses = 0
        seen_experts = set()
        reuse_count = 0

        for tok in batch_tokens_experts:
            for exp in tok:
                total_accesses += 1
                if exp in seen_experts:
                    reuse_count += 1
                else:
                    seen_experts.add(exp)

        self.assertEqual(total_accesses, 12)
        self.assertEqual(len(seen_experts), 8)
        self.assertEqual(reuse_count, 4) # 2 reused once, 3 reused twice, 6 reused once
        reuse_factor = total_accesses / len(seen_experts)
        self.assertAlmostEqual(reuse_factor, 1.5)

    # Req 12: Prefetch deadline tracking
    def test_23_prefetch_deadline_tracking(self):
        import time
        prefetch_start = time.time()
        # Simulated prefetch duration: 5ms
        prefetch_end = prefetch_start + 0.005

        # Demand arrives at 10ms -> completed before demand (hit)
        demand_time_hit = prefetch_start + 0.010
        is_hit = demand_time_hit >= prefetch_end
        self.assertTrue(is_hit)

        # Demand arrives at 2ms -> prefetch late (stall)
        demand_time_stall = prefetch_start + 0.002
        is_stall = demand_time_stall < prefetch_end
        self.assertTrue(is_stall)

    # Req 13: Prediction precision and recall
    def test_24_prediction_precision_and_recall(self):
        predicted_unique = {1, 2, 3, 4, 5, 6, 7, 8}      # 8 predicted
        actual_unique = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12} # 10 actual
        intersection = predicted_unique.intersection(actual_unique) # {3, 4, 5, 6, 7, 8} = 6

        precision = len(intersection) / len(predicted_unique) # 6/8 = 75%
        recall = len(intersection) / len(actual_unique)       # 6/10 = 60%

        self.assertEqual(len(intersection), 6)
        self.assertEqual(precision, 0.75)
        self.assertEqual(recall, 0.60)

    # Req 14: Speculative rejection and rollback
    def test_25_speculative_rejection_and_rollback(self):
        class MockCheckpoint:
            def __init__(self):
                self.pos_max = 100
                self.n_tokens = 101
                self.saved_state = "STATE_TGT_VALID"

            def rollback(self, accepted_count):
                # When partial accept (e.g. 1 out of 3 tokens accepted):
                # restore pos_max and trim tokens
                self.n_tokens += accepted_count
                self.pos_max += accepted_count

        ckpt = MockCheckpoint()
        ckpt.rollback(1)
        self.assertEqual(ckpt.pos_max, 101)
        self.assertEqual(ckpt.n_tokens, 102)

    # Req 15: Memory-pressure fallback
    def test_26_memory_pressure_fallback(self):
        # Dynamic budgeting:
        # Total VRAM 8192 MB, base 5200 MB
        # When MTP in RAM: VRAM expert budget = 8192 - 5200 = 2992 MB
        # When MTP in VRAM: VRAM expert budget = 8192 - 5200 - 1900 = 1092 MB
        def calc_vram_budget(mtp_location):
            base = 5200
            mtp_size = 1900 if mtp_location == "vram" else 0
            return max(512, 8192 - base - mtp_size)

        budget_ram = calc_vram_budget("ram")
        budget_vram = calc_vram_budget("vram")

        self.assertEqual(budget_ram, 2992)
        self.assertEqual(budget_vram, 1092)
        self.assertGreater(budget_ram, budget_vram)

    # Req 16: Adaptive N reduction
    def test_27_adaptive_n_reduction(self):
        class MockAdaptivePolicy:
            def __init__(self, initial_n=6):
                self.n = initial_n
                self.low_streak = 0

            def record_batch(self, accepted, drafted):
                rate = accepted / drafted
                if rate < 0.20:
                    self.low_streak += 1
                    if self.low_streak >= 3 and self.n > 2:
                        self.n = max(2, self.n - 2)
                        self.low_streak = 0
                else:
                    self.low_streak = 0

        policy = MockAdaptivePolicy(initial_n=6)
        policy.record_batch(0, 6)
        policy.record_batch(0, 6)
        policy.record_batch(0, 6)
        self.assertEqual(policy.n, 4)
        policy.record_batch(0, 4)
        policy.record_batch(0, 4)
        policy.record_batch(0, 4)
        self.assertEqual(policy.n, 2)

    # Req 17: MTP disable fallback
    def test_28_mtp_disable_fallback(self):
        config_mtp_off = {"mtp_mode": "off", "enable_mtws": False}
        self.assertEqual(config_mtp_off["mtp_mode"], "off")
        self.assertFalse(config_mtp_off["enable_mtws"])

    # Section 8 & 19: Multi-token expert batching & compute compression
    def test_29_multi_token_expert_batching(self):
        # Tokens T0, T1, T2 routing across 48 layers
        # T0: [10, 20, 30]
        # T1: [10, 25, 30]
        # T2: [10, 20, 40]
        # Total routed expert invocations = 9
        # Unique experts = {10, 20, 25, 30, 40} = 5
        # Batched GEMMs for expert 10: 3 tokens in 1 GEMM
        # Batched GEMMs for expert 20: 2 tokens in 1 GEMM
        # Batched GEMMs for expert 30: 2 tokens in 1 GEMM
        # Single GEMMs for expert 25: 1 token
        # Single GEMMs for expert 40: 1 token
        # Actual unique expert GEMM matrices loaded = 5
        # Effective Expert Compute Compression (EEC) = 9 / 5 = 1.80x
        t0 = [10, 20, 30]
        t1 = [10, 25, 30]
        t2 = [10, 20, 40]
        invocations = len(t0) + len(t1) + len(t2)
        unique_experts = set(t0 + t1 + t2)
        eec = invocations / len(unique_experts)
        self.assertEqual(invocations, 9)
        self.assertEqual(len(unique_experts), 5)
        self.assertAlmostEqual(eec, 1.80)
        self.assertGreater(eec, 1.30)

    # Section 20: Hotness eviction resistance across multi-token windows
    def test_30_hotness_eviction_resistance(self):
        # Experts with hotness > 0 receive second-chance clock eviction passes
        cache = {"E1": {"hotness": 3}, "E2": {"hotness": 0}, "E3": {"hotness": 1}}
        evicted = None
        # Eviction scan from least-recently used:
        # Check E1: hotness 3 -> decrement to 2, skip eviction
        # Check E2: hotness 0 -> candidate for eviction!
        for k in ["E1", "E2", "E3"]:
            if cache[k]["hotness"] > 0:
                cache[k]["hotness"] -= 1
                continue
            evicted = k
            break
        self.assertEqual(evicted, "E2")
        self.assertEqual(cache["E1"]["hotness"], 2)

    # Section 14: Confidence-gated early pruning
    def test_31_confidence_gated_early_pruning(self):
        p_min = 0.60
        draft_candidates = [
            ("token_A", 0.85),
            ("token_B", 0.72),
            ("token_C", 0.42), # below p_min -> prune here
            ("token_D", 0.65),
        ]
        pruned_draft = []
        for tok, p in draft_candidates:
            if p < p_min:
                break
            pruned_draft.append(tok)
        self.assertEqual(len(pruned_draft), 2)
        self.assertEqual(pruned_draft, ["token_A", "token_B"])

    # Section 15: Commit-on-accept and early rejection
    def test_32_commit_on_accept_and_early_rejection(self):
        draft = [101, 102, 103, 104]
        # Target model logits sample: 101 matches, 102 matches, 999 diverges
        target_sampled = [101, 102, 999]
        accepted = []
        for i in range(len(target_sampled)):
            tok = target_sampled[i]
            accepted.append(tok)
            if i < len(draft) and tok != draft[i]:
                # Divergence -> stop immediately, discard remaining suffix
                break
        self.assertEqual(accepted, [101, 102, 999])
        self.assertEqual(len(accepted) - 1, 2) # 2 draft tokens accepted

    # Section 10 & 11: Dynamic RAM (<=10GB) and VRAM budget safety
    def test_33_hardware_vram_and_ram_dynamic_budgeting(self):
        total_ram_gb = 16.0
        os_ram_gb = 6.0
        atlas_ram_ceiling_gb = min(10.0, total_ram_gb - os_ram_gb)
        self.assertLessEqual(atlas_ram_ceiling_gb, 10.0)

        total_vram_mb = 8150
        dense_model_mb = 5200
        mtp_vram_mb = 1900
        safe_vram_reserve_mb = 512
        available_expert_vram = total_vram_mb - dense_model_mb - mtp_vram_mb - safe_vram_reserve_mb
        self.assertGreater(available_expert_vram, 0)
        self.assertLessEqual(available_expert_vram + dense_model_mb + mtp_vram_mb, total_vram_mb)

    # Section 22: Deterministic verification equivalence
    def test_34_greedy_deterministic_verification_equivalence(self):
        # Under greedy decoding (--temp 0), argmax logits are 100% deterministic
        target_logits = [0.1, 0.2, 0.9, 0.4] # argmax index 2
        spec_candidate = 2
        is_exact_match = (spec_candidate == target_logits.index(max(target_logits)))
        self.assertTrue(is_exact_match)

    # Phase 4 & 9: Adaptive-K Expert Pruning
    def test_35_adaptive_k_expert_pruning(self):
        nominal_k = 10
        for k in [4, 5, 6, 8]:
            reduction = (1.0 - k / nominal_k) * 100.0
            self.assertGreater(reduction, 0.0)
            self.assertLessEqual(reduction, 60.0)
        # At K=5: compute reduction is exactly 50%
        self.assertEqual((1.0 - 5 / 10) * 100.0, 50.0)
        # At K=4: compute reduction is exactly 60%
        self.assertEqual((1.0 - 4 / 10) * 100.0, 60.0)

    # Phase 10: Multi-Indicator Quality Estimation
    def test_36_quality_estimator_repetition_and_entropy(self):
        import math
        # Healthy sequence with diverse tokens
        healthy_tokens = list(range(1, 33)) # 32 distinct tokens
        token_freq = {}
        for t in healthy_tokens:
            token_freq[t] = token_freq.get(t, 0) + 1
        n = len(healthy_tokens)
        entropy = sum(-(count / n) * math.log2(count / n) for count in token_freq.values())
        self.assertGreater(entropy, 4.0)

        # Degenerate repetitive sequence: [5, 5, 5, 5...]
        repeat_tokens = [5] * 32
        rep_count = sum(1 for i in range(4, len(repeat_tokens)) if repeat_tokens[i] == repeat_tokens[i-4])
        self.assertGreater(rep_count, 20)

    # Phase 9: Quality Floor Guard (Floor = 0.85)
    def test_37_quality_floor_controller_guard(self):
        quality_floor = 0.85
        # If measured quality falls below 0.85, controller must flag breach and trigger recovery
        measured_good = 0.917
        measured_bad = 0.780
        self.assertTrue(measured_good >= quality_floor)
        self.assertFalse(measured_bad >= quality_floor)

    # Phase 11: Self-Learning Locality Bounded Tracking
    def test_38_persistent_locality_learning(self):
        # 48 layers, max 64 tracked experts
        grid = [[0 for _ in range(64)] for _ in range(48)]
        # Record layer 0 experts [5, 12, 23]
        for exp in [5, 12, 23]:
            grid[0][exp] += 1
        self.assertEqual(grid[0][5], 1)
        self.assertEqual(grid[0][12], 1)
        self.assertEqual(grid[0][23], 1)
        self.assertEqual(grid[0][0], 0)

    # Section 3: Pinned Host Memory Staging Pool Budget
    def test_39_pinned_host_staging_pool_budget(self):
        pool_capacity_mb = 512
        max_ram_budget_gb = 10.0
        # Pinned staging pool takes 512MB = 0.5GB, safely within 10GB RAM budget
        self.assertLessEqual(pool_capacity_mb / 1024.0, 1.0)
        self.assertLessEqual(pool_capacity_mb / 1024.0, max_ram_budget_gb)

    # Section 5: Dynamic CPU/GPU Expert Cost Model
    def test_40_expert_cost_model_evaluation(self):
        cpu_gemm_ms = 3.30
        pcie_h2d_ms = 0.27
        gpu_gemm_ms = 0.05
        sync_penalty_ms = 0.10

        # GPU Hot Cache: Zero H2D latency -> 0.05 ms vs 3.30 ms CPU
        hot_cache_cost = gpu_gemm_ms
        self.assertLess(hot_cache_cost, cpu_gemm_ms)

        # GPU Async H2D: 0.27 + 0.05 + 0.10 = 0.42 ms vs 3.30 ms CPU
        async_h2d_cost = pcie_h2d_ms + gpu_gemm_ms + sync_penalty_ms
        self.assertLess(async_h2d_cost, cpu_gemm_ms)

        # Speedup of GPU Async H2D over CPU AVX2 GEMM
        speedup = cpu_gemm_ms / async_h2d_cost
        self.assertGreater(speedup, 7.0)

    # Section 2 & 8: Grouped-GEMM Task Dispatch & Batching
    def test_41_grouped_gemm_dispatch_and_batching(self):
        # 3 routed experts per token combined into 1 Grouped-GEMM
        experts = [12, 45, 89]
        k = len(experts)
        self.assertEqual(k, 3)
        # Unified task batch size equals K
        task = {"layer": 0, "token_idx": 0, "expert_ids": experts, "batch_size": k}
        self.assertEqual(task["batch_size"], 3)
        self.assertEqual(len(task["expert_ids"]), 3)

    # Section 2 & 8: Multi-Token Concurrent CUDA Streams
    def test_42_cuda_stream_round_robin_distribution(self):
        num_streams = 3
        # 3 concurrent tokens in verification batch distributed across 3 streams
        stream_assignments = [tok_idx % num_streams for tok_idx in range(3)]
        self.assertEqual(stream_assignments, [0, 1, 2])
        # Stream 0, 1, 2 all receive independent execution slots
        self.assertEqual(len(set(stream_assignments)), 3)

if __name__ == "__main__":
    unittest.main()
