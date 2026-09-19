"""Atlas tiered-memory simulator.

This is intentionally a trace-driven *hypothesis simulator*, not a GPU runtime.
It models the policy question Atlas cares about:

    NVMe -> RAM -> VRAM -> compute

with request-local VRAM caches, a global RAM cache, causal/oracle prefetch,
optional prompt priors, optional affinity-region expansion, and overlap between
prefetch I/O and the compute window of the current token.

The simulator never changes the expert selected by the trace. Prefetch can only
move data earlier; an actual router miss remains an actual miss.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from atlas_predictor import AtlasPredictor


def percentile(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    if lo == hi:
        return float(s[lo])
    return s[lo] * (hi - k) + s[hi] * (k - lo)


def load_sessions(path):
    sessions = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            records = obj.get("records", [])
            # Explicit phase traces: decode only. Legacy traces: all records.
            decode = [r for r in records if r.get("phase") == "decode"]
            unknown = [r for r in records if r.get("phase") not in {"prompt", "decode"}]
            if decode:
                records = decode
            elif unknown:
                records = unknown
            else:
                records = []
            sessions.append({"prompt_idx": obj.get("prompt_idx", len(sessions)),
                             "prompt": obj.get("prompt", ""),
                             "records": records})
    return sessions


def layers(rec):
    return {str(k): set(v) for k, v in (rec.get("experts_by_layer") or {}).items()}


def all_keys(rec):
    return {(str(layer), int(e)) for layer, es in layers(rec).items() for e in es}


def prompt_profile(session_obj):
    records = session_obj.get("_all_records", [])
    prompt = [r for r in records if r.get("phase") == "prompt"]
    prof = defaultdict(Counter)
    for rec in prompt:
        for layer, es in layers(rec).items():
            prof[layer].update(es)
    return prof


def build_affinity_regions(sessions, max_region_size=8, min_count=2):
    counts = Counter()
    for s in sessions:
        for rec in s["records"]:
            for layer, es in layers(rec).items():
                xs = sorted(es)
                for i, a in enumerate(xs):
                    for b in xs[i + 1:]:
                        counts[(layer, a, b)] += 1
    by_layer = defaultdict(list)
    for (layer, a, b), c in counts.items():
        if c >= min_count:
            by_layer[layer].append((c, a, b))

    expert_region = {}
    region_members = {}
    rid = 0
    # Deterministic greedy regions; singleton experts are added later.
    for layer in sorted(by_layer, key=lambda x: int(x)):
        adj = defaultdict(set)
        for _, a, b in sorted(by_layer[layer], reverse=True):
            adj[a].add(b); adj[b].add(a)
        unassigned = set(adj)
        while unassigned:
            seed = max(unassigned, key=lambda e: (len(adj[e] & unassigned), -int(e)))
            region = [seed]
            unassigned.remove(seed)
            while len(region) < max_region_size:
                cand = [e for e in unassigned if adj[e] & set(region)]
                if not cand:
                    break
                e = max(cand, key=lambda x: (len(adj[x] & set(region)), -int(x)))
                region.append(e); unassigned.remove(e)
            key = rid; rid += 1
            members = {(layer, int(e)) for e in region}
            region_members[key] = members
            for e in region:
                expert_region[(layer, int(e))] = key

    # Include experts that never appeared in an affinity edge.
    all_experts = set()
    for s in sessions:
        for rec in s["records"]:
            all_experts |= all_keys(rec)
    for key in sorted(all_experts):
        if key not in expert_region:
            expert_region[key] = rid
            region_members[rid] = {key}
            rid += 1
    return expert_region, region_members


@dataclass
class CacheItem:
    size_mb: float
    ready_ms: float


class GlobalLRU:
    def __init__(self, capacity_mb):
        self.capacity_mb = max(0.0, capacity_mb)
        self.used_mb = 0.0
        self.data = OrderedDict()

    def has(self, key):
        return key in self.data

    def touch(self, key):
        if key in self.data:
            item = self.data.pop(key)
            self.data[key] = item
            return item
        return None

    def put(self, key, item):
        if key in self.data:
            old = self.data.pop(key)
            self.used_mb -= old.size_mb
        self.data[key] = item
        self.used_mb += item.size_mb
        while self.data and self.used_mb > self.capacity_mb:
            _, old = self.data.popitem(last=False)
            self.used_mb -= old.size_mb


class FrequencyBiasedVRAM:
    """VRAM cache with frequency-biased eviction.

    Experts are evicted in order of ascending access frequency rather than
    pure LRU recency.  This prevents high-frequency experts (those that
    re-appear every 1-3 tokens) from being displaced by one-off accesses.

    The frequency counter uses a session-local window so that stale
    cross-request frequencies do not poison the current session.
    """

    def __init__(self, capacity_mb: float):
        self.capacity_mb = max(0.0, float(capacity_mb))
        self.used_mb = 0.0
        self.data: dict = {}           # key -> CacheItem
        self._freq: Counter = Counter()  # key -> session hit count

    def has(self, key) -> bool:
        return key in self.data

    def get(self, key):
        item = self.data.get(key)
        if item is not None:
            self._freq[key] += 1
        return item

    def put(self, key, item: CacheItem) -> bool:
        if item.size_mb > self.capacity_mb:
            return False
        if key in self.data:
            self.used_mb -= self.data[key].size_mb
        self.data[key] = item
        self.used_mb += item.size_mb
        while self.data and self.used_mb > self.capacity_mb:
            # Evict lowest-frequency expert first (cold-first policy).
            victim = min(self.data, key=lambda k: self._freq.get(k, 0))
            self.used_mb -= self.data.pop(victim).size_mb
            self._freq.pop(victim, None)
        return key in self.data

    def move_to_end(self, key):
        # No-op for frequency-based cache; frequency is the ranking key.
        pass

    def clear(self):
        self.data.clear()
        self._freq.clear()
        self.used_mb = 0.0

    def reset_session(self):
        """Reset per-session frequency counters without evicting items."""
        self._freq.clear()


def collect_all_expert_keys(sessions) -> set:
    """Return the set of all (layer, expert) keys seen in the trace."""
    keys = set()
    for s in sessions:
        for rec in s['records']:
            keys |= all_keys(rec)
    return keys


class TieredSim:
    """Discrete-event approximation of NVMe -> RAM -> PCIe -> VRAM.

    The important correction versus the old simulator is that prefetch work is
    scheduled during the *current token's compute window*.  It therefore can be
    hidden by compute instead of being serialized before the token is allowed to
    run.  VRAM is a global byte budget, not an artificial per-layer budget.
    """
    def __init__(self, expert_mb, vram_capacity_mb, ram_gb, nvme_gbps,
                 pcie_gbps, ram_gbps, nvme_latency_ms, pcie_latency_ms,
                 compute_ms, vram_policy='lru'):
        self.expert_mb = float(expert_mb)
        self.vram_capacity_mb = max(0.0, float(vram_capacity_mb))
        self.ram = GlobalLRU(ram_gb * 1024.0)
        self.nvme_gbps = float(nvme_gbps)
        self.pcie_gbps = float(pcie_gbps)
        self.ram_gbps = float(ram_gbps)
        self.nvme_latency_ms = float(nvme_latency_ms)
        self.pcie_latency_ms = float(pcie_latency_ms)
        self.compute_ms = float(compute_ms)
        self.vram_policy = str(vram_policy)
        if self.vram_policy == 'freq':
            self.vram = FrequencyBiasedVRAM(self.vram_capacity_mb)
        else:
            self.vram = OrderedDict()  # key -> CacheItem; global across layers
        self.vram_used_mb = 0.0
        self.nvme_busy_until = 0.0
        self.pcie_busy_until = 0.0
        self.ram_busy_until = 0.0
        self.now_ms = 0.0
        self.stats = Counter()
        self.token_exposed_ms = []

    def preload_experts_to_ram(self, all_expert_keys):
        """Preload the entire expert corpus into RAM from NVMe.
        
        Calculates sequential NVMe read duration + host RAM write duration.
        """
        total_mb = len(all_expert_keys) * self.expert_mb
        if total_mb > self.ram.capacity_mb:
            return False, 0.0
        size_gb = total_mb / 1024.0
        nvme_dur = self.nvme_latency_ms + (size_gb / self.nvme_gbps * 1000.0 if self.nvme_gbps > 0 else float('inf'))
        nvme_done = self.now_ms + nvme_dur
        ram_dur = (size_gb / self.ram_gbps * 1000.0 if self.ram_gbps > 0 else float('inf'))
        ram_done = nvme_done + ram_dur
        for k in sorted(all_expert_keys):
            self.ram.put(k, CacheItem(self.expert_mb, ram_done))
        self.stats['ram_preload_mb'] = total_mb
        self.stats['ram_preload_ms'] = ram_done - self.now_ms
        self.now_ms = ram_done
        return True, ram_done

    def _vram_item(self, key):
        if isinstance(self.vram, FrequencyBiasedVRAM):
            return self.vram.get(key)
        item = self.vram.get(key)
        if item is not None:
            self.vram.move_to_end(key)
        return item

    def _vram_has(self, key, now_ms):
        item = self._vram_item(key)
        return item is not None and item.ready_ms <= now_ms

    def _vram_pending(self, key):
        item = self._vram_item(key)
        return item if item is not None else None

    def _vram_put(self, key, ready_ms=0.0):
        if self.expert_mb > self.vram_capacity_mb:
            self.stats['vram_capacity_miss'] += 1
            return False
        if isinstance(self.vram, FrequencyBiasedVRAM):
            ok = self.vram.put(key, CacheItem(self.expert_mb, ready_ms))
            if not ok:
                self.stats['vram_capacity_miss'] += 1
            return ok
        old = self.vram.pop(key, None)
        if old is not None:
            self.vram_used_mb -= old.size_mb
        self.vram[key] = CacheItem(self.expert_mb, ready_ms)
        self.vram_used_mb += self.expert_mb
        while self.vram and self.vram_used_mb > self.vram_capacity_mb:
            _, old = self.vram.popitem(last=False)
            self.vram_used_mb -= old.size_mb
            self.stats['vram_evictions'] += 1
        return key in self.vram

    def _transfer_ram_to_vram(self, key, start_ms):
        # PCIe, not DRAM bandwidth, is the host->GPU bottleneck for this leg.
        size_gb = self.expert_mb / 1024.0
        resource = max(start_ms, self.pcie_busy_until)
        duration = self.pcie_latency_ms + (size_gb / self.pcie_gbps * 1000.0
                                           if self.pcie_gbps > 0 else float('inf'))
        done = resource + duration
        self.pcie_busy_until = done
        self.stats['ram_to_vram_mb'] += self.expert_mb
        self.stats['pcie_ms'] += duration
        self._vram_put(key, done)
        return done

    def _transfer_nvme_to_ram_to_vram(self, key, start_ms):
        size_gb = self.expert_mb / 1024.0
        nvme_resource = max(start_ms, self.nvme_busy_until)
        nvme_duration = self.nvme_latency_ms + (size_gb / self.nvme_gbps * 1000.0
                                                 if self.nvme_gbps > 0 else float('inf'))
        nvme_done = nvme_resource + nvme_duration
        self.nvme_busy_until = nvme_done
        self.stats['nvme_mb'] += self.expert_mb
        self.stats['nvme_ms'] += nvme_duration

        # Model the host RAM write as a separate serialized resource.  It is
        # intentionally conservative; users can set --ram-gbps to measured copy BW.
        ram_resource = max(nvme_done, self.ram_busy_until)
        ram_duration = (size_gb / self.ram_gbps * 1000.0
                        if self.ram_gbps > 0 else float('inf'))
        ram_done = ram_resource + ram_duration
        self.ram_busy_until = ram_done
        self.stats['ram_write_mb'] += self.expert_mb

        # RAM is warm only after the file has reached host memory.
        self.ram.put(key, CacheItem(self.expert_mb, ram_done))
        return self._transfer_ram_to_vram(key, ram_done)

    def ensure(self, key, request_ms):
        item = self._vram_pending(key)
        if item is not None:
            if item.ready_ms <= request_ms:
                self.stats['vram_hit'] += 1
                return request_ms, 'vram'
            # A prefetch is already in flight. Wait only for its completion.
            self.stats['prefetch_wait'] += 1
            return item.ready_ms, 'prefetch_wait'

        item = self.ram.touch(key)
        if item is not None:
            self.stats['ram_hit'] += 1
            done = self._transfer_ram_to_vram(key, max(request_ms, item.ready_ms))
            return done, 'ram'

        self.stats['nvme_miss'] += 1
        done = self._transfer_nvme_to_ram_to_vram(key, request_ms)
        return done, 'nvme'

    def prefetch(self, keys, start_ms):
        seen = set()
        for key in sorted(keys):
            if key in seen:
                continue
            seen.add(key)
            item = self._vram_pending(key)
            if item is not None:
                self.stats['prefetch_already_hot'] += 1
                continue
            ram_item = self.ram.touch(key)
            if ram_item is not None:
                self.stats['prefetch_from_ram'] += 1
                self._transfer_ram_to_vram(key, max(start_ms, ram_item.ready_ms))
            else:
                self.stats['prefetch_from_nvme'] += 1
                self._transfer_nvme_to_ram_to_vram(key, start_ms)
            self.stats['prefetch_requested'] += 1

    def run_token(self, actual_keys, prefetch_keys=None):
        token_start = self.now_ms
        max_ready = token_start
        sources = Counter()

        # The actual token must first have all routed experts available.
        for key in sorted(actual_keys):
            done, source = self.ensure(key, token_start)
            max_ready = max(max_ready, done)
            sources[source] += 1

        exposed = max(0.0, max_ready - token_start)
        compute_start = max_ready
        compute_end = compute_start + self.compute_ms

        # Prefetch is overlapped with compute, not serialized in front of it.
        if prefetch_keys:
            self.prefetch(prefetch_keys, compute_start)

        self.stats['tokens'] += 1
        self.stats['actual_experts'] += len(actual_keys)
        self.stats['exposed_io_ms'] += exposed
        self.token_exposed_ms.append(exposed)
        self.now_ms = compute_end
        return exposed, sources

def _keys_to_layers(keys):
    """Convert a set of (layer, expert) keys back to a layers dict.

    Used by Experiment F to synthesise the hypothetical 'current state' for
    the second-step prediction from the first-step candidate set.
    """
    d = {}
    for layer, expert in keys:
        d.setdefault(str(layer), set()).add(int(expert))
    return d


def causal_predictor_state():
    return defaultdict(lambda: defaultdict(Counter))


def predict_from_transition(transitions, prev_layers, top_k_by_layer):
    """Predict the next routing set without reading any future expert IDs.

    The caller supplies only the observed top-k cardinality per layer.  This is
    important: passing the next record merely to learn its k would make the
    predictor subtly depend on future data.
    """
    out = set()
    for layer, k in top_k_by_layer.items():
        scores = Counter()
        for pe in prev_layers.get(layer, set()):
            scores.update(transitions[layer][pe])
        out |= {(layer, int(e)) for e, _ in scores.most_common(k)}
    return out


def top_k_by_layer(layers_obj):
    return {layer: len(es) for layer, es in layers_obj.items()}


def prompt_hybrid_predict(transitions, prompt_profile_by_layer, prev_layers, top_k_by_layer, alpha):
    out = set()
    for layer, k in top_k_by_layer.items():
        trans = Counter()
        for pe in prev_layers.get(layer, set()):
            trans.update(transitions[layer][pe])
        # Normalize each signal independently so frequency scale cannot dominate.
        score = Counter()
        if trans:
            mx = max(trans.values())
            for e, v in trans.items():
                score[e] += alpha * v / mx
        prof = prompt_profile_by_layer.get(layer, Counter())
        if prof:
            mx = max(prof.values())
            for e, v in prof.items():
                score[e] += (1.0 - alpha) * v / mx
        out |= {(layer, int(e)) for e, _ in score.most_common(k)}
    return out


def evaluate_policy(sessions, mode, vram_capacity_mb, expert_mb, ram_gb, nvme_gbps,
                    pcie_gbps, ram_gbps, nvme_latency_ms, pcie_latency_ms,
                    compute_ms, prefetch_horizon, hybrid_alpha, region_map=None, region_members=None,
                    region_expand=False, predictor_kwargs=None, preload_ram=False,
                    vram_policy='lru', deadline_aware=False, step2_budget_fraction=0.5):
    sim = TieredSim(expert_mb, vram_capacity_mb, ram_gb, nvme_gbps,
                    pcie_gbps, ram_gbps, nvme_latency_ms, pcie_latency_ms, compute_ms,
                    vram_policy=vram_policy)
    
    if preload_ram:
        all_keys_corpus = collect_all_expert_keys(sessions)
        sim.preload_experts_to_ram(all_keys_corpus)

    exposed = []
    wasted = correct = predicted = actual_total = 0
    step1_count = step2_count = step2_useful = step2_wasted_approx = 0
    # Persistent predictor state is updated only after a complete session.
    # This permits realistic cross-request learning without leaking the target
    # session's future into its own predictions.
    atlas_pred = AtlasPredictor(**(predictor_kwargs or {}))

    # Calculate deadline bandwidth cap: max transferable experts over PCIe during compute_ms
    max_pcie_mb = max(0.0, (compute_ms - pcie_latency_ms) / 1000.0 * pcie_gbps * 1024.0) if pcie_gbps > 0 else float('inf')
    max_deadline_candidates = max(1, int(max_pcie_mb / expert_mb)) if expert_mb > 0 else 100000

    for s in sessions:
        # VRAM is request-local by design; RAM remains warm across requests.
        if isinstance(sim.vram, FrequencyBiasedVRAM):
            sim.vram.clear()
        else:
            sim.vram.clear()
            sim.vram_used_mb = 0.0
        records = s['records']
        transitions = causal_predictor_state()
        pred_state = atlas_pred.begin_session()
        prev = None
        prompt_prof = prompt_profile(s)

        for idx, rec in enumerate(records):
            current = layers(rec)
            actual = all_keys(rec)
            pred = set()
            step2_pred = set()
            if idx + 1 < len(records):
                target_layers = layers(records[idx + 1])
                if mode == 'oracle':
                    pred = all_keys(records[idx + 1])
                elif mode == 'atlas_predictor' and idx > 0:
                    vram_keys = set(sim.vram.data.keys()) if isinstance(sim.vram, FrequencyBiasedVRAM) else set(sim.vram.keys())
                    phase_str = str(rec.get('phase', 'decode'))
                    pp = pred_state.predict(current, prompt_prof, phase=phase_str, exclude_keys=vram_keys)
                    raw_pred = pp.keys

                    # --- Deadline cap for step-1 ---
                    step1_cap = max_deadline_candidates if deadline_aware else len(raw_pred) + 1
                    if len(raw_pred) > step1_cap:
                        if pp.ranked_keys_by_layer:
                            b_per_l = max(1, step1_cap // max(1, len(pp.ranked_keys_by_layer)))
                            capped = set()
                            for _layer, rkeys in pp.ranked_keys_by_layer.items():
                                capped.update(rkeys[:b_per_l])
                            step1_pred = capped
                        else:
                            step1_pred = set(sorted(raw_pred)[:step1_cap])
                    else:
                        step1_pred = raw_pred

                    # --- Experiment F: 2-step-ahead prediction ---
                    step2_pred = set()
                    if prefetch_horizon >= 2 and step1_pred:
                        hypo_layers = _keys_to_layers(step1_pred)
                        pp2 = pred_state.predict(
                            hypo_layers,
                            phase=phase_str,
                            exclude_keys=vram_keys | step1_pred,
                        )
                        # Budget: fraction of remaining capacity after step-1
                        remaining_budget = max(0, max_deadline_candidates - len(step1_pred))
                        step2_cap = max(0, int(remaining_budget * step2_budget_fraction))
                        if step2_cap > 0 and pp2.keys:
                            if pp2.ranked_keys_by_layer:
                                b_per_l = max(1, step2_cap // max(1, len(pp2.ranked_keys_by_layer)))
                                for _layer, rkeys in pp2.ranked_keys_by_layer.items():
                                    step2_pred.update(rkeys[:b_per_l])
                                # Trim to cap if layers × b_per_l exceeds it
                                if len(step2_pred) > step2_cap:
                                    step2_pred = set(list(step2_pred)[:step2_cap])
                            else:
                                step2_pred = set(sorted(pp2.keys)[:step2_cap])
                        step2_pred -= step1_pred  # deduplicate against step-1

                    pred = step1_pred | step2_pred
                    step1_count += len(step1_pred)
                    step2_count += len(step2_pred)
                elif mode == 'causal' and prev is not None:
                    k_by_layer = top_k_by_layer(current)
                    pred = predict_from_transition(transitions, current, k_by_layer)
                elif mode == 'prompt' and prev is not None:
                    k_by_layer = top_k_by_layer(current)
                    pred = prompt_hybrid_predict(transitions, prompt_prof, current, k_by_layer, hybrid_alpha)
                elif mode == 'causal_region' and prev is not None:
                    k_by_layer = top_k_by_layer(current)
                    base = predict_from_transition(transitions, current, k_by_layer)
                    pred = set(base)
                    if region_map and region_members:
                        for key in list(base):
                            rid = region_map.get(key)
                            if rid is not None:
                                pred |= region_members.get(rid, set())

                target_actual = all_keys(records[idx + 1])
                predicted += len(pred)
                correct += len(pred & target_actual)
                wasted += len(pred - target_actual)
                actual_total += len(target_actual)
                # Track step-2 specific usefulness against the *immediate* next token.
                # A step-2 prefetch that lands in target_actual at t+1 arrived early
                # (better than nothing); the main benefit accrues at t+2 which we
                # approximate by checking the next-token hit rate contribution.
                if step2_pred:
                    step2_useful += len(step2_pred & target_actual)
                    step2_wasted_approx += len(step2_pred - target_actual)

            exp, _ = sim.run_token(actual, pred)
            exposed.append(exp)

            if prev is not None:
                for layer, prev_set in prev.items():
                    nxt = current.get(layer, set())
                    for pe in prev_set:
                        transitions[layer][pe].update(nxt)
            pred_state.observe(current, phase=str(rec.get('phase','decode')))
            prev = current

        if mode == 'atlas_predictor':
            pred_state.finish()

    tokens = sum(len(s['records']) for s in sessions)
    total_exposed = sum(exposed)
    wall_ms = sim.now_ms
    step2_total = step2_useful + step2_wasted_approx
    return {
        'mode': mode,
        'tokens': tokens,
        'actual_experts': actual_total,
        'predicted': predicted,
        'correct_prefetch': correct,
        'false_prefetch': wasted,
        'prefetch_precision': correct / predicted if predicted else 0.0,
        'prefetch_recall': correct / actual_total if actual_total else 0.0,
        'exposed_io_ms_mean': total_exposed / len(exposed) if exposed else 0.0,
        'exposed_io_ms_p50': percentile(exposed, 50),
        'exposed_io_ms_p95': percentile(exposed, 95),
        'exposed_io_ms_p99': percentile(exposed, 99),
        'exposed_io_ms_max': max(exposed) if exposed else 0.0,
        'nvme_mb': sim.stats['nvme_mb'],
        'ram_to_vram_mb': sim.stats['ram_to_vram_mb'],
        'ram_write_mb': sim.stats['ram_write_mb'],
        'ram_preload_mb': sim.stats.get('ram_preload_mb', 0.0),
        'ram_preload_ms': sim.stats.get('ram_preload_ms', 0.0),
        'vram_hits': sim.stats['vram_hit'],
        'ram_hits': sim.stats['ram_hit'],
        'nvme_misses': sim.stats['nvme_miss'],
        'prefetch_waits': sim.stats['prefetch_wait'],
        'vram_evictions': sim.stats['vram_evictions'],
        'prefetch_requested': sim.stats['prefetch_requested'],
        'total_exposed_io_ms': total_exposed,
        'wall_ms': wall_ms,
        'effective_tps': tokens / (wall_ms / 1000.0) if wall_ms > 0 else 0.0,
        # Experiment F step-2 metrics
        'step1_prefetch_count': step1_count,
        'step2_prefetch_count': step2_count,
        'step2_useful_prefetch': step2_useful,
        'step2_wasted_prefetch': step2_wasted_approx,
        'step2_precision': step2_useful / step2_total if step2_total > 0 else 0.0,
    }

def main():
    ap = argparse.ArgumentParser(description="Trace-driven Atlas NVMe/RAM/VRAM hypothesis simulator")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--expert-size-mb", type=float, required=True)
    ap.add_argument("--vram-gb", type=float, default=8.0,
                    help="Physical GPU VRAM budget (target environment: 8 GB)")
    ap.add_argument("--vram-reserve-gb", type=float, default=1.0,
                    help="VRAM reserved for non-Atlas runtime state (KV/cache/workspace)")
    ap.add_argument("--ram-gb", type=float, default=16.0,
                    help="Host RAM budget available to Atlas warm cache (target: 16 GB)")
    ap.add_argument("--nvme-gbps", type=float, default=7.0)
    ap.add_argument("--pcie-gbps", type=float, default=12.0)
    ap.add_argument("--ram-gbps", type=float, default=40.0,
                    help="Effective host RAM copy bandwidth used by NVMe->RAM stage")
    ap.add_argument("--nvme-latency-ms", type=float, default=0.08)
    ap.add_argument("--pcie-latency-ms", type=float, default=0.02)
    ap.add_argument("--compute-ms", type=float, default=None,
                    help="Per-token compute window. Default: 1000/target-tps")
    ap.add_argument("--target-tps", type=float, default=20.0)
    ap.add_argument("--hybrid-alpha", type=float, default=0.75,
                    help="Causal-vs-prompt weight for prompt hybrid policy")
    ap.add_argument("--predictor-confidence-floor", type=float, default=0.22)
    ap.add_argument("--predictor-budget-scale", type=float, default=1.25)
    ap.add_argument("--predictor-history-window", type=int, default=4)
    ap.add_argument("--predictor-max-candidates", type=int, default=32)
    ap.add_argument("--predictor-min-count", type=int, default=2)
    ap.add_argument("--region-size", type=int, default=8)
    ap.add_argument("--min-affinity-count", type=int, default=2)
    ap.add_argument("--preload-experts-to-ram", action="store_true",
                    help="Preload entire expert corpus into RAM (eliminates NVMe during decode)")
    ap.add_argument("--vram-policy", choices=["lru", "freq"], default="lru",
                    help="VRAM eviction policy: pure recency (lru) or frequency-biased (freq)")
    ap.add_argument("--deadline-aware-budget", action="store_true",
                    help="Cap prefetch candidate budget by PCIe compute window transfer capacity")
    ap.add_argument("--prefetch-horizon", type=int, default=1, choices=[1, 2],
                    help="Prefetch horizon: 1=next-token only (default), 2=Experiment F 2-step-ahead")
    ap.add_argument("--step2-budget-fraction", type=float, default=0.5,
                    help="Fraction of remaining PCIe budget to allocate to step-2 predictions (0.0-1.0)")
    ap.add_argument("--json-report", default=None)
    args = ap.parse_args()

    if args.vram_reserve_gb < 0 or args.vram_reserve_gb >= args.vram_gb:
        raise SystemExit("--vram-reserve-gb must be >= 0 and smaller than --vram-gb")
    vram_capacity_mb = (args.vram_gb - args.vram_reserve_gb) * 1024.0
    compute_ms = args.compute_ms if args.compute_ms is not None else 1000.0 / args.target_tps
    sessions = load_sessions(args.trace)
    # Re-read full sessions for prompt profile while keeping decode records.
    with open(args.trace, "r", encoding="utf-8") as f:
        raw = [json.loads(x) for x in f if x.strip()]
    for i, s in enumerate(sessions):
        s["_all_records"] = raw[i].get("records", [])

    region_map, region_members = build_affinity_regions(sessions, args.region_size, args.min_affinity_count)
    modes = ["none", "causal", "prompt", "causal_region", "atlas_predictor", "oracle"]
    reports = {}
    print("=" * 88)
    print("ATLAS TIERED MEMORY HYPOTHESIS SIMULATOR")
    print("=" * 88)
    print(f"sessions={len(sessions)} tokens={sum(len(s['records']) for s in sessions)}")
    print(f"VRAM={args.vram_gb:.1f}GB reserve={args.vram_reserve_gb:.1f}GB cache={vram_capacity_mb/1024:.1f}GB RAM={args.ram_gb:.1f}GB NVMe={args.nvme_gbps:.2f}GB/s PCIe={args.pcie_gbps:.2f}GB/s")
    print(f"expert={args.expert_size_mb:.3f}MB compute_window={compute_ms:.3f}ms/token policy_vram={args.vram_policy} preload={args.preload_experts_to_ram} horizon={args.prefetch_horizon}")
    print()
    print(f"{'Policy':<18} {'Pref P':>8} {'Pref R':>8} {'NVMe MB':>12} {'Expose ms/tok':>15} {'P95 ms':>10} {'TPS':>10}")
    print("-" * 88)
    for mode in modes:
        r = evaluate_policy(sessions, mode, vram_capacity_mb, args.expert_size_mb,
                            args.ram_gb, args.nvme_gbps, args.pcie_gbps, args.ram_gbps,
                            args.nvme_latency_ms, args.pcie_latency_ms, compute_ms,
                            args.prefetch_horizon, args.hybrid_alpha,
                            region_map if mode == "causal_region" else None,
                            region_members if mode == "causal_region" else None,
                            mode == "causal_region",
                            {"confidence_floor": args.predictor_confidence_floor, "budget_scale": args.predictor_budget_scale,
                             "history_window": args.predictor_history_window, "max_candidates_per_layer": args.predictor_max_candidates,
                             "min_count": args.predictor_min_count} if mode == "atlas_predictor" else None,
                            preload_ram=args.preload_experts_to_ram,
                            vram_policy=args.vram_policy,
                            deadline_aware=args.deadline_aware_budget,
                            step2_budget_fraction=args.step2_budget_fraction)
        reports[mode] = r
        print(f"{mode:<18} {r['prefetch_precision']*100:>7.1f}% {r['prefetch_recall']*100:>7.1f}% "
              f"{r['nvme_mb']:>12.1f} {r['exposed_io_ms_mean']:>15.3f} "
              f"{r['exposed_io_ms_p95']:>10.3f} {r['effective_tps']:>10.2f}")
    print()
    print(f"Affinity regions: {len(region_members)}")
    print("NOT: region-lar semantic etiket deyil; trace co-activation grouping-dir.")
    print("NOT: bu simulator GPU kernel/dequantization/gerçek async I/O implementasiyası deyil.")
    print("      Məqsəd siyasətləri eyni trace və eyni hardware fərziyyələri altında müqayisə etməkdir.")

    if args.json_report:
        report = {
            "config": vars(args) | {"compute_ms": compute_ms, "vram_capacity_gb": vram_capacity_mb / 1024.0},
            "regions": {str(k): sorted([list(x) for x in v]) for k, v in region_members.items()},
            "policies": reports,
        }
        Path(args.json_report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[+] JSON report: {args.json_report}")


if __name__ == "__main__":
    main()
