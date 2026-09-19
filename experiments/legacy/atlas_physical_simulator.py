from __future__ import annotations
import argparse, json, math
from collections import OrderedDict, defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
import statistics
from atlas_predictor import AtlasPredictor


def pct(xs,p):
    if not xs:return 0.0
    s=sorted(xs); k=(len(s)-1)*p/100; lo=int(k); hi=min(lo+1,len(s)-1)
    return s[lo] if lo==hi else s[lo]*(hi-k)+s[hi]*(k-lo)


def load_sessions(path):
    out=[]
    with open(path,encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            o=json.loads(line); rec=o.get('records',[])
            out.append(rec)
    return out


def keys(rec):
    return {(str(l),int(e)) for l,es in (rec.get('experts_by_layer') or {}).items() for e in es}

@dataclass
class Item:
    size_mb: float
    ready_ms: float

class LRU:
    def __init__(self,cap_mb): self.cap=max(0.0,cap_mb); self.used=0.0; self.d=OrderedDict()
    def get(self,k):
        x=self.d.get(k)
        if x:self.d.move_to_end(k)
        return x
    def put(self,k,x):
        if x.size_mb>self.cap:return False
        old=self.d.pop(k,None)
        if old:self.used-=old.size_mb
        self.d[k]=x; self.used+=x.size_mb
        while self.d and self.used>self.cap:
            _,old=self.d.popitem(last=False); self.used-=old.size_mb
        return k in self.d

class PhysicalSim:
    def __init__(self,physical, vram_gb, ram_gb, nvme_gbps, pcie_gbps, ram_gbps, nvme_lat, pcie_lat, compute_ms, merge_gap_kb=4):
        self.physical=physical; self.vram=LRU(vram_gb*1024); self.ram=LRU(ram_gb*1024)
        self.nvme=nvme_gbps; self.pcie=pcie_gbps; self.ram_bw=ram_gbps; self.nvme_lat=nvme_lat; self.pcie_lat=pcie_lat; self.compute=compute_ms
        self.merge_gap=merge_gap_kb*1024; self.nvme_busy=0; self.pcie_busy=0; self.ram_busy=0; self.now=0; self.stats=Counter(); self.exposed=[]
        self.experts=physical.get('experts',{})
        self._preloaded=False
        self._reset_runtime_tracking()

    def size_mb(self,k):
        e=self.experts.get(f'{k[0]}:{k[1]}')
        if not e or not e.get('chunks') or not e.get('exact',True): return None
        return sum(c.get('size_bytes') or 0 for c in e['chunks'])/1048576

    def _reset_runtime_tracking(self):
        self.seen_ever=set()
        self.pending_predictions={}
        self.prediction_leads=[]

    def _cache_state(self, k):
        vit=self.vram.get(k)
        if vit:
            return 'vram', vit
        rit=self.ram.get(k)
        if rit:
            return 'ram', rit
        return None, None

    def _record_actual_prediction_outcome(self, actual, start_ms):
        for k in actual:
            meta=self.pending_predictions.pop(k, None)
            if not meta:
                continue
            predicted_at, source_was_nvme = meta
            self.prediction_leads.append(max(0.0, start_ms-predicted_at))
            if not source_was_nvme:
                continue
            state, item = self._cache_state(k)
            ready = item.ready_ms if item else float('inf')
            if state and ready <= start_ms:
                self.stats['predictions_that_prevented_nvme_miss'] += 1
            else:
                self.stats['predictions_arrived_too_late'] += 1
        self.pending_predictions.clear()

    def ranges_for(self, ks):
        rows=[]
        for k in ks:
            e=self.experts.get(f'{k[0]}:{k[1]}')
            if not e or not e.get('exact',True): continue
            for c in e.get('chunks',[]):
                if c.get('offset') is not None and c.get('size_bytes'):
                    rows.append((int(c['offset']),int(c['offset'])+int(c['size_bytes']),k))
        rows.sort(); groups=[]
        for a,b,k in rows:
            if not groups or a > groups[-1][1] + self.merge_gap:
                groups.append([a,b,{k}])
            else:
                groups[-1][1]=max(groups[-1][1],b); groups[-1][2].add(k)
        return groups

    def _nvme_group(self,start, total_bytes):
        size_gb=total_bytes/1073741824; s=max(start,self.nvme_busy); dur=self.nvme_lat + (size_gb/self.nvme*1000 if self.nvme>0 else 1e18); done=s+dur; self.nvme_busy=done
        self.stats['nvme_read_mb'] += total_bytes/1048576; self.stats['nvme_io_ops']+=1; self.stats['nvme_ms']+=dur
        return done

    def _ram_write(self,start,total_bytes):
        s=max(start,self.ram_busy); size_gb=total_bytes/1073741824; dur=(size_gb/self.ram_bw*1000 if self.ram_bw>0 else 1e18); done=s+dur; self.ram_busy=done; self.stats['ram_write_mb']+=total_bytes/1048576; return done

    def _pcie_group(self,start,total_bytes):
        s=max(start,self.pcie_busy); size_gb=total_bytes/1073741824; dur=self.pcie_lat+(size_gb/self.pcie*1000 if self.pcie>0 else 1e18); done=s+dur; self.pcie_busy=done; self.stats['pcie_transfer_mb']+=total_bytes/1048576; self.stats['pcie_io_ops']+=1; return done

    def _preload_experts_to_ram(self, start):
        """Load the complete exact expert pool into RAM, without touching VRAM.

        This models Atlas' intended tiering when the expert corpus fits entirely
        in host RAM: NVMe becomes cold storage while decode only pays RAM->VRAM.
        The transfer is startup work and is reported separately from decode TPS.
        """
        ks=[]
        for ek, e in self.experts.items():
            if e.get('exact', True) and e.get('chunks'):
                try:
                    layer, expert = ek.rsplit(':', 1)
                    k=(str(layer), int(expert))
                except ValueError:
                    continue
                if self.size_mb(k) is not None:
                    ks.append(k)
        groups=self.ranges_for(ks)
        total_bytes=0
        for a,b,_ in groups:
            total_bytes += max(0, b-a)
        if total_bytes/1048576 > self.ram.cap:
            return False, start, total_bytes/1048576
        cursor=start
        for a,b,gks in groups:
            total=b-a
            nvdone=self._nvme_group(cursor,total)
            ramdone=self._ram_write(nvdone,total)
            for k in gks:
                sz=self.size_mb(k)
                self.ram.put(k,Item(sz,ramdone))
            cursor=ramdone
        self.stats['ram_preload_mb'] += total_bytes/1048576
        self.stats['ram_preload_ms'] += max(0.0, cursor-start)
        self._preloaded=True
        return True, cursor, total_bytes/1048576

    def _stage_from_nvme(self, ks, start):
        groups=self.ranges_for(ks); completions={}
        for a,b,gks in groups:
            requested_bytes=sum((self.size_mb(k) or 0)*1048576 for k in gks)
            total=b-a; wasted=max(0,total-requested_bytes); self.stats['read_ahead_wasted_mb']+=wasted/1048576
            nvdone=self._nvme_group(start,total); ramdone=self._ram_write(nvdone,total)
            for k in gks:
                sz=self.size_mb(k); d=self._pcie_group(ramdone,sz*1048576)
                self.ram.put(k,Item(sz,ramdone))
                self.vram.put(k,Item(sz,d)); completions[k]=d
        return completions

    def _stage_from_ram(self,ks,start):
        comps={}
        for k in ks:
            it=self.ram.get(k)
            if not it: continue
            done=self._pcie_group(max(start,it.ready_ms),it.size_mb*1048576); self.vram.put(k,Item(it.size_mb,done)); comps[k]=done
        return comps

    def ensure_batch(self, ks, start):
        missing_ram=[]; pending={}; max_ready=start
        for k in sorted(ks):
            it=self.vram.get(k)
            if it:
                if it.ready_ms<=start: self.stats['vram_hit']+=1; continue
                self.stats['prefetch_wait']+=1; pending[k]=it.ready_ms; max_ready=max(max_ready,it.ready_ms); continue
            rit=self.ram.get(k)
            if rit: missing_ram.append(k); continue
            missing_ram.append(k); self.stats['nvme_miss']+=1
        ram_keys=[]; nvme_keys=[]
        for k in missing_ram:
            if self.ram.get(k): ram_keys.append(k)
            else: nvme_keys.append(k)
        comps={}
        if nvme_keys:
            comps.update(self._stage_from_nvme(nvme_keys,start))
        if ram_keys:
            comps.update(self._stage_from_ram(ram_keys,start))
        for k,d in comps.items(): max_ready=max(max_ready,d)
        return max_ready

    def prefetch(self,ks,start):
        ks=[k for k in sorted(ks) if self.size_mb(k) is not None]
        todo=[]
        for k in ks:
            if self.vram.get(k): self.stats['prefetch_already_hot']+=1; continue
            todo.append(k)
        if not todo:return
        from_ram=[k for k in todo if self.ram.get(k)]
        from_nv=[k for k in todo if not self.ram.get(k)]
        if from_nv:
            self.stats['prefetch_requested']+=len(from_nv); self._stage_from_nvme(from_nv,start)
        if from_ram:
            self.stats['prefetch_requested']+=len(from_ram); self._stage_from_ram(from_ram,start)

    def run(self,sessions,policy='none', predictor_kwargs=None):
        predictor_kwargs = dict(predictor_kwargs or {})
        preload_experts_to_ram = bool(predictor_kwargs.get('preload_experts_to_ram', False))
        if preload_experts_to_ram:
            ok, preload_done, preload_mb = self._preload_experts_to_ram(self.now)
            self.stats['ram_preload_requested'] = 1
            self.stats['ram_preload_fit'] = 1 if ok else 0
            self.stats['ram_preload_capacity_mb'] = self.ram.cap
            self.stats['ram_preload_target_mb'] = preload_mb
            if ok:
                self.now = preload_done
        predictor_kwargs.pop('preload_experts_to_ram', None)
        prefetch_horizon = int(predictor_kwargs.pop('prefetch_horizon', 1))
        step2_budget_fraction = float(predictor_kwargs.pop('step2_budget_fraction', 0.5))
        predictor = AtlasPredictor(**predictor_kwargs) if policy == 'atlas_predictor' else None
        decode_tokens_total = 0
        decode_active_ms = 0.0
        decode_start_ms = None
        prompt_tokens_total = 0
        for recs in sessions:
            self.vram=LRU(self.vram.cap)
            prev=None; transitions=defaultdict(lambda:defaultdict(Counter))
            pstate = predictor.begin_session() if predictor else None
            prompt_last = None
            for i,rec in enumerate(recs):
                phase=str(rec.get('phase','decode'))
                actual=keys(rec); pred=set()
                next_phase=str(recs[i+1].get('phase','decode')) if i+1<len(recs) else None

                if policy == 'oracle' and i+1<len(recs):
                    # Oracle is allowed to see the exact next record by design.
                    pred=keys(recs[i+1])
                elif policy == 'causal' and phase=='decode' and i+1<len(recs) and next_phase=='decode' and prev is not None:
                    for layer in {x[0] for x in actual}:
                        scores=Counter()
                        for pe in prev.get(layer,set()): scores.update(transitions[layer][pe])
                        k=len(actual) // max(1,len({x[0] for x in actual}))
                        pred|={(layer,e) for e,_ in scores.most_common(k)}
                elif policy == 'atlas_predictor':
                    cur={l:{e for ll,e in actual if ll==l} for l in {ll for ll,_ in actual}}
                    if phase=='prompt' and next_phase=='decode' and pstate is not None:
                        # Cold-start prediction uses only completed prior-session
                        # prompt->first-decode boundary statistics.
                        pred=pstate.predict_boundary(cur, max_candidates_per_layer=(predictor_kwargs or {}).get('max_candidates_per_layer',1)).keys
                    elif phase=='decode' and next_phase=='decode' and pstate is not None:
                        vram_keys = {(str(l), int(e)) for l, e in self.vram.d.keys()}
                        pp = pstate.predict(cur, phase='decode', exclude_keys=vram_keys)
                        step1_pred = pp.keys
                        step2_pred = set()
                        if prefetch_horizon >= 2 and step1_pred:
                            hypo_layers = defaultdict(set)
                            for l, e in step1_pred:
                                hypo_layers[str(l)].add(int(e))
                            pp2 = pstate.predict(hypo_layers, phase='decode', exclude_keys=vram_keys | step1_pred)
                            max_pcie_mb = max(0.0, (self.compute - self.pcie_lat) / 1000.0 * self.pcie * 1024.0) if self.pcie > 0 else float('inf')
                            max_deadline_candidates = max(1, int(max_pcie_mb / 1.2))
                            remaining_budget = max(0, max_deadline_candidates - len(step1_pred))
                            step2_cap = max(0, int(remaining_budget * step2_budget_fraction))
                            if step2_cap > 0 and pp2.keys:
                                if pp2.ranked_keys_by_layer:
                                    b_per_l = max(1, step2_cap // max(1, len(pp2.ranked_keys_by_layer)))
                                    for _layer, rkeys in pp2.ranked_keys_by_layer.items():
                                        step2_pred.update(rkeys[:b_per_l])
                                    if len(step2_pred) > step2_cap:
                                        step2_pred = set(list(step2_pred)[:step2_cap])
                                else:
                                    step2_pred = set(sorted(pp2.keys)[:step2_cap])
                            step2_pred -= step1_pred
                        pred = step1_pred | step2_pred

                if pred and i+1<len(recs):
                    nxt_eval=keys(recs[i+1])
                    self.stats['predicted_experts'] += len(pred)
                    self.stats['correct_prefetch'] += len(pred & nxt_eval)
                    self.stats['false_prefetch'] += len(pred - nxt_eval)
                start=self.now
                self._record_actual_prediction_outcome(actual, start)
                # Snapshot the cache state at the moment the prediction is made.
                # This must happen BEFORE ensure_batch(actual), otherwise a
                # prediction that is also part of the current demand can look
                # artificially hot and cold-prediction coverage becomes zero.
                prediction_state = {}
                if pred:
                    for k in pred:
                        state, item = self._cache_state(k)
                        prediction_state[k] = (state, item.ready_ms if item else float('inf'))
                        if state:
                            self.stats['predictions_already_hot'] += 1
                        else:
                            self.stats['predictions_cold'] += 1

                for k in actual:
                    if k in self.seen_ever:
                        state, _ = self._cache_state(k)
                        if state is None:
                            self.stats['eviction_nvme_misses'] += 1
                    else:
                        state, _ = self._cache_state(k)
                        if state is None:
                            self.stats['cold_nvme_misses'] += 1
                ready=self.ensure_batch(actual,start); exposed=max(0,ready-start); compute_start=ready
                if pred:
                    for k in pred:
                        state, item_ready = prediction_state[k]
                        # Lead time begins when the prediction was emitted, not
                        # after current-demand staging has completed.
                        self.pending_predictions[k]=(compute_start, state is None)
                    self.prefetch(pred,compute_start)
                self.exposed.append(exposed)

                if phase=='prompt':
                    prompt_tokens_total += 1
                else:
                    if decode_start_ms is None:
                        decode_start_ms = compute_start
                    decode_tokens_total += 1
                    decode_active_ms += exposed + self.compute
                    self.stats['tokens'] += 1
                    self.stats['actual_experts'] += len(actual)
                    self.stats['exposed_io_ms'] += exposed

                self.now=compute_start+self.compute
                cur=defaultdict(set)
                for l,e in actual: cur[l].add(e)
                if pstate is not None:
                    pstate.observe(cur, phase=phase)
                if phase=='decode' and prev is not None:
                    for l,ps in prev.items():
                        for pe in ps: transitions[l][pe].update(cur.get(l,set()))
                prev=cur if phase=='decode' else None
                self.seen_ever.update(actual)
            if pstate is not None:
                pstate.finish()

        self.decode_tokens_total=decode_tokens_total
        self.prompt_tokens_total=prompt_tokens_total
        self.decode_start_ms=decode_start_ms
        self.decode_active_ms=decode_active_ms
        return self.result(policy)

    def result(self,policy):
        ex=self.exposed; wall=self.now
        return {'policy':policy,'tokens':self.stats['tokens'],'exposed_io_ms_mean':sum(ex)/len(ex) if ex else 0,'p50_ms':pct(ex,50),'p95_ms':pct(ex,95),'p99_ms':pct(ex,99),'nvme_read_mb':self.stats['nvme_read_mb'],'nvme_io_ops':self.stats['nvme_io_ops'],'pcie_transfer_mb':self.stats['pcie_transfer_mb'],'pcie_io_ops':self.stats['pcie_io_ops'],'read_ahead_wasted_mb':self.stats['read_ahead_wasted_mb'],'nvme_misses':self.stats['nvme_miss'],'cold_nvme_misses':self.stats['cold_nvme_misses'],'eviction_nvme_misses':self.stats['eviction_nvme_misses'],'prefetch_waits':self.stats['prefetch_wait'],'prefetch_requested':self.stats['prefetch_requested'],'prefetch_already_hot':self.stats['prefetch_already_hot'],'predictions_cold':self.stats['predictions_cold'],'predictions_already_hot':self.stats['predictions_already_hot'],'predictions_that_prevented_nvme_miss':self.stats['predictions_that_prevented_nvme_miss'],'predictions_arrived_too_late':self.stats['predictions_arrived_too_late'],'predicted_experts':self.stats['predicted_experts'],'correct_prefetch':self.stats['correct_prefetch'],'false_prefetch':self.stats['false_prefetch'],'prefetch_precision':self.stats['correct_prefetch']/self.stats['predicted_experts'] if self.stats['predicted_experts'] else 0.0,'prediction_lead_ms_p50':pct(self.prediction_leads,50),'prediction_lead_ms_p95':pct(self.prediction_leads,95),'prediction_lead_ms_p99':pct(self.prediction_leads,99),'ram_preload_mb':self.stats['ram_preload_mb'],'ram_preload_ms':self.stats['ram_preload_ms'],'ram_preload_fit':bool(self.stats['ram_preload_fit']),'decode_wall_ms':((wall-self.decode_start_ms) if self.decode_start_ms is not None else 0.0),'decode_active_ms':self.decode_active_ms,'effective_tps_including_startup':(self.decode_tokens_total/(wall/1000)) if wall>0 and self.decode_tokens_total else 0.0,'effective_tps':(self.decode_tokens_total/(self.decode_active_ms/1000)) if self.decode_active_ms>0 else 0.0,'prompt_tokens':self.prompt_tokens_total}


def main():
    ap=argparse.ArgumentParser(description='Atlas physical GGUF I/O simulator')
    ap.add_argument('--trace',required=True); ap.add_argument('--physical-map',required=True)
    ap.add_argument('--vram-gb',type=float,default=8); ap.add_argument('--vram-reserve-gb',type=float,default=1)
    ap.add_argument('--ram-gb',type=float,default=16); ap.add_argument('--ram-reserve-gb',type=float,default=2)
    ap.add_argument('--nvme-gbps',type=float,default=7); ap.add_argument('--pcie-gbps',type=float,default=12); ap.add_argument('--ram-bandwidth-gbps',type=float,default=40)
    ap.add_argument('--nvme-latency-ms',type=float,default=.08); ap.add_argument('--pcie-latency-ms',type=float,default=.02)
    ap.add_argument('--target-tps',type=float,default=20); ap.add_argument('--merge-gap-kb',type=float,default=4)
    ap.add_argument('--predictor-confidence-floor',type=float,default=0.10)
    ap.add_argument('--predictor-budget-scale',type=float,default=1.00)
    ap.add_argument('--predictor-max-candidates',type=int,default=1)
    ap.add_argument('--predictor-source-top-n',type=int,default=32)
    ap.add_argument('--predictor-persistent-fallback',action='store_true')
    ap.add_argument('--preload-experts-to-ram',action='store_true')
    args=ap.parse_args()
    pm=json.loads(Path(args.physical_map).read_text(encoding='utf-8')); sessions=load_sessions(args.trace)
    vram=args.vram_gb-args.vram_reserve_gb; ram=args.ram_gb-args.ram_reserve_gb
    if vram<=0 or ram<=0: raise SystemExit('reserve must be smaller than physical capacity')
    compute=1000/args.target_tps
    print(f'Physical map: {pm.get("source")} exact experts={sum(1 for e in pm.get("experts",{}).values() if e.get("exact",True))}')
    print(f'Target budget: VRAM={vram:.2f}GB RAM={ram:.2f}GB NVMe={args.nvme_gbps}GB/s PCIe={args.pcie_gbps}GB/s')
    for mode in ('none','causal','atlas_predictor','oracle'):
        sim=PhysicalSim(pm,vram,ram,args.nvme_gbps,args.pcie_gbps,args.ram_bandwidth_gbps,args.nvme_latency_ms,args.pcie_latency_ms,compute,args.merge_gap_kb)
        kwargs = {'confidence_floor': args.predictor_confidence_floor, 'budget_scale': args.predictor_budget_scale, 'history_window': 4, 'max_candidates_per_layer': args.predictor_max_candidates, 'source_top_n': args.predictor_source_top_n, 'min_count': 2, 'persistent_fallback': args.predictor_persistent_fallback, 'preload_experts_to_ram': args.preload_experts_to_ram} if mode == 'atlas_predictor' else {'preload_experts_to_ram': args.preload_experts_to_ram}
        r=sim.run(sessions,mode, kwargs); print(json.dumps(r,ensure_ascii=False))

if __name__=='__main__':main()
