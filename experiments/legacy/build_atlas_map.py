"""Build a trace-derived Atlas logical map.

This is deliberately NOT a GGUF tensor-offset parser. It creates the data-driven
routing/affinity/region layer that can later be joined to a real GGUF structural
map. The output never labels experts as "reasoning" or "coding".
"""
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path


def load(path):
    out=[]
    with open(path,encoding='utf-8') as f:
        for line in f:
            if line.strip(): out.append(json.loads(line))
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--trace',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--region-size',type=int,default=8)
    ap.add_argument('--min-affinity-count',type=int,default=2)
    args=ap.parse_args()
    sessions=load(args.trace)
    expert_stats=Counter(); session_stats=defaultdict(set); transitions=Counter(); affinity=Counter()
    layers=set(); experts_by_layer=defaultdict(set)
    for si,s in enumerate(sessions):
        records=s.get('records',[])
        for rec in records:
            if rec.get('phase')=='prompt': continue
            ebl=rec.get('experts_by_layer',{}) or {}
            for l,es in ebl.items():
                l=str(l); layers.add(l)
                xs=sorted(set(map(int,es)))
                for e in xs:
                    key=(l,e); expert_stats[key]+=1; session_stats[key].add(si); experts_by_layer[l].add(e)
                for i,a in enumerate(xs):
                    for b in xs[i+1:]: affinity[(l,a,b)]+=1
            ordered=sorted(ebl,key=lambda x:int(x))
            for a,b in zip(ordered,ordered[1:]):
                for ea in set(ebl[a]):
                    for eb in set(ebl[b]): transitions[(str(a),int(ea),str(b),int(eb))]+=1
    # Greedy regions using strong same-layer edges.
    by_layer=defaultdict(list)
    for (l,a,b),c in affinity.items():
        if c>=args.min_affinity_count: by_layer[l].append((c,a,b))
    regions={}; expert_region={}; rid=0
    for l in sorted(experts_by_layer,key=lambda x:int(x)):
        adj=defaultdict(set)
        for c,a,b in sorted(by_layer[l],reverse=True): adj[a].add(b); adj[b].add(a)
        remaining=set(experts_by_layer[l])
        while remaining:
            seed=max(remaining,key=lambda e:(len(adj[e]&remaining),-e))
            group=[seed]; remaining.remove(seed)
            while len(group)<args.region_size:
                cand=[e for e in remaining if adj[e]&set(group)]
                if not cand: break
                e=max(cand,key=lambda x:(len(adj[x]&set(group)),-x)); group.append(e); remaining.remove(e)
            regions[str(rid)]={'layer':l,'experts':sorted(group),'kind':'co_activation'}
            for e in group: expert_region[(l,e)]=rid
            rid+=1
    for l in sorted(experts_by_layer,key=lambda x:int(x)):
        for e in sorted(experts_by_layer[l]):
            if (l,e) not in expert_region:
                regions[str(rid)]={'layer':l,'experts':[e],'kind':'singleton'}
                expert_region[(l,e)]=rid; rid+=1
    payload={
      'format':'atlas-logical-map-v1',
      'source':str(Path(args.trace).name),
      'note':'Trace-derived affinity map; not a semantic expert map and not a GGUF tensor-offset map.',
      'layers':{}, 'regions':regions, 'experts':{}, 'transitions':{}
    }
    for l in sorted(layers,key=lambda x:int(x)):
        payload['layers'][l]={'expert_count':len(experts_by_layer[l])}
        for e in sorted(experts_by_layer[l]):
            k=f'{l}:{e}'
            payload['experts'][k]={
              'layer':int(l),'expert':e,'observations':expert_stats[(l,e)],
              'sessions':len(session_stats[(l,e)]),'region':expert_region[(l,e)]
            }
    # Keep only strongest transition edges to prevent map explosion.
    for (a,ea,b,eb),c in sorted(transitions.items(),key=lambda x:x[1],reverse=True)[:50000]:
        payload['transitions'][f'{a}:{ea}->{b}:{eb}']=c
    payload['affinity_edge_count']=sum(1 for c in affinity.values() if c>=args.min_affinity_count)
    Path(args.out).write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'[+] map: {args.out}')
    print(f'[+] layers={len(layers)} experts={len(payload["experts"])} regions={len(regions)} transitions_kept={len(payload["transitions"])}')

if __name__=='__main__': main()
