"""Score causal association against sparse private visual labels; no model inference."""
import argparse,json,time
from pathlib import Path
import sys
import numpy as np
from frame_data import read_audit,load_calibration,Ownership
from temporal_association import TemporalAssociation
from beam_association import BeamAssociation
from run_mesh_trial import pose_matrix

def verdict(expected,actual):
    if expected is None:
        return 'rejected_nonhand' if actual is None else 'false_positive'
    return 'missing' if actual is None else 'correct' if actual==expected else 'wrong'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,required=True);parser.add_argument('--upstream',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--labels',type=Path,help='Independent annotation file; defaults to session development labels')
    parser.add_argument('--policies',choices=('fresh','temporal','bearing','continuity','beam'),nargs='+',default=['fresh','temporal'])
    a=parser.parse_args();sys.path.insert(0,str(a.upstream));p=a.session
    audits=read_audit(p/'frame/live-views.txt')
    labeldata=json.loads((a.labels or p/'visual-review/identity-labels.json').read_text())
    labels={(x['raw_ns'],x['camera'],x['source_id']):x['physical_side'] for x in labeldata['labels']}
    policies={name:Ownership() if name=='fresh' else BeamAssociation(load_calibration(p/'calibration')) if name=='beam' else TemporalAssociation(load_calibration(p/'calibration'),retain_depth=name=='bearing',reattach_prior=name=='continuity') for name in a.policies}
    scores={n:dict(correct=0,wrong=0,missing=0,rejected_nonhand=0,false_positive=0,view_outputs=0,ms=[]) for n in policies};rows=[]
    a.output.mkdir(parents=True,exist_ok=False)
    with (a.output/'trace.jsonl').open('x') as trace:
        for line in (p/'predictions.jsonl').open():
            r=json.loads(line);stamp=r['capture_raw_ns'];obs=audits[stamp];world=pose_matrix(r['capture_pose']['capture'])
            for name,policy in policies.items():
                t=time.perf_counter();assigned,candidates,_=policy.assign(obs,world) if name!='fresh' else policy.assign(obs)
                scores[name]['ms'].append((time.perf_counter()-t)*1000)
                choices={(v['camera'],v['id']):side for side,i in assigned.items() for v in candidates[i]}
                scores[name]['view_outputs']+=len(choices)
                if name!='fresh':trace.write(json.dumps(dict(policy=name,stamp=stamp,**policy.last_debug,ambiguous=bool(policy.ambiguous)))+'\n')
                for v in obs['views']:
                    key=(stamp,v['camera'],v['id'])
                    if key not in labels:continue
                    actual=choices.get(key[1:]);result=verdict(labels[key],actual)
                    scores[name][result]+=1
                    rows.append(dict(policy=name,stamp=stamp,camera=v['camera'],id=v['id'],expected=labels[key],actual=actual,verdict=result))
    if any(sum(1 for r in rows if r['policy']==name)!=len(labeldata['labels']) for name in policies):raise ValueError('Not all annotations were evaluated, or labels contain duplicates')
    for s in scores.values():s['processing_ms_p50_p95_max']=np.percentile(s.pop('ms'),[50,95,100]).tolist()
    (a.output/'labels.json').write_text(json.dumps(rows,indent=2)+'\n');(a.output/'summary.json').write_text(json.dumps(scores,indent=2)+'\n');print(json.dumps(scores,indent=2))
if __name__=='__main__':main()
