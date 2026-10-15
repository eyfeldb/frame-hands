"""Paired availability and pose-proxy audit for current-seed crop replay."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np


def stats(values):
    if not values:return None
    return dict(count=len(values),median=float(np.median(values)),p95=float(np.percentile(values,95)),maximum=float(max(values)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    counts=Counter();groups={};previous={};runs={};gaps={};current_gap={};wrist_shift=[]
    path=a.result/'frames.jsonl'
    with path.open() as f:
        for line in f:
            row=json.loads(line);phase='prefix' if row['seconds']<18 else 'evaluation'
            fits={(v['side'],v['branch']):v for v in row['fits'] if v['accepted']}
            for side in (0,1):
                before=(side,'box') in fits;after=(side,'rays') in fits
                counts[f'{phase}_side{side}_rescued']+=int(after and not before);counts[f'{phase}_side{side}_lost']+=int(before and not after)
                common=before and after
                if common:wrist_shift.append(float(np.linalg.norm(np.asarray(fits[(side,'box')]['world_joints_mm'])[0]-np.asarray(fits[(side,'rays')]['world_joints_mm'])[0])))
                for branch in ('box','rays'):
                    key=(phase,side,branch);fit=fits.get((side,branch))
                    if fit is None:
                        current_gap[key]=current_gap.get(key,0)+1
                    elif current_gap.get(key):
                        gaps.setdefault(key,[]).append(current_gap.pop(key))
                    if fit is None:continue
                    counts[f'{phase}_side{side}_{branch}_accepted']+=1
                    # Compare geometry quality only on exact frames where both
                    # branches produced a fit; never compare unequal populations.
                    if not common:previous.pop(key,None);continue
                    label='_'.join(map(str,key));metrics=groups.setdefault(label,dict(ncnn_median_px=[],max_joint_step_mm=[],resets=Counter()))
                    metrics['ncnn_median_px'].extend(v['median'] for v in fit['ncnn_disagreement_px'].values());metrics['resets'][str(fit['reset_reason'])]+=1
                    old=previous.get(key);now=np.asarray(fit['world_joints_mm'])
                    if old is not None and fit['reset_reason'] is None and 0<row['stamp']-old[0]<=100000000:
                        metrics['max_joint_step_mm'].append(float(np.linalg.norm(now-old[1],axis=1).max()))
                    previous[key]=(row['stamp'],now)
    for key,n in current_gap.items():gaps.setdefault(key,[]).append(n)
    for metrics in groups.values():
        for key in ('ncnn_median_px','max_joint_step_mm'):metrics[key]=stats(metrics[key])
    summary=json.loads((a.result/'summary.json').read_text())
    controls={name:{key:dict(active=v['active'],presence=v.get('presence'),reason=v['reason']) for key,v in row.items() if key in ('background','blank')} for name,row in summary['controls'].items()}
    out=dict(counts=counts,common_frame_metrics=groups,paired_wrist_shift_mm=stats(wrist_shift),missing_frame_runs={str(k):stats(v) for k,v in gaps.items()},controls=controls,exact_original_model_outputs=summary['counts']['exact_original_model_outputs'],exact_original_native_fits=summary['counts']['exact_original_native_fits'],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),frames_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),limitations='Current external seed/identity restrict availability. Missing-frame runs include startup and external absence. Motion comparisons omit branch reset steps and can differ slightly in counts; they are not speed-normalized accuracy. ncnn references are missing for originally rejected camera crops; rescued-frame geometry needs visual or separately joined evidence. Common-frame agreement is not3Dground truth.')
    a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))


if __name__=='__main__':main()
