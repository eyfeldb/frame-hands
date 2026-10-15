"""Audit fallback preservation, image agreement and visible output steps."""
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--result',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    counts=Counter();metrics={};differences={};previous={};switches=[]
    frames=a.result/'frames.jsonl'
    with frames.open() as stream:
        for line in stream:
            row=json.loads(line);phase='prefix' if row['seconds']<18 else 'evaluation'
            accepted={(f['side'],f['branch']):f for f in row['fits'] if f['accepted']}
            for side in (0,1):
                original=accepted.get((side,'box'))
                for branch in ('box','rays','fallback_carry','fallback_reset'):
                    fit=accepted.get((side,branch));key=f'{phase}_{side}_{branch}'
                    if fit is None:continue
                    counts[key+'_accepted']+=1
                    current=np.asarray(fit['world_joints_mm'])
                    metrics.setdefault(key,dict(common_ncnn_px=[],recovered_ncnn_px=[],step_including_resets_mm=[],step_without_resets_mm=[],switch_step_mm=[]))
                    label='common_ncnn_px' if original else 'recovered_ncnn_px'
                    metrics[key][label].extend(v['median'] for v in fit['ncnn_disagreement_px'].values())
                    if original and branch.startswith('fallback'):
                        assert all(v=='box' for v in fit['crop_modes'].values())
                        counts[branch+'_original_supported']+=1;counts[branch+'_exact_original_fit']+=int(fit['fit']==original['fit'])
                        delta=float(np.linalg.norm(current-np.asarray(original['world_joints_mm']),axis=1).max())
                        differences.setdefault(branch,[]).append(delta)
                    if not original and branch.startswith('fallback'):counts[branch+'_recovered']+=1
                    old=previous.get((side,branch))
                    if old is not None and old[2]==phase and old[3]==fit['generation'] and 0<row['stamp']-old[0]<=100000000:
                        step=float(np.linalg.norm(current-old[1],axis=1).max());metrics[key]['step_including_resets_mm'].append(step)
                        if fit['reset_reason'] is None:metrics[key]['step_without_resets_mm'].append(step)
                        if fit['crop_switch']:
                            metrics[key]['switch_step_mm'].append(step)
                            switches.append(dict(seconds=row['seconds'],side=side,branch=branch,max_joint_step_mm=step,modes=fit['crop_modes']))
                    previous[(side,branch)]=(row['stamp'],current,phase,fit['generation'])
                if original:
                    for branch in ('fallback_carry','fallback_reset'):
                        counts[branch+'_lost_original']+=int((side,branch) not in accepted)
    for group in metrics.values():
        for key,values in group.items():group[key]=stats(values)
    summary=json.loads((a.result/'summary.json').read_text())
    out=dict(counts=counts,metrics=metrics,original_supported_max_joint_difference_mm={k:stats(v) for k,v in differences.items()},largest_switch_steps=sorted(switches,key=lambda x:x['max_joint_step_mm'],reverse=True)[:12],exact_controls=dict(box=summary['counts']['box_exact_fits'],rays=summary['counts']['rays_exact_fits']),frames_sha256=hashlib.sha256(frames.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),limitations='Image agreement uses exact source-ID matched ncnn observations, including new joins for previously rejected crops. It is not3Dtruth. Steps include real movement and gaps up to100ms; reset-inclusive figures expose visible jumps but do not measure jitter alone. External crop/side seeds remain. No live acceptance.')
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(counts=counts,original_supported_differences=out['original_supported_max_joint_difference_mm'],largest_switch_steps=out['largest_switch_steps']),indent=2))
    for key,group in metrics.items():
        if key.endswith(('fallback_carry','fallback_reset')):print(key,json.dumps(group))


if __name__=='__main__':main()
