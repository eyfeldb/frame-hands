"""Audit exact controls, fallback preservation and paired stereo-start outcomes."""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
from audit_mercury_crop_fallback import stats


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--result',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    protocol=json.loads((a.result/'protocol.json').read_text());reference=Path(protocol['reference']);counts=Counter();metrics={};last={};events=[];starts=[]
    with (reference/'frames.jsonl').open() as original,(a.result/'frames.jsonl').open() as candidate:
        for line in original:
            ref=json.loads(line);row=json.loads(next(candidate));assert ref['stamp']==row['stamp'];assert len(ref['fits'])==len(row['fits'])
            for before,fit in zip(ref['fits'],row['fits']):
                assert all(fit[k]==v for k,v in before.items());counts['exact_controls']+=1
                unchanged=fit['fit']==fit['candidate_fit'];counts['unchanged_candidate_fits']+=unchanged
                if fit['initialization'] is not None:
                    counts['initialization_'+fit['initialization']['reason']]+=1
                    if not fit['initialization']['accepted']:assert unchanged;counts['exact_rejected_startup_fits']+=1
                    else:starts.append(dict(seconds=row['seconds'],stamp=row['stamp'],side=fit['side'],branch=fit['branch'],initialization=fit['initialization']))
                phase='prefix' if row['seconds']<18 else 'evaluation';key=f'{phase}_side{fit["side"]}_{fit["branch"]}';group=metrics.setdefault(key,{name:dict(ncnn_px=[],joint_step_mm=[]) for name in ('native','stereo')})
                for name,field,position in [('native','ncnn_disagreement_px','world_joints_mm'),('stereo','candidate_ncnn_disagreement_px','candidate_world_joints_mm')]:
                    group[name]['ncnn_px'].extend(v['median'] for v in fit[field].values());previous=last.get((key,name));points=np.asarray(fit[position])
                    if previous is not None and previous[2]==fit['generation'] and 0<row['stamp']-previous[0]<=100000000:group[name]['joint_step_mm'].append(float(np.linalg.norm(points-previous[1],axis=1).max()))
                    last[(key,name)]=(row['stamp'],points,fit['generation'])
                if fit['ncnn_disagreement_px'] and fit['candidate_ncnn_disagreement_px']:
                    native=float(np.mean([v['median'] for v in fit['ncnn_disagreement_px'].values()]));stereo=float(np.mean([v['median'] for v in fit['candidate_ncnn_disagreement_px'].values()]))
                    events.append(dict(seconds=row['seconds'],stamp=row['stamp'],side=fit['side'],branch=fit['branch'],native_px=native,stereo_px=stereo,delta_px=stereo-native))
        assert next(candidate,None) is None
    for group in metrics.values():
        for arm in group.values():
            for key,value in arm.items():arm[key]=stats(value)
    assert counts['exact_controls']==4464
    relevant=[event for event in events if event['branch']=='fallback_carry']
    report=dict(counts=counts,metrics=metrics,accepted_initializations=starts,largest_regressions=sorted(relevant,key=lambda r:r['delta_px'],reverse=True)[:8],largest_improvements=sorted(relevant,key=lambda r:r['delta_px'])[:8],frames_sha256=hashlib.sha256((a.result/'frames.jsonl').read_bytes()).hexdigest(),limitations='Fixed observation/reset/side schedule. Candidate changes only startup and following own history; image agreement and steps are proxies, not physical ground truth. Selected extremes are diagnostic, not independent accuracy validation.')
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(counts=counts,starts=starts,regression=report['largest_regressions'][:1],improvement=report['largest_improvements'][:1]),indent=2))


if __name__=='__main__':main()
