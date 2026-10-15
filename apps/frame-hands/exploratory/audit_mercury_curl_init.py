"""Audit factorial startup controls and paired recorded outcomes."""
import argparse,json,hashlib
from collections import Counter
from pathlib import Path
import numpy as np
from audit_mercury_crop_fallback import stats


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--result',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();protocol=json.loads((a.result/'protocol.json').read_text());counts=Counter();metrics={};last={};events=[];finger_targets=[]
    with (Path(protocol['reference'])/'frames.jsonl').open() as original,(a.result/'frames.jsonl').open() as candidate:
        for line in original:
            ref=json.loads(line);row=json.loads(next(candidate));assert row['stamp']==ref['stamp'] and len(row['fits'])==len(ref['fits'])
            for before,fit in zip(ref['fits'],row['fits']):
                assert all(fit[k]==v for k,v in before.items());counts['exact_original_controls']+=1
                assert fit['candidate_arms']['stereo']['fit']==before['candidate_fit'];counts['exact_stereo_controls']+=1
                phase='prefix' if row['seconds']<18 else 'evaluation';key=f'{phase}_side{fit["side"]}_{fit["branch"]}';arms={'native':dict(fit=fit['fit'],world_joints_mm=fit['world_joints_mm'],ncnn_disagreement_px=fit['ncnn_disagreement_px']),**fit['candidate_arms']};group=metrics.setdefault(key,{name:dict(ncnn_px=[],joint_step_mm=[]) for name in arms})
                for name,arm in arms.items():
                    counts[name+'_accepted']+=bool(arm['fit']['native_active'] and arm['fit']['passes_native_reprojection_gate']);counts[name+'_same_as_native']+=arm['fit']==fit['fit']
                    group[name]['ncnn_px'].extend(v['median'] for v in arm['ncnn_disagreement_px'].values());points=np.asarray(arm['world_joints_mm']);previous=last.get((key,name))
                    if previous is not None and previous[2]==fit['generation'] and 0<row['stamp']-previous[0]<=100000000:group[name]['joint_step_mm'].append(float(np.linalg.norm(points-previous[1],axis=1).max()))
                    last[(key,name)]=(row['stamp'],points,fit['generation'])
                init=fit['candidate_arms']['both']['initialization']
                if init is not None:
                    counts['startup_'+init['stereo_status']]+=1
                    curls=np.asarray(init['fingers_observed_target_angles']);np.testing.assert_allclose(curls[:,2:].sum(1),curls[:,1],rtol=0,atol=1e-6);finger_targets.extend(curls[:,0].tolist())
                if fit['branch']=='fallback_carry' and arms['native']['ncnn_disagreement_px'] and arms['both']['ncnn_disagreement_px']:
                    means={name:float(np.mean([v['median'] for v in arm['ncnn_disagreement_px'].values()])) for name,arm in arms.items()}
                    events.append(dict(seconds=row['seconds'],stamp=row['stamp'],side=fit['side'],branch=fit['branch'],agreement_px=means,delta_px=means['both']-means['native']))
        assert next(candidate,None) is None
    assert counts['exact_original_controls']==counts['exact_stereo_controls']==4464
    for group in metrics.values():
        for arm in group.values():
            for key,values in arm.items():arm[key]=stats(values)
    result=dict(counts=counts,metrics=metrics,learned_startup_curl_sums=stats(finger_targets),largest_regressions=sorted(events,key=lambda r:r['delta_px'],reverse=True)[:8],largest_improvements=sorted(events,key=lambda r:r['delta_px'])[:8],frames_sha256=hashlib.sha256((a.result/'frames.jsonl').read_bytes()).hexdigest(),limitations='Frozen current observations and original reset/identity schedule. Scores are source-model agreement and movement proxies, not3Dtruth. Learned curl variance is not calibrated reliability. Extremes are selected for diagnostic raw review.')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(counts=counts,regression=result['largest_regressions'][:1],improvement=result['largest_improvements'][:1])))
    for key,group in metrics.items():
        if key.endswith('fallback_carry'):print(key,json.dumps(group))


if __name__=='__main__':main()
