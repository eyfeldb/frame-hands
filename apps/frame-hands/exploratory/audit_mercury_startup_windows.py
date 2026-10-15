"""Audit cross-window startup controls and paired cold/continuing metrics."""
import argparse,json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from audit_mercury_crop_fallback import stats


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--result',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();summary=json.loads((a.result/'summary.json').read_text());control=json.loads(Path(summary['protocol']['control']).read_text());controls={r['case']['name']:r for r in control['reports']};counts=Counter();metrics={};previous={};events=[]
    for row in summary['reports']:
        group=row['case']['name'].rsplit('-',1)[0];reference=controls[row['case']['name']];assert row['stamp']==reference['stamp'];counts['frames']+=1
        if 'skipped' in row:assert row['skipped']==reference['skipped'];counts['skipped_frames']+=1;continue
        for name,arm in reference['arms'].items():
            if 'fit' in arm:assert row['arms'][name]['fit']==arm['fit'];counts['exact_original_fits']+=1
        for source in ('off','on'):
            for mode in ('cold','compensated'):
                original=row['arms'][source+'-'+mode]
                if 'fit' not in original:continue
                for policy in ('original','stereo','curl','both'):
                    name=source+'-'+mode+('' if policy=='original' else '-'+policy);arm=row['arms'][name]
                    assert arm['reset']==original['reset'] and arm['reset_reason']==original['reset_reason']
                    key=f'{group}_{source}_{mode}_{policy}';data=metrics.setdefault(key,dict(ncnn_px=[],input_px=[],max_joint_step_mm=[],reset_ncnn_px=[],continuing_ncnn_px=[],fit_residual=[]))
                    values=[v['ncnn_disagreement_px']['median'] for v in arm['compare'].values()];data['ncnn_px'].extend(values);data['input_px'].extend(v['input_disagreement_px']['median'] for v in arm['compare'].values());data['fit_residual'].append(arm['fit']['reprojection_error'])
                    data['reset_ncnn_px' if arm['reset'] else 'continuing_ncnn_px'].extend(values);counts[policy+'_fits']+=1;counts[policy+'_accepted']+=bool(arm['fit']['native_active'] and arm['fit']['passes_native_reprojection_gate'])
                    if policy!='original':
                        init=arm['initialization'];assert (init is not None)==arm['reset']
                        if init is not None:
                            counts[policy+'_startup_'+init['stereo_status']]+=1
                            if policy in ('curl','both'):
                                fingers=np.asarray(init['fingers_observed_target_angles']);np.testing.assert_allclose(fingers[:,2:].sum(1),fingers[:,1],rtol=0,atol=1e-6)
                    points=np.asarray(arm['world_joints_mm']);old=previous.get(key)
                    if old is not None and old[2]==row['generation'] and 0<row['stamp']-old[0]<=100000000:data['max_joint_step_mm'].append(float(np.linalg.norm(points-old[1],axis=1).max()))
                    previous[key]=(row['stamp'],points,row['generation'])
                candidate=row['arms'][source+'-'+mode+'-both'];before=float(np.mean([v['ncnn_disagreement_px']['median'] for v in original['compare'].values()]));after=float(np.mean([v['ncnn_disagreement_px']['median'] for v in candidate['compare'].values()]))
                events.append(dict(case=row['case'],stamp=row['stamp'],seconds=row['seconds'],group=group,source=source,mode=mode,native_px=before,both_px=after,delta_px=after-before))
    assert counts['exact_original_fits']==680
    for group in metrics.values():
        for key,values in group.items():group[key]=stats(values)
    unconditioned=[r for r in events if r['source']=='off' and r['mode']=='compensated'];selected=[]
    for group in sorted({r['group'] for r in unconditioned}):selected.append(max((r for r in unconditioned if r['group']==group),key=lambda r:abs(r['delta_px'])))
    report=dict(counts=counts,metrics=metrics,selected_visual_review=selected,largest_regressions=sorted(events,key=lambda r:r['delta_px'],reverse=True)[:8],largest_improvements=sorted(events,key=lambda r:r['delta_px'])[:8],summary_sha256=hashlib.sha256((a.result/'summary.json').read_bytes()).hexdigest(),limitations='Pre-existing short windows, not newly held-out unseen data. OFF/ON learned observations were externally conditioned and remain frozen. Cold versus continuing fits separated; model agreement and movement proxies are not3Daccuracy or pure jitter. Visual extrema are diagnostic selection.')
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(counts=counts,selected=selected)))
    for key,data in metrics.items():
        if '_off_compensated_' in key and key.endswith(('original','both')):print(key,json.dumps(data))


if __name__=='__main__':main()
