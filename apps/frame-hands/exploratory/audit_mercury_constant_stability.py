"""Paired constant-finger-stability metrics with exact original-control checks."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def stats(values):
    if not values:return None
    return dict(count=len(values),median=float(np.median(values)),p95=float(np.percentile(values,95)),maximum=float(max(values)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('long','long_control','windows','window_control','output'):p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    a=p.parse_args();groups={};previous={};events=[];exact_long=0;exact_windows=0
    def add(group,stamp,seconds,side,native,constant,reset):
        pair={'native':native,'constant':constant};means={}
        for name,fit in pair.items():
            key=group+'_'+name;m=groups.setdefault(key,dict(frames=0,ncnn_px=[],joint_step_mm=[],angle_step_rad=[],wrist_step_mm=[],joint_step_no_reset_mm=[],fit_error=[]));m['frames']+=1
            values=[v['median'] for v in fit['ncnn_disagreement_px'].values()];m['ncnn_px'].extend(values);means[name]=float(np.mean(values)) if values else None
            m['fit_error'].append(fit['fit']['reprojection_error']);joints=np.asarray(fit['world_joints_mm']);angles=np.asarray(fit['finger_angles']);old=previous.get(key)
            if old is not None and 0<stamp-old[0]<=100000000:
                delta=np.linalg.norm(joints-old[1],axis=1);m['joint_step_mm'].append(float(delta.max()));m['wrist_step_mm'].append(float(delta[0]));m['angle_step_rad'].append(float(abs(angles-old[2]).max()))
                if not reset:m['joint_step_no_reset_mm'].append(float(delta.max()))
            previous[key]=(stamp,joints,angles)
        if all(v is not None for v in means.values()):events.append(dict(group=group,seconds=seconds,stamp=stamp,side=side,ncnn_change_px=means['constant']-means['native'],native_ncnn_px=means['native'],constant_ncnn_px=means['constant']))
    with (a.long/'frames.jsonl').open() as f,(a.long_control/'frames.jsonl').open() as old:
        for line in f:
            row=json.loads(line);baseline=json.loads(next(old));assert row['stamp']==baseline['stamp']
            lookup={(v['branch'],v['side']):v for v in row['fits']}
            for fit in baseline['fits']:
                assert lookup[(fit['branch'],fit['side'])]['fit']==fit['fit'];exact_long+=1
            for side in (0,1):
                for native,candidate in [('box','box_constant'),('fallback_carry','fallback_constant')]:
                    original=lookup.get((native,side));changed=lookup.get((candidate,side));assert (original is None)==(changed is None)
                    if original is None:continue
                    group='long_'+('prefix' if row['seconds']<18 else 'evaluation')+'_side'+str(side)+'_'+native
                    add(group,row['stamp'],row['seconds'],side,original,changed,original['reset_reason'] is not None)
    windows=json.loads((a.windows/'summary.json').read_text());control=json.loads((a.window_control/'summary.json').read_text());old={v['case']['name']:v for v in control['reports']}
    for row in windows['reports']:
        original=old[row['case']['name']]
        for name,fit in original['arms'].items():
            if 'fit' in fit:assert fit['fit']==row['arms'][name]['fit'];exact_windows+=1
        for source in ('off','on'):
            native=row['arms'].get(source+'-compensated');constant=row['arms'].get(source+'-constant')
            if native is None or 'fit' not in native:continue
            def convert(f):return dict(fit=f['fit'],world_joints_mm=f['world_joints_mm'],finger_angles=f['finger_angles'],ncnn_disagreement_px={k:v['ncnn_disagreement_px'] for k,v in f['compare'].items()})
            group=row['case']['name'].rsplit('-',1)[0]+'_'+source
            add(group,row['stamp'],row['seconds'],row['case']['side'],convert(native),convert(constant),native['reset'])
    for group in groups.values():
        for key,values in list(group.items()):
            if key!='frames':group[key]=stats(values)
    out=dict(exact_long_controls=exact_long,exact_window_controls=exact_windows,groups=groups,largest_image_agreement_regressions=sorted(events,key=lambda r:r['ncnn_change_px'],reverse=True)[:16],largest_image_agreement_improvements=sorted(events,key=lambda r:r['ncnn_change_px'])[:16],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),limitations='Same-frame estimator agreement and motion are proxies. Full trajectory and short viewer window provide no calibrated gesture latency or3Dtruth. Counts include repeated branches/conditions; do not treat as independent trials. Smaller steps can preserve wrong poses or lag actual movement; inspect both regression and improvement frames.')
    a.output.write_text(json.dumps(out,indent=2)+'\n');print('exact controls',exact_long,exact_windows)
    for name,g in groups.items():print(name,'motion',round(g['joint_step_mm']['p95'],2) if g['joint_step_mm'] else None,'angle',round(g['angle_step_rad']['p95'],3) if g['angle_step_rad'] else None,'ncnn',round(g['ncnn_px']['p95'],2))
    print('largest regressions',json.dumps(out['largest_image_agreement_regressions'][:5]))


if __name__=='__main__':main()
