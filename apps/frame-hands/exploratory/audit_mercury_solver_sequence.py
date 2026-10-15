"""Audit paired solver replay and synthetic camera-motion controls."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def stats(values):
    a=np.asarray(values);return dict(median=float(np.median(a)),p95=float(np.percentile(a,95)),maximum=float(np.max(a)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('recorded','previous','synthetic','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();current=json.loads(a.recorded.read_text());old=json.loads(a.previous.read_text());synthetic=json.loads(a.synthetic.read_text());parity=0
    if len(current['reports'])!=len(old['reports']):raise AssertionError('Frame count changed')
    for r,prior in zip(current['reports'],old['reports']):
        if r['stamp']!=prior['stamp']:raise AssertionError('Frame order changed')
        for name,v in prior['arms'].items():
            new=r['arms'][name]
            for field in ('fit','world_joints_mm','projected','reset','reset_reason'):
                if v.get(field)!=new.get(field):raise AssertionError('Baseline branch changed: '+name+' '+field)
            parity+=1
    groups={}
    for group in sorted({r['case']['name'].rsplit('-',1)[0] for r in current['reports']}):
        frames=[r for r in current['reports'] if r['case']['name'].rsplit('-',1)[0]==group];arms={}
        for name in sorted({n for r in frames for n in r['arms']}):
            rows=[(r,r['arms'][name]) for r in frames if name in r['arms'] and 'fit' in r['arms'][name]];steps=[];timing=[];disagreement=[];previous=None
            for r,v in rows:
                if previous:
                    pr,pv=previous;dt=(r['stamp']-pr['stamp'])/1e9
                    if 0<dt<=.1 and r['generation']==pr['generation']:
                        steps.append(float(np.max(np.linalg.norm(np.asarray(v['world_joints_mm'])-pv['world_joints_mm'],axis=1))));timing.append(dt*1000)
                previous=(r,v);disagreement.extend(x['ncnn_disagreement_px']['median'] for x in v['compare'].values())
            arms[name]=dict(frames=len(rows),gate_pass=sum(v['fit']['passes_native_reprojection_gate'] for r,v in rows),resets=sum(v['reset'] for r,v in rows),max_joint_step_mm=stats(steps),step_intervals_ms=stats(timing),median_joint_ncnn_disagreement_px=stats(disagreement),native_error=stats([v['fit']['reprojection_error'] for r,v in rows]))
        groups[group]=dict(arms=arms,skips=[r['case']['name'] for r in frames if 'skipped' in r],max_head_step_mm=max(r['head_step']['translation_mm'] for r in frames if r['head_step']),max_head_step_deg=max(r['head_step']['rotation_deg'] for r in frames if r['head_step']))
    controls=[]
    for side in (0,1):
        for smoothing,transport in ((0,False),(2,False),(2,True)):
            static=next(r for r in synthetic['rows'] if r['side']==side and not r['moving'] and r['smoothing']==smoothing and r['transport']==transport)
            moving=next(r for r in synthetic['rows'] if r['side']==side and r['moving'] and r['smoothing']==smoothing and r['transport']==transport)
            if transport:
                plain=next(r for r in synthetic['rows'] if r['side']==side and not r['moving'] and r['smoothing']==smoothing and not r['transport'])
                if static['frames']!=plain['frames']:raise AssertionError('Identity transport changed stationary fit')
            drift=np.asarray([np.linalg.norm(np.asarray(x['estimated_world_m'])-y['estimated_world_m'],axis=1)*1000 for x,y in zip(moving['frames'],static['frames'])])
            controls.append(dict(side=side,smoothing=smoothing,transport=transport,motion_added_max_joint_drift_mm=stats(drift[30:91].max(1)),motion_added_wrist_drift_mm=stats(drift[30:91,0]),post_stop_added_max_joint_drift_mm=stats(drift[91:].max(1)),stationary_model_inverse_error_mm=static['moving_phase_max_joint_error_mm']))
    result=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inputs_sha256={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in [('recorded',a.recorded),('previous',a.previous),('synthetic',a.synthetic)]},exact_unchanged_baseline_fits=parity,recorded=groups,synthetic=controls,metric_limits='Recorded output steps include reacquisition steps with matched generation and elapsed time<=100ms; report their actual intervals. They are neither true motion nor rendered jitter. ncnn is another estimator. Synthetic drift subtracts a matched stationary-camera run; stationary inverse-fit bias is reported separately and remains substantial.')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print('Unchanged baseline fits:',parity)
    for c in controls:print(c['side'],c['smoothing'],c['transport'],c['motion_added_max_joint_drift_mm'])

if __name__=='__main__':main()
