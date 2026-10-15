"""Offline causal finger-length ablation using actual accepted C++ Stability.
No detector rerun, device access, wearer trial, new identity or visibility policy.
"""
import argparse, hashlib, json, platform, subprocess, sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'exploratory'))
from compare_replay import source_hands
from frame_data import capture_poses, read_audit, load_calibration
PARENTS=np.array([0,0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19])
FINGERS=np.array([2,3,4,6,7,8,10,11,12,14,15,16,18,19,20])
TIPS=np.array([4,8,12,16,20])
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def stats(v):return dict(zip(('p50','p95','max'),np.percentile(v,[50,95,100]).tolist())) if len(v) else None
def errors(points,views,cameras):
    out=[]
    for v in views:
        cam=cameras[v['camera']];eye=cam.world_to_eye(points)
        if np.any(eye[:,2]<=0):return None
        out.append(np.linalg.norm(cam.eye_to_window(eye)-v['points'],axis=1))
    return np.array(out) if out else None

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('session','upstream','executable','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.upstream));a.output.mkdir(parents=True,exist_ok=False)
    if '143912Z' in str(a.session):raise ValueError('Latest accepted holdout excluded')
    source=source_hands(a.session);poses=capture_poses(a.session)
    audits=read_audit(a.session/'frame/live-views.txt');cameras=load_calibration(a.session/'frame/snapshot/device/persist')
    items=[];skips=dict(missing_pose=0,duplicate_side=0);first=min(source)
    with (a.output/'input.txt').open('x') as out:
        for rawstamp,ob in sorted(audits.items()):
            mono=ob['mono_ns'];hands=source.get(mono,{})
            if mono not in poses:skips['missing_pose']+=len(hands);continue
            world=poses[mono]
            sides=[h['side'] for h in hands.values()]
            if len(sides)!=len(set(sides)):skips['duplicate_side']+=len(hands);continue
            for ident,h in sorted(hands.items(),key=lambda pair:pair[1]['side']):
                views=[v for v in ob['views'] if v['id']==ident and v['camera'] in cameras]
                err=errors(h['points'],views,cameras)
                good=bool(h['stereo'] and len(views)==2 and len({v['camera'] for v in views})==2 and min(v['presence'] for v in views)>=.8 and err is not None and np.median(err)<=8 and np.percentile(err,95)<=20)
                t=1000+(mono-first)//1_000_000
                points=(h['points']@world[:3,:3].T+world[:3,3])*.001
                row=[t,ident,h['side'],int(good),*points.reshape(-1)]
                out.write(' '.join(map(str,row))+'\n')
                items.append(dict(time=int(t),id=ident,side=h['side'],rawstamp=rawstamp,mono=mono,good=good,views=views,world=world,points=points))
    subprocess.run([str(a.executable.resolve()),str((a.output/'input.txt').resolve()),str((a.output/'output.txt').resolve())],check=True)
    rows=(a.output/'output.txt').read_text().splitlines()
    if len(rows)!=len(items):raise ValueError('Incomplete C++ output')
    values={k:[] for k in ('control_tip_error_px','candidate_tip_error_px','tip_error_change_px','coordinate_change_mm','control_length_step_mm','candidate_length_step_mm','control_point_step_mm','candidate_point_step_mm')}
    counts=dict(input=len(items),admitted_calibration=sum(r['good'] for r in items),accepted=0,fully_calibrated=0,changed=0,tip_error_worse_over_5px=0,tip_error_better_over_5px=0)
    previous={};review=[]
    with (a.output/'poses.jsonl').open('x') as out:
        for item,line in zip(items,rows):
            f=line.split();t,ident,side,accepted,applied,n=map(int,f[:6])
            if (t,ident,side)!=(item['time'],item['id'],item['side']):raise ValueError('C++ source join mismatch')
            if not accepted:continue
            counts['accepted']+=1;counts['fully_calibrated']+=n==15
            if len(f)!=132:raise ValueError('Invalid output geometry size')
            points=np.array(list(map(float,f[6:]))).reshape(2,21,3);world=item['world']
            head=(points*1000-world[:3,3])@world[:3,:3]
            delta=float(np.max(np.linalg.norm(points[1]-points[0],axis=1))*1000)
            values['coordinate_change_mm'].append(delta);counts['changed']+=delta>1e-6
            e=[errors(v,item['views'],cameras) for v in head];change=None
            if all(x is not None for x in e):
                means=[float(x[:,TIPS].mean()) for x in e];change=means[1]-means[0]
                for k,v in zip(('control_tip_error_px','candidate_tip_error_px','tip_error_change_px'),[*means,change]):values[k].append(v)
                counts['tip_error_worse_over_5px']+=change>5;counts['tip_error_better_over_5px']+=change< -5
            lengths=np.linalg.norm(points[:,FINGERS]-points[:,PARENTS[FINGERS]],axis=2)
            old=previous.get(side)
            if old and old['id']==ident and 0<t-old['time']<=80:
                for j,label in enumerate(('control','candidate')):
                    values[label+'_length_step_mm'].extend((np.abs(lengths[j]-old['lengths'][j])*1000).tolist())
                    values[label+'_point_step_mm'].append(float(np.max(np.linalg.norm(points[j]-old['points'][j],axis=1))*1000))
            previous[side]=dict(id=ident,time=t,points=points,lengths=lengths)
            row=dict(raw_ns=item['rawstamp'],mono_ns=item['mono'],side=side,source_id=ident,calibration_admitted=item['good'],fixed_lengths=n,applied=bool(applied),control_head_mm=head[0].tolist(),candidate_head_mm=head[1].tolist(),max_change_mm=delta,tip_error_change_px=change)
            out.write(json.dumps(row)+'\n');review.append(row)
    report=dict(scope=__doc__,counts=counts,skips=skips,metrics={k:stats(v) for k,v in values.items()},parameters=dict(samples=15,max_relative_MAD=.10,min_presence=.8,max_median_reprojection_px=8,max_p95_reprojection_px=20,reset_gap_ms=80),limitations=['Reprojection compares to the same detector observations; not physical accuracy.','Zero transport delay at capture timestamps; not actual driver playback.','No new identity, acquisition or dropout behavior. Ambiguous same-side source sets excluded.','First causal qualifying calibration window can still contain systematic error.','Frame timing and wearer quality not evaluated.'],provenance=dict(session=a.session.name,platform=platform.platform(),executable_sha256=sha(a.executable),inputs={n:sha(a.session/n) for n in ('relay-trace.bin','sampler-trace.bin','frame/live-views.txt')},sources={n:sha(BASE/n) for n in ('audit_native_lengths.py','stable_lengths.h','replay_native_lengths.cpp')},accepted_filter_sha256=sha(BASE.parent/'pc-input/stability.h')))
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    changed=[r for r in review if r['max_change_mm']>1e-6 and r['tip_error_change_px'] is not None]
    picks=sorted(changed,key=lambda r:r['tip_error_change_px'])
    (a.output/'review-selection.json').write_text(json.dumps((picks[:2]+picks[-2:]) if picks else [],indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
