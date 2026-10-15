"""Compare published-native, fresh-stereo and fitted-stereo through identical C++ Stability.
Saved poses only; ideal zero delivery delay. No hardware or user-acceptance claim.
"""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE.parent/'exploratory'))
from frame_data import read_audit,capture_poses,load_calibration
from compare_replay import source_hands
from audit_native_lengths import PARENTS,FINGERS,TIPS,errors,stats,sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('session','upstream','replay','executable','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.upstream));a.output.mkdir(parents=True,exist_ok=False)
    source=source_hands(a.session);poses=capture_poses(a.session);audit=read_audit(a.session/'frame/live-views.txt');cams=load_calibration(a.session/'frame/snapshot/device/persist')
    rows=[json.loads(v) for v in (a.replay/'poses.jsonl').read_text().splitlines()];first=min(source)
    joined=[];labels=('published','stereo','fitted');files=[(a.output/(k+'-input.txt')).open('x') for k in labels]
    try:
        for r in rows:
            ob=audit[r['raw_ns']];mono=ob['mono_ns'];ident=r['source_id']
            if mono not in poses or ident not in source.get(mono,{}):continue
            h=source[mono][ident];xf=poses[mono];t=1000+(mono-first)//1_000_000
            inputs=[h['points'],np.array(r['control_head_mm']),np.array(r['candidate_head_mm'])]
            for f,xyz in zip(files,inputs):
                world=(xyz@xf[:3,:3].T+xf[:3,3])*.001
                f.write(' '.join(map(str,[t,ident,h['side'],0,*world.reshape(-1)]))+'\n')
            joined.append(dict(stamp=r['raw_ns'],time=t,id=ident,side=h['side'],world=xf,views=[v for v in ob['views'] if v['id']==ident and v['camera'] in cams]))
    finally:
        for f in files:f.close()
    records={};reports={}
    for label in labels:
        input_path=a.output/(label+'-input.txt');output_path=a.output/(label+'-output.txt')
        subprocess.run([str(a.executable.resolve()),str(input_path.resolve()),str(output_path.resolve())],check=True)
        lines=output_path.read_text().splitlines()
        if len(lines)!=len(joined):raise ValueError('Incomplete output')
        values={k:[] for k in ('tip_error_px','length_step_mm','point_step_mm')};accepted=0;previous={};records[label]={}
        for r,line in zip(joined,lines):
            f=line.split();t,ident,side,ok=map(int,f[:4])
            if (t,ident,side)!=(r['time'],r['id'],r['side']):raise ValueError('Bad exact join')
            if not ok:continue
            accepted+=1;world=np.array(list(map(float,f[6:69]))).reshape(21,3)
            xyz=(world*1000-r['world'][:3,3])@r['world'][:3,:3]
            e=errors(xyz,r['views'],cams)
            if e is not None:values['tip_error_px'].append(float(e[:,TIPS].mean()))
            length=np.linalg.norm(world[FINGERS]-world[PARENTS[FINGERS]],axis=1)*1000
            old=previous.get(side)
            if old and old[0]==ident and 0<t-old[1]<=80:
                values['length_step_mm'].extend(np.abs(length-old[2]).tolist());values['point_step_mm'].append(float(np.max(np.linalg.norm(world-old[3],axis=1))*1000))
            previous[side]=(ident,t,length,world);records[label][(r['stamp'],ident)]=world
        reports[label]=dict(accepted=accepted,rejected=len(joined)-accepted,metrics={k:stats(v) for k,v in values.items()})
    paired=set(records['stereo'])&set(records['fitted']);delta=[float(np.max(np.linalg.norm(records['stereo'][k]-records['fitted'][k],axis=1))*1000) for k in paired]
    report=dict(scope=__doc__,session=a.session.name,input_hands=len(joined),branches=reports,paired_stereo_fitted=len(paired),fitted_newly_rejected=len(set(records['stereo'])-set(records['fitted'])),fitted_newly_accepted=len(set(records['fitted'])-set(records['stereo'])),paired_output_difference_mm=stats(delta),provenance=dict(executable_sha256=sha(a.executable),replay_sha256=sha(a.replay/'poses.jsonl'),source_sha256=sha(__file__)),limitations=['Only the subset with paired stereo reconstruction is compared.','Update acceptance is not rendered visibility; holds are not sampled.','Point-step includes genuine movement and is not a jitter/accuracy metric.','Current detector pixels are not independent anatomical labels.'])
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
