"""Paired fixed-palm-profile replay, with later frames excluded from shape calibration."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE.parent/'exploratory'))
from frame_data import load_calibration,read_audit
from compare_replay import COMMON
from audit_native_lengths import sha,stats

def execute(executable,args):subprocess.run([str(executable.resolve()),*[str(x.resolve()) for x in args]],check=True,capture_output=True,text=True,timeout=90)
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('session','upstream','generic','profiles','calibration','control','initializer','fitter','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();sys.path.insert(0,str(a.upstream));a.output.mkdir(parents=True,exist_ok=False)
 if '143912Z' in str(a.session):raise ValueError('Accepted holdout excluded')
 cal=json.loads(a.calibration.read_text());manifest=json.loads((a.profiles/'manifest.json').read_text())
 if cal['session']!=a.session.name or manifest['calibration_sha256']!=sha(a.calibration):raise ValueError('Profile calibration mismatch')
 records=[json.loads(x) for x in (a.control/'isotropic-poses.jsonl').read_text().splitlines()];initial_input=np.loadtxt(a.control/'isotropic3d-input.txt');pixel_input=np.loadtxt(a.control/'isotropic-input.txt');baseline=np.loadtxt(a.control/'isotropic-fitted.txt');candidate=np.zeros_like(baseline);times=np.zeros(len(records));side_column=initial_input[:,0]
 if len(records)!=len(baseline) or len(initial_input)!=len(records) or len(pixel_input)!=len(records):raise ValueError('Incomplete control')
 for side in (0,1):
  profile=a.profiles/f'side{side}.fhrig'
  if sha(profile)!=manifest['profiles'][str(side)]['profile_sha256']:raise ValueError('Changed profile')
  ids=np.flatnonzero(side_column==side);inp=a.output/f'side{side}-3d-input.txt';np.savetxt(inp,initial_input[ids],fmt='%.17g');initpath=a.output/f'side{side}-3d-fit.txt';execute(a.initializer,[profile,inp,initpath]);init=np.loadtxt(initpath);px=pixel_input[ids].copy();px[:,:28]=init[:,8:36];inp=a.output/f'side{side}-pixel-input.txt';np.savetxt(inp,px,fmt='%.17g');output=a.output/f'side{side}-pixel-fit.txt';execute(a.fitter,[profile,a.control/'cameras.txt',inp,output]);out=np.loadtxt(output)
  if init.shape!=(len(ids),99) or out.shape!=init.shape or not np.isfinite(out).all() or np.any(out[:,0]!=1) or np.any(out[:,8]!=side):raise ValueError('Invalid fit')
  candidate[ids]=out;times[ids]=out[:,7]+init[:,7]
 # Loader change must preserve generic control behavior, beyond FK-only checks.
 checkids=np.linspace(0,len(records)-1,32,dtype=int);np.savetxt(a.output/'generic-input.txt',pixel_input[checkids],fmt='%.17g');execute(a.fitter,[a.generic,a.control/'cameras.txt',a.output/'generic-input.txt',a.output/'generic-output.txt']);check=np.loadtxt(a.output/'generic-output.txt');np.testing.assert_allclose(check[:,8:],baseline[checkids,8:],atol=1e-8,rtol=0)
 audit=read_audit(a.session/'frame/live-views.txt');cams=load_calibration(a.session/'frame/snapshot/device/persist');first=min(audit);boundary=cal['selection_parameters']['before_seconds'];ri,mi=np.array(COMMON).T;metrics={};counts={}
 def group(name):
  if name not in metrics:metrics[name]={branch:{kind:[] for kind in ('wrist_px','knuckle_px','tip_px','all_px','max_correction_mm')} for branch in ('baseline','candidate')};counts[name]=dict(hands=0,wrist_improved_over_5px=0,wrist_worse_over_5px=0,baseline_guard_reject=0,candidate_guard_reject=0)
 with (a.output/'poses.jsonl').open('x') as output:
  for i,row in enumerate(records):
   seconds=(row['raw_ns']-first)/1e9;side=row['side'];phase='later' if seconds>=boundary else 'early';names=[phase,phase+f'_side{side}'];raw=np.array(row['raw_head_mm']);views=[v for v in audit[row['raw_ns']]['views'] if v['id']==row['source_id']];es={};corr={}
   for branch,points in [('raw',raw[mi]),('baseline',baseline[i,36:].reshape(21,3)[ri]),('candidate',candidate[i,36:].reshape(21,3)[ri])]:
    values=[]
    for v in views:
     c=cams[v['camera']];eye=c.world_to_eye(points)
     if np.any(eye[:,2]<=0):raise ValueError('Behind camera output')
     values.append(np.linalg.norm(c.eye_to_window(eye)-v['points'][mi],axis=1))
    es[branch]=np.array(values);corr[branch]=float(np.max(np.linalg.norm(points-raw[mi],axis=1)))
   for name in names:
    group(name);counts[name]['hands']+=1
    for branch in ('baseline','candidate'):
     e=es[branch];values=dict(wrist_px=float(e[:,ri==5].mean()),knuckle_px=float(e[:,np.isin(ri,[8,11,14,17])].mean()),tip_px=float(e[:,ri<5].mean()),all_px=float(e.mean()),max_correction_mm=corr[branch])
     for key,value in values.items():metrics[name][branch][key].append(value)
     counts[name][branch+'_guard_reject']+=int(not(np.all(e.mean(1)<=es['raw'].mean(1)+2) and corr[branch]<=50))
    change=float(es['candidate'][:,ri==5].mean()-es['baseline'][:,ri==5].mean());counts[name]['wrist_improved_over_5px']+=change< -5;counts[name]['wrist_worse_over_5px']+=change>5
   output.write(json.dumps(dict(raw_ns=row['raw_ns'],source_id=row['source_id'],side=side,raw_head_mm=raw.tolist(),fit_rig_head_mm=candidate[i,36:].reshape(21,3).tolist(),tip_reprojection_change_px=float(es['candidate'][:,ri<5].mean()-es['raw'][:,ri<5].mean()),max_correction_mm=corr['candidate']))+'\n')
 report=dict(scope=__doc__,session=a.session.name,metrics={phase:{branch:{key:stats(value) for key,value in m.items() if value} for branch,m in branches.items()} for phase,branches in metrics.items()},counts=counts,host_combined_fit_us=stats(times),generic_control_max_difference_mm=float(np.max(np.abs(check[:,36:]-baseline[checkids,36:]))),provenance=dict(calibration_sha256=sha(a.calibration),profile_manifest_sha256=sha(a.profiles/'manifest.json'),control_summary_sha256=sha(a.control/'summary.json'),initializer_sha256=sha(a.initializer),fitter_sha256=sha(a.fitter),source_sha256=sha(Path(__file__))),limitations=['Later frames excluded from profile fitting but same recording/person; no independent camera labels.','Only palm length/width change; fixed finger lengths, detector, source identity and observation inputs retained.','No live integration, temporal seeds, uncertainty acceptance or visibility changes.','Experimental profile has not been accepted by wearer.'])
 (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(later=report['metrics'].get('later'),counts=counts.get('later')),indent=2))
if __name__=='__main__':main()
