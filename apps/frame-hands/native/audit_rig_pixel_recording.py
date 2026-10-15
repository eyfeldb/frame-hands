"""Matched direct-pixel and 3D-then-pixel fits on frozen recorded observations."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE.parent/'exploratory'))
from frame_data import read_audit,load_calibration
from rig_pixel_data import write_cameras,pixel_row,NAMES
from compare_replay import COMMON
from audit_native_lengths import sha,stats

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('session','upstream','profile','executable','control','three-d-executable','output'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();sys.path.insert(0,str(a.upstream));a.output.mkdir(parents=True,exist_ok=False)
 if '143912Z' in str(a.session):raise ValueError('Holdout excluded')
 control_meta=json.loads((a.control/'summary.json').read_text())
 if control_meta['session']!=a.session.name or control_meta['provenance']['profile_sha256']!=sha(a.profile):raise ValueError('Mismatched control')
 for n,h in control_meta['provenance']['inputs'].items():
  if sha(a.session/n)!=h:raise ValueError('Recording changed')
 control=np.loadtxt(a.control/'fitted.txt');original_input=np.loadtxt(a.control/'observations.txt');seeds=original_input[:,:31];records=[json.loads(x) for x in (a.control/'poses.jsonl').read_text().splitlines()]
 if len(control)!=len(records) or len(seeds)!=len(records):raise ValueError('Incomplete control')
 isotropic_input=original_input.copy();isotropic_input[:,31:].reshape(-1,21,10)[:,:,4:]=[3,0,3,0,0,3]
 np.savetxt(a.output/'isotropic3d-input.txt',isotropic_input,fmt='%.17g')
 subprocess.run([str(a.three_d_executable.resolve()),str(a.profile.resolve()),str((a.output/'isotropic3d-input.txt').resolve()),str((a.output/'isotropic3d-fit.txt').resolve())],check=True,timeout=90)
 isotropic=np.loadtxt(a.output/'isotropic3d-fit.txt')
 if isotropic.shape!=control.shape or not np.isfinite(isotropic).all():raise ValueError('Incomplete isotropic initializer')
 cams=load_calibration(a.session/'frame/snapshot/device/persist');write_cameras(cams,a.output/'cameras.txt');audits=read_audit(a.session/'frame/live-views.txt');ri,mi=np.array(COMMON).T;rows={'direct':[],'staged':[],'isotropic':[]};views=[]
 for i,r in enumerate(records):
  source=[v for v in audits[r['raw_ns']]['views'] if v['id']==r['source_id']];uv=np.zeros((2,21,2));mask=np.zeros((2,21),bool)
  if len(source)!=2 or set(v['camera'] for v in source)!=set(NAMES) or seeds[i,0]!=r['side'] or control[i,8]!=r['side']:raise ValueError('Source join changed')
  for v,n in enumerate(NAMES):uv[v,ri]=next(x for x in source if x['camera']==n)['points'][mi];mask[v,ri]=True
  rows['direct'].append(pixel_row(seeds[i,:28],uv,mask));rows['staged'].append(pixel_row(control[i,8:36],uv,mask));rows['isotropic'].append(pixel_row(isotropic[i,8:36],uv,mask));views.append(uv)
 outputs={}
 for name in rows:
  np.savetxt(a.output/(name+'-input.txt'),rows[name],fmt='%.17g');subprocess.run([str(a.executable.resolve()),str(a.profile.resolve()),str((a.output/'cameras.txt').resolve()),str((a.output/(name+'-input.txt')).resolve()),str((a.output/(name+'-fitted.txt')).resolve())],check=True,timeout=90)
  outputs[name]=np.loadtxt(a.output/(name+'-fitted.txt'))
  if outputs[name].shape!=control.shape or not np.isfinite(outputs[name]).all() or np.any(outputs[name][:,5]>outputs[name][:,4]+1e-9):raise ValueError('Invalid candidate')
 metrics={name:{k:[] for k in ('tip_reprojection_px','point_reprojection_px','max_correction_mm','pose_fit_us')} for name in ('raw','control','direct','staged','isotropic')};counts={name:dict(worse_over_5px_vs_raw=0,worse_over_5px_vs_control=0,better_over_5px_vs_control=0,previous_guard_reject=0,behind_camera=0) for name in ('control','direct','staged','isotropic')}
 handles={name:(a.output/(name+'-poses.jsonl')).open('x') for name in ('direct','staged','isotropic')}
 try:
  for i,r in enumerate(records):
   raw=np.array(r['raw_head_mm']);points={'raw':raw[mi],'control':control[i,36:].reshape(21,3)[ri],**{name:out[i,36:].reshape(21,3)[ri] for name,out in outputs.items()}};errors={}
   for name,x in points.items():
    es=[]
    for v,n in enumerate(NAMES):
     eye=cams[n].world_to_eye(x)
     if np.any(eye[:,2]<=0):break
     es.append(np.linalg.norm(cams[n].eye_to_window(eye)-views[i][v,ri],axis=1))
    if len(es)!=2:
     if name!='raw':counts[name]['behind_camera']+=1
     continue
    es=np.array(es);errors[name]=es;metrics[name]['tip_reprojection_px'].append(float(es[:,ri<5].mean()));metrics[name]['point_reprojection_px'].append(float(es.mean()));metrics[name]['max_correction_mm'].append(float(np.max(np.linalg.norm(x-raw[mi],axis=1))))
    if name=='control':metrics[name]['pose_fit_us'].append(float(control[i,7]))
    elif name in outputs:metrics[name]['pose_fit_us'].append(float(outputs[name][i,7]+(control[i,7] if name=='staged' else isotropic[i,7] if name=='isotropic' else 0)))
   for name in ('control','direct','staged','isotropic'):
    if name not in errors:continue
    tip=float(errors[name][:,ri<5].mean());rawtip=float(errors['raw'][:,ri<5].mean());controltip=float(errors['control'][:,ri<5].mean());counts[name]['worse_over_5px_vs_raw']+=tip-rawtip>5;counts[name]['worse_over_5px_vs_control']+=tip-controltip>5;counts[name]['better_over_5px_vs_control']+=tip-controltip< -5;counts[name]['previous_guard_reject']+=not(np.all(errors[name].mean(1)<=errors['raw'].mean(1)+2) and metrics[name]['max_correction_mm'][-1]<=50)
    if name in handles:handles[name].write(json.dumps(dict(raw_ns=r['raw_ns'],source_id=r['source_id'],side=r['side'],raw_head_mm=raw.tolist(),fit_rig_head_mm=outputs[name][i,36:].reshape(21,3).tolist(),tip_reprojection_change_px=tip-rawtip,max_correction_mm=metrics[name]['max_correction_mm'][-1]))+'\n')
 finally:
  for f in handles.values():f.close()
 report=dict(scope=__doc__,session=a.session.name,hands=len(records),metrics={name:{key:stats(vals) for key,vals in m.items()} for name,m in metrics.items()},counts={n:{k:int(v) for k,v in c.items()} for n,c in counts.items()},provenance=dict(control_summary_sha256=sha(a.control/'summary.json'),control_input_sha256=sha(a.control/'observations.txt'),control_output_sha256=sha(a.control/'fitted.txt'),profile_sha256=sha(a.profile),camera_sha256=sha(a.output/'cameras.txt'),executable_sha256=sha(a.executable),initializer_sha256=sha(a.three_d_executable),sources={n:sha(BASE/n) for n in ('fit_rig_pixels.h','evaluate_rig_pixels.cpp','rig_pixel_data.py','audit_rig_pixel_recording.py')}),limitations=['Same image observations and generic anatomy; reprojection is training residual, not physical accuracy.','Direct uses frozen original seed. Staged starts from prior covariance-weighted3D fit. Isotropic uses uniform3mm3D residuals only for initialization, then identical pixel refinement.','Both staged variants include their3D initialization host cost.','Both views grouped per landmark Huber; noise assumes3px independent isotropic image error.','30iterations per stage: staged uses twice the maximum fitting work.','No confidence, identity, calibration, temporal state, native runtime or live acceptance.'])
 (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
