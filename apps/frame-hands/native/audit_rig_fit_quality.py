"""Compare native per-part quality to independent saved-camera reprojection."""
import argparse,bisect,json,subprocess,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE.parent/'exploratory'))
from frame_data import load_calibration
from audit_native_lengths import sha

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('native','run','session','upstream'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args()
 if '143912Z' in str(a.session):raise ValueError('Holdout excluded')
 sys.path.insert(0,str(a.upstream));z=np.loadtxt(a.native/'whole-pose-history-v1/output.txt');inp=np.loadtxt(a.native/'rig-stage-v2/recorded-input.txt');obs=inp[:,87:].reshape(-1,21,2,4);points=z[:,128:191].reshape(-1,21,3)
 cams=load_calibration(a.session/'frame/snapshot/device/persist');es=[]
 for vi,name in enumerate(('slam_left','slam_right')):
  c=cams[name];uv=c.eye_to_window(c.world_to_eye(points.reshape(-1,3))).reshape(-1,21,2);es.append(np.sum((uv-obs[:,:,vi,1:3])**2,axis=2))
 es=np.stack(es,axis=2);mask=obs[:,:,:,0]>0;parts={};counts={}
 for name,indices in [('all',list(range(21))),('palm',[5,8,11,14,17]),('fingers',[i for i in range(21) if i not in (5,8,11,14,17)])]:
  counts[name]=mask[:,indices].sum((1,2));parts[name]=np.sqrt((es[:,indices]*mask[:,indices]).sum((1,2))/counts[name])
 np.testing.assert_allclose(parts['all'],z[:,100],rtol=1e-5,atol=1e-4)
 native_input=np.column_stack((z[:,2],z[:,101:128],inp[:,87:]));np.savetxt(a.run/'quality-input.txt',native_input,fmt='%.17g')
 exe=(a.run/'probe-quality.exe').resolve();output=a.run/'quality-output.txt'
 subprocess.run([str(exe),str((a.native/'palm-profile-v1/side0.fhrig').resolve()),str((a.native/'palm-profile-v1/side1.fhrig').resolve()),str((a.native/'rig-pixels-viewer-v2/cameras.txt').resolve()),str((a.run/'quality-input.txt').resolve()),str(output.resolve())],check=True)
 got=np.loadtxt(output);assert got.shape==(2940,10) and np.isfinite(got).all() and np.all(got[:,0]==1)
 errors=[]
 for offset,name in ((1,'all'),(4,'palm'),(7,'fingers')):
  np.testing.assert_array_equal(got[:,offset],counts[name]);np.testing.assert_allclose(got[:,offset+1],parts[name],rtol=1e-5,atol=1e-4);np.testing.assert_allclose(got[:,offset+2],got[:,offset+1]/3,rtol=1e-10,atol=1e-10);errors.append(float(np.max(abs(got[:,offset+1]-parts[name]))))
 poor=parts['all']>9;report=dict(hands=len(z),poor_whole=int(poor.sum()),poor_whole_but_palm_under9=int((poor&(parts['palm']<=9)).sum()),max_cpp_python_error_px=errors,scope='Agreement with the same detector observations; low palm error is not independent proof of tracking correctness.')
 analysis=json.loads((a.run/'analysis.json').read_text());events=json.loads((a.run/'events.json').read_text())['events'];times=[e['now'] for e in events]
 for r in analysis['review']:
  event=events[bisect.bisect_right(times,r['display_time'])-1];matches=[h for h in event['hands'] if h['side']==r['side']]
  if matches:r['part_rmse_px']={k:float(v[matches[0]['fit_index']]) for k,v in parts.items()}
 report['review']=analysis['review'];report['source_sha256']={f:sha(BASE/f) for f in ('rig_fit_quality.h','probe_rig_fit_quality.cpp','test_rig_fit_quality.cpp','audit_rig_fit_quality.py')};report['input_sha256']={str(f):sha(f) for f in [a.native/'whole-pose-history-v1/output.txt',a.native/'rig-stage-v2/recorded-input.txt',a.native/'rig-pixels-viewer-v2/cameras.txt',a.native/'palm-profile-v1/side0.fhrig',a.native/'palm-profile-v1/side1.fhrig']};report['probe_sha256']=sha(exe);report['output_sha256']=sha(output)
 (a.run/'quality-parts.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');np.savez(a.run/'quality-parts.npz',**parts);print(json.dumps({k:v for k,v in report.items() if k!='review' and not k.endswith('sha256')},indent=2))
if __name__=='__main__':main()
