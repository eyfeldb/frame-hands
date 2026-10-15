"""Synthetic shared-palm recovery and native-profile rigid-finger invariants."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE.parent/'exploratory'))
from palm_calibration import calibrate,PALM_RIG,NAMES
from frame_data import load_calibration
from stereo_uncertainty import sensitivity
from audit_native_lengths import sha

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('profile','profiles','reference','executable','persist','upstream','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();sys.path.insert(0,str(a.upstream));a.output.mkdir(parents=True,exist_ok=False);rest=np.loadtxt(a.profile,skiprows=21)[:,:3];cams=load_calibration(a.persist);rng=np.random.default_rng(50111);tests=[]
 forward=sum(cams[n].camera_to_world_xf[:3,2] for n in NAMES);forward/=np.linalg.norm(forward);origin=np.mean([cams[n].camera_to_world_xf[:3,3] for n in NAMES],0)
 for side,shape in [(0,[.82,1.08]),(1,[.91,.94])]:
  R=Rotation.random(8,random_state=rng).as_matrix();t=origin+forward*520+rng.uniform(-25,25,(8,3));X=rest[PALM_RIG].copy();X[:,0]*=shape[0]*(-1 if side else 1);X[:,2]*=shape[1];truth=np.einsum('nij,kj->nki',R,X)+t[:,None];uv=np.stack([cams[n].eye_to_window(cams[n].world_to_eye(truth.reshape(-1,3))).reshape(8,5,2) for n in NAMES],1)
  for noise in (0.,1.):
   measured=uv+rng.normal(0,noise,uv.shape);xyz=np.array([sensitivity(cams,[dict(camera=n,points=measured[i,v]) for v,n in enumerate(NAMES)])[0] for i in range(8)]);result=calibrate(rest,side,cams,measured,xyz)
   error=float(np.max(np.abs(np.array(result['shape'])-shape)))
   if not result['success'] or result['near_shape_bound'] or error>(.005 if noise==0 else .04):raise ValueError('Shared-shape recovery failed: '+str(result))
   tests.append(dict(side=side,truth=shape,pixel_noise_sigma=noise,estimated=result['shape'],maximum_scale_error=error,nfev=result['nfev']))
 poses=np.loadtxt(a.reference/'poses.txt');reference=np.load(a.reference/'reference.npz')['expected'];outputs={};max_finger_error=0.;wrong_side=[]
 groups=((0,6,7),(1,8,9,10),(2,11,12,13),(3,14,15,16),(4,17,18,19));parent_joints=np.loadtxt(a.profile,skiprows=1,max_rows=20)
 # Unchanged generic behavior after adding the experimental profile status.
 subprocess.run([str(a.executable.resolve()),str(a.profile.resolve()),str((a.reference/'poses.txt').resolve()),str((a.output/'generic.txt').resolve())],check=True,capture_output=True,text=True)
 generic=np.loadtxt(a.output/'generic.txt')[:,1:].reshape(-1,21,3);np.testing.assert_allclose(generic,reference,atol=1e-6,rtol=0)
 for side in (0,1):
  selected=poses[:,0]==side;source=a.output/f'side{side}.txt';np.savetxt(source,poses[selected],fmt='%.17g');profile=a.profiles/f'side{side}.fhrig';out=a.output/f'side{side}-output.txt'
  subprocess.run([str(a.executable.resolve()),str(profile.resolve()),str(source.resolve()),str(out.resolve())],check=True,capture_output=True,text=True)
  actual=np.loadtxt(out)[:,1:].reshape(-1,21,3);expected=reference[selected].copy();joints=np.loadtxt(profile,skiprows=1,max_rows=20);rot=Rotation.from_quat(poses[selected][:,[2,3,4,1]])
  for f,group in enumerate(groups):
   delta=joints[4*f,3:6]-parent_joints[4*f,3:6];delta[0]*=(-1 if side else 1);expected[:,group]+=rot.apply(np.tile(delta,(selected.sum(),1)))[:,None]
  valid=[i for group in groups for i in group]+[5];error=float(np.max(np.abs(actual[:,valid]-expected[:,valid])));max_finger_error=max(max_finger_error,error)
  if error>1e-6:raise ValueError('Shape changed finger articulation/length: '+str(error))
  bad=a.output/f'wrong-side{side}.txt';np.savetxt(bad,poses[~selected][:1],fmt='%.17g');r=subprocess.run([str(a.executable.resolve()),str(profile.resolve()),str(bad.resolve()),str((a.output/f'wrong-side{side}.out').resolve())],capture_output=True,text=True)
  if r.returncode==0:raise ValueError('Experimental profile accepted wrong hand')
  wrong_side.append(dict(profile_side=side,exit_code=r.returncode,message=r.stderr.strip()));outputs[side]=dict(poses=int(selected.sum()),profile_sha256=sha(profile))
 rejected=[]
 for name,side,pixels,xyz in [('too_few_frames',0,np.zeros((2,2,5,2)),np.zeros((2,5,3))),('wrong_side',2,np.zeros((3,2,5,2)),np.zeros((3,5,3))),('nonfinite_pixels',0,np.full((3,2,5,2),np.nan),np.zeros((3,5,3)))]:
  try:calibrate(rest,side,cams,pixels,xyz)
  except ValueError:rejected.append(name)
  else:raise ValueError('Invalid calibration accepted: '+name)
 report=dict(scope=__doc__,synthetic_tests=tests,generic_reference_max_difference_mm=float(np.max(np.abs(generic-reference))),profile_tests=outputs,maximum_finger_rigid_offset_error_mm=max_finger_error,wrong_side_rejected=wrong_side,invalid_calibration_rejected=rejected,provenance=dict(executable_sha256=sha(a.executable),source_sha256={n:sha(BASE/n) for n in ('palm_calibration.py','export_palm_profile.py','test_palm_calibration.py','hand_rig.h')}),limitations=['Synthetic anatomy and model agree; no biological calibration proof.','Wrist/MCP shape only, no independent finger or mesh calibration.','Real profile fixture remains explicitly experimental.'])
 (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
