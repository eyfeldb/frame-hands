"""Reproduce the combined native fitting stage against the frozen separated workflow."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from audit_causal_rig import sha,stats


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('native','executable','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();base=a.native;a.output.mkdir(parents=True,exist_ok=False)
 source=base/'palm-profile-viewer-v2';records=[json.loads(x) for x in (source/'poses.jsonl').read_text().splitlines()]
 sides=np.array([r['side'] for r in records]);initial=np.zeros((len(records),241));pixels=np.zeros((len(records),199))
 for side in (0,1):
  ids=np.flatnonzero(sides==side);initial[ids]=np.loadtxt(source/f'side{side}-3d-input.txt');pixels[ids]=np.loadtxt(source/f'side{side}-pixel-input.txt')
 points=initial[:,31:].reshape(-1,21,10)[:,:,:4].reshape(-1,84)
 rows=np.column_stack([[r['raw_ns'] for r in records],[r['source_id'] for r in records],sides,points,pixels[:,31:]])
 files=[base/'rig-generic-v1/generic.fhrig',base/'palm-profile-v1/side0.fhrig',base/'palm-profile-v1/side1.fhrig',base/'rig-pixels-viewer-v2/cameras.txt']
 def run(name,data,code=0):
  inp=a.output/(name+'-input.txt');out=a.output/(name+'-output.txt');np.savetxt(inp,data,fmt='%.17g')
  result=subprocess.run([str(a.executable.resolve()),*[str(f.resolve()) for f in files],str(inp.resolve()),str(out.resolve())],capture_output=True,text=True,timeout=90)
  if result.returncode!=code:raise RuntimeError(name+': '+result.stderr)
  if code:return
  values=np.loadtxt(out,ndmin=2)
  if values.shape!=(len(data),104) or not np.isfinite(values).all():raise ValueError('Invalid output '+name)
  return values
 result=run('recorded',rows);control=np.loadtxt(base/'causal-rig-viewer-v2/causal-output.txt')
 if np.any(result[:,0]!=1):raise ValueError('Lost initialized hand')
 seed_translation=np.linalg.norm(result[:,11:14]-initial[:,5:8],axis=1)
 seed_rotation=(Rotation.from_quat(result[:,7:11][:,[1,2,3,0]])*Rotation.from_quat(initial[:,1:5][:,[1,2,3,0]]).inv()).magnitude()
 if seed_translation.max()>1e-6 or seed_rotation.max()>1e-8:raise ValueError('Wrist initialization mismatch')
 displacement=np.linalg.norm((result[:,41:]-control[:,36:99]).reshape(-1,21,3),axis=2)
 if displacement.max()>.05:raise ValueError('Combined fit differs from separated control')
 indices=np.r_[0,2:104];prefix=run('prefix',rows[:128]);np.testing.assert_array_equal(prefix[:,indices],result[:128,indices])
 single=rows[np.flatnonzero(sides==0)[:3]].copy();single[:,0]=[1000000000,1033333333,1066666666];single[:,1]=5
 controls=[]
 for name in ('missing','missing_palm','degenerate_palm'):
  altered=single.copy()
  if name=='missing':altered[1,3:87:4]=0;altered[1,87::4]=0
  elif name=='missing_palm':altered[1,3+5*4]=0
  else:
   for j in (5,8,11,14,17):altered[1,4+j*4:7+j*4]=0
  out=run(name,altered);fresh=run(name+'-fresh',altered[2:])
  if out[1,0]!=0:raise ValueError('Invalid observations fitted')
  np.testing.assert_array_equal(out[2,indices],fresh[0,indices]);controls.append(name)
 for name in ('identity','gap'):
  altered=single[:2].copy()
  if name=='identity':altered[1,1]=6
  else:altered[1,0]=1200000000
  out=run(name,altered);fresh=run(name+'-fresh',altered[1:]);np.testing.assert_array_equal(out[1,indices],fresh[0,indices]);controls.append(name)
 # An unobserved palm-center dummy must not influence the fit.
 dirty=rows[:128].copy();dirty[:,4+20*4:7+20*4]=1e9
 np.testing.assert_array_equal(run('hidden_dummy',dirty)[:,indices],prefix[:,indices]);controls.append('hidden_dummy')
 for name in ('duplicate','negative_source','invalid_side'):
  altered=single[:2].copy()
  if name=='duplicate':altered[1,0]=altered[0,0]
  elif name=='negative_source':altered[1,1]=-1
  else:altered[1,2]=2
  run(name,altered,4 if name=='invalid_side' else 7);controls.append(name)
 for side in (0,1):
  limits=np.loadtxt(files[1+side],skiprows=1,max_rows=20)[:,6:8];angles=result[sides==side,21:41]
  if not np.all((angles>=limits[:,0]-1e-8)&(angles<=limits[:,1]+1e-8)):raise ValueError('Joint limits violated')
 report=dict(scope=__doc__,cases=len(rows),seed_translation_max_mm=float(seed_translation.max()),seed_rotation_max_rad=float(seed_rotation.max()),max_separated_workflow_landmark_difference_mm=float(displacement.max()),host_combined_stage_us=stats(result[:,1]),history_choices_different=int(np.sum(result[:,5]!=control[:,100])),controls=controls,prefix_exact=True,provenance=dict(executable_sha256=sha(a.executable),files={str(f):sha(f) for f in files},input_sha256=sha(a.output/'recorded-input.txt'),control_sha256=sha(base/'causal-rig-viewer-v2/causal-output.txt'),sources={n:sha(Path(__file__).parent/n) for n in ('rig_stage.h','replay_rig_stage.cpp','audit_rig_stage.py','causal_rig_fit.h','fit_hand_rig.h','fit_rig_pixels.h','hand_rig.h')}),limitations=['Frozen original detector stereo points/pixels and identities; no new inference, monocular pose path or per-joint visibility.','Generic-reference rigid wrist initialization intentionally matches frozen experiment; side-specific anatomy used in both fitting stages.','No input hold, output policy, rendering or live integration. Host timings exclude capture/detector/triangulation.','Numerical parity to experimental rig fits does not establish physical accuracy. Latest accepted holdout unused.'])
 (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
