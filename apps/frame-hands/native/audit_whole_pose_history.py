"""Validate reusable whole-pose history against frozen replay and synthetic sequences."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from audit_native_lengths import stats,sha

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('run','previous','native'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();data=np.loadtxt(a.run/'output.txt');old=np.loadtxt(a.previous/'output.txt')
 assert data.shape==old.shape==(2940,191)
 np.testing.assert_array_equal(data[:,:5],old[:,:5]);np.testing.assert_array_equal(data[:,6:],old[:,6:])
 x=np.loadtxt(a.run/'sequences.txt');assert x.shape==(960,12) and np.isfinite(x).all();assert np.all(x[:,7]<=x[:,6])
 names=['clean_fast_gestures','noisy_landmarks','flipped_initialization','loss_and_reacquisition'];metrics={}
 for mode,name in enumerate(names):
  group=x[(x[:,1]==mode)&(x[:,3]==1)]
  metrics[name]=dict(valid=len(group),eligible=int(group[:,4].sum()),selected=int(group[:,5].sum()),pose_rmse_mm=[stats(group[:,i]) for i in (8,9)],motion_delta_error_mm=[stats(group[:,i]) for i in (10,11)],worst_accuracy_regression_mm=float(max(group[:,9]-group[:,8])))
  if mode!=1:assert max(group[:,9])<1e-4
 for side in (0,1):
  group=x[(x[:,1]==3)&(x[:,0]==side)]
  for f in (45,46,47):assert group[f,3]==0
  for f in (48,75,76,80,90):assert group[f,4]==0
  for f in (49,77,81,91):assert group[f,4]==1
 sources=['whole_pose_fitter.h','whole_pose_seed.h','probe_whole_pose_seed.cpp','test_whole_pose_history.cpp','probe_whole_pose_sequence.cpp','audit_whole_pose_history.py','rig_stage.h','causal_rig_fit.h','fit_rig_pixels.h','fit_hand_rig.h','hand_rig.h','rig_bound_step.h']
 src=a.run/'source';src.mkdir(exist_ok=True);base=Path(__file__).parent
 for name in sources:shutil.copy2(base/name,src/name)
 report=dict(scope=__doc__,recorded_replay=dict(cases=len(data),exact_pre_refactor_decisions_and_geometry=True,full_refiner_host_us=stats(data[data[:,3]==1,5]),timing_scope='New timer includes validation and history persistence; not comparable with old seed/fit-only timing or native performance.'),
 sequences=metrics,history_controls=(a.run/'test-result.txt').read_text().strip(),
 source_sha256={n:sha(src/n) for n in sources},input_sha256={str(f):sha(f) for f in [a.previous/'heads.txt',a.previous/'output.txt',a.native/'rig-stage-v2/recorded-input.txt',a.native/'palm-profile-v1/side0.fhrig',a.native/'palm-profile-v1/side1.fhrig',a.native/'rig-pixels-viewer-v2/cameras.txt']},output_sha256={n:sha(a.run/n) for n in ['output.txt','sequences.txt']},executable_sha256={n:sha(a.run/n) for n in ['probe.exe','probe-aarch64','sequence.exe','sequence-aarch64','test.exe']},
 limitations=['Synthetic sequences share fitting anatomy, calibrated cameras and correct identity. No images or learned detector; no real-world accuracy guarantee.', 'Noisy sequence uses3mm independent 3D noise and2px independent image noise. Lower fit cost can worsen truth error.', 'History clears on loss; it does not reconstruct hidden hands or extend gesture confidence.', 'Actual recorded fits/decisions match prior reviewed evidence exactly. No new live camera/XR/input/driver changes. Latest143912Zholdout excluded.', 'ARM64 links only. Integrated capture-time HMD supply and native performance remain unverified.'])
 (a.run/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(recorded=report['recorded_replay'],sequences=metrics),indent=2))
if __name__=='__main__':main()
