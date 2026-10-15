"""Paired finite/analytic projection derivatives with identical observations, seeds and solver budgets."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from audit_native_lengths import sha,stats
BASE=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('executable','generic-profile','synthetic','profiles','recording','cameras','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 cases=[dict(name='generic_synthetic',profile=a.generic_profile,input=a.synthetic/'staged-input.txt',reference=a.synthetic/'staged-fit.txt')]+[dict(name=f'palm_side{side}',profile=a.profiles/f'side{side}.fhrig',input=a.recording/f'side{side}-pixel-input.txt',reference=a.recording/f'side{side}-pixel-fit.txt') for side in (0,1)]
 results=[]
 for item in cases:
  root=a.output/item['name'];root.mkdir();outputs={}
  for mode in ('finite','analytic','hybrid'):
   subprocess.run([str(a.executable.resolve()),str(item['profile'].resolve()),str(a.cameras.resolve()),str(item['input'].resolve()),str((root/(mode+'.txt')).resolve()),mode],check=True,capture_output=True,text=True,timeout=90);outputs[mode]=np.loadtxt(root/(mode+'.txt'))
  old=np.loadtxt(item['reference']);finite=outputs['finite']
  for mode in ('analytic','hybrid'):
   candidate=outputs[mode]
   if finite.shape!=old.shape or candidate.shape!=old.shape or not np.isfinite(candidate).all():raise ValueError('Incomplete/nonfinite result')
   if np.any(candidate[:,:2]!=finite[:,:2]) or np.any(candidate[:,8]!=finite[:,8]) or np.any(candidate[:,5]>candidate[:,4]+1e-9):raise ValueError('Changed side, sufficiency or cost invariant')
   delta=np.max(np.linalg.norm((candidate[:,36:]-finite[:,36:]).reshape(-1,21,3),axis=2),axis=1);control_delta=float(np.max(np.abs(finite[:,36:]-old[:,36:])))
   entry=dict(name=item['name'],mode=mode,cases=len(old),frozen_control_max_coordinate_difference_mm=control_delta,maximum_hand_displacement_mm=stats(delta),hands_over_point05mm=int((delta>.05).sum()),cost_increase=stats(candidate[:,5]-finite[:,5]),finite_us=stats(finite[:,7]),candidate_us=stats(candidate[:,7]),largest_change_case=int(np.argmax(delta)),provenance=dict(profile_sha256=sha(item['profile']),input_sha256=sha(item['input']),reference_sha256=sha(item['reference'])))
   results.append(entry)
 report=dict(scope=__doc__,results=results,provenance=dict(executable_sha256=sha(a.executable),camera_sha256=sha(a.cameras),sources={n:sha(BASE/n) for n in ('fit_rig_pixels.h','evaluate_rig_pixels.cpp','compare_rig_jacobian.py')}),limitations=['Analytic camera derivative composed with existing finite-difference FK; hybrid retains finite translation columns. Solver, tolerances and iteration budgets retained.','Numerical pose agreement and host timing, not physical tracking accuracy or native performance.'])
 (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
