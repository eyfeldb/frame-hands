"""Check analytic KB derivatives against independent five-point reference projection differences."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE.parent/'exploratory'))
from frame_data import load_calibration
from rig_pixel_data import write_cameras,NAMES
from audit_native_lengths import sha

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('upstream','persist','head-points','executable','output'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();sys.path.insert(0,str(a.upstream));a.output.mkdir(parents=True,exist_ok=False);cams=load_calibration(a.persist);write_cameras(cams,a.output/'cameras.txt');head=np.loadtxt(a.head_points);extra=[]
 # Exercise both optical axes and both sides of the small-angle series boundary.
 for name in NAMES:
  c=cams[name];R=c.camera_to_world_xf[:3,:3].T;t=-R@c.camera_to_world_xf[:3,3]
  for radius in (0.,1e-9,1e-6,.001,.049,.05,.051):extra.append(np.linalg.solve(R,np.array([radius,0,500])-t))
 head=np.vstack([head,extra]);np.savetxt(a.output/'points.txt',head,fmt='%.17g');subprocess.run([str(a.executable.resolve()),str((a.output/'cameras.txt').resolve()),str((a.output/'points.txt').resolve()),str((a.output/'jacobian.txt').resolve())],check=True)
 actual=np.loadtxt(a.output/'jacobian.txt').reshape(-1,2,2,3);reference=np.empty_like(actual);step=.001
 for v,name in enumerate(NAMES):
  c=cams[name]
  def project(p):return c.eye_to_window(c.world_to_eye(p))
  for axis in range(3):
   d=np.eye(3)[axis]*step;reference[:,v,:,axis]=(-project(head+2*d)+8*project(head+d)-8*project(head-d)+project(head-2*d))/(12*step)
 maximum=float(np.max(np.abs(actual-reference)))
 if actual.shape!=(len(head),2,2,3) or not np.isfinite(actual).all() or maximum>2e-7:raise ValueError('Analytic derivative mismatch: '+str(maximum))
 bad=cams[NAMES[0]].camera_to_world_xf[:3,3]-100*cams[NAMES[0]].camera_to_world_xf[:3,2];np.savetxt(a.output/'behind.txt',bad[None]);rejected=subprocess.run([str(a.executable.resolve()),str((a.output/'cameras.txt').resolve()),str((a.output/'behind.txt').resolve()),str((a.output/'behind-output.txt').resolve())],capture_output=True,text=True)
 if rejected.returncode==0:raise ValueError('Behind-camera derivative accepted')
 report=dict(scope=__doc__,points=len(head),extra_axis_and_series_boundary_points=len(extra),five_point_step_mm=step,max_derivative_difference_px_per_mm=maximum,behind_camera_rejected=True,provenance=dict(executable_sha256=sha(a.executable),camera_sha256=sha(a.output/'cameras.txt'),points_sha256=sha(a.output/'points.txt'),sources={n:sha(BASE/n) for n in ('fit_rig_pixels.h','evaluate_rig_camera_jacobian.cpp','test_projection_jacobian.py')}))
 (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
