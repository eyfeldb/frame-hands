"""Synthetic oracle check for native camera-relative optimizer history.

A fixed native hand is rendered analytically into a moving stereo camera pair.
No image model, physical correctness claim, or live state is involved.
"""
import argparse,json,hashlib,shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from mercury_native_solver import NativeSolver,ML_TO_XR
from test_mercury_native_solver import POSE,synthetic_views,observations_from_joints


def camera_motion(index):
    # Thirty stationary frames, turn/translate out and back over sixty, then stop.
    phase=np.clip((index-30)/60,0,1);wave=np.sin(np.pi*phase)
    return Rotation.from_euler('y',wave*30,degrees=True).as_matrix(),np.array([wave*.06,0,0])


def summary(values):
    a=np.asarray(values);return dict(median=float(np.median(a)),p95=float(np.percentile(a,95)),maximum=float(np.max(a)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dll',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);rows=[]
    for side in (0,1):
        with NativeSolver(a.dll,POSE,side) as seed:
            truth=np.asarray(seed.run(synthetic_views(side),depth_weight=0)['joints_left_camera_m'])[ML_TO_XR]
        for moving in (False,True):
            for smoothing,transport in ((0,False),(2,False),(2,True)):
                frames=[];previous_camera=None
                with NativeSolver(a.dll,POSE,side) as solver:
                    for index in range(120):
                        rotation,translation=camera_motion(index) if moving else (np.eye(3),np.zeros(3))
                        observed=(truth-translation)@rotation
                        camera=np.eye(4);camera[:3,:3]=rotation;camera[:3,3]=translation
                        if transport and previous_camera is not None:solver.transport(np.linalg.inv(camera)@previous_camera)
                        previous_camera=camera
                        fit=solver.run(observations_from_joints(observed,side),smoothing=smoothing,depth_weight=0)
                        estimated=np.asarray(fit['joints_left_camera_m'])[ML_TO_XR]@rotation.T+translation
                        error=np.linalg.norm(estimated-truth,axis=1)*1000
                        frames.append(dict(index=index,seconds=index/30,truth_world_m=truth.tolist(),estimated_world_m=estimated.tolist(),error_mm=error.tolist(),native_error=fit['reprojection_error']))
                rows.append(dict(side=side,moving=moving,smoothing=smoothing,transport=transport,moving_phase_max_joint_error_mm=summary([max(f['error_mm']) for f in frames[30:91]]),stopped_phase_max_joint_error_mm=summary([max(f['error_mm']) for f in frames[91:]]),frames=frames))
                print(side,moving,smoothing,transport,rows[-1]['moving_phase_max_joint_error_mm'],flush=True)
    result=dict(scope='Synthetic exact stereo projections of a solver-generated constant hand, 6cm parallel baseline,30Hz,30degree/60mm camera excursion. Thirty startup frames. No noise/dropout, image model, Frame calibration, or measured human ground truth. Native size90mm fixed; relative-depth observations disabled.',dll_sha256=hashlib.sha256(a.dll.read_bytes()).hexdigest(),source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('probe_mercury_head_motion.py','test_mercury_native_solver.py','mercury_native_solver.py')},rows=rows)
    for name in result['source_sha256']:shutil.copy2(Path(__file__).with_name(name),a.output/name)
    shutil.copy2(a.dll.parent/'build.json',a.output/'build.json')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
