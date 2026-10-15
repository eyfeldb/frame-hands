"""Fit pinned native Mercury kinematics to frozen saved stereo observations.

No capture, inference, association, or display state is created by this probe.
All image distances compare estimators, not physical ground truth.
"""
import argparse, hashlib, json, shutil, sys, time
from pathlib import Path
import cv2
import numpy as np
from frame_data import Recording, load_calibration
from mercury_native_solver import NativeSolver, camera_transforms, ML_TO_XR
from probe_mercury_keypoints import draw


def project(result, head_from_left, cameras):
    joints=np.asarray(result['joints_left_camera_m'])[ML_TO_XR]
    head=(joints@head_from_left[:3,:3].T+head_from_left[:3,3])*1000
    return {name:cam.eye_to_window(cam.world_to_eye(head)) for name,cam in cameras.items()}


def stats(values):
    a=np.asarray(values)
    return dict(median=float(np.median(a)),p95=float(np.percentile(a,95)),maximum=float(np.max(a)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['observations','dll','output','upstream']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();sys.path.insert(0,str(args.upstream.resolve()));root=args.observations.parent;source=json.loads(args.observations.read_text());protocol=json.loads((root/'protocol.json').read_text())
    args.output.mkdir(parents=True,exist_ok=False);snapshot=args.output/'snapshot';snapshot.mkdir()
    for name in ['mercury_native_solver.py','probe_mercury_solver.py']:
        shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    shutil.copy2(args.dll.parent/'build.json',args.output/'build.json')
    reports=[]
    for report in source['reports']:
        case=report['case'];session=Path(case['session']);cal=session/'calibration'
        cameras=load_calibration(cal if cal.exists() else session/'frame/snapshot/device/persist');pose,head_from_left=camera_transforms(cameras)
        views={v['camera']:v['result'] for v in report['views']}
        with NativeSolver(args.dll,pose,case['side']) as solver:
            start=time.perf_counter();fitted=solver.run(views);elapsed=(time.perf_counter()-start)*1000
            solver.run(views);again=solver.run(views,reset=True)
        reset_delta=float(np.max(np.abs(np.asarray(fitted['joints_left_camera_m'])-again['joints_left_camera_m'])))
        if reset_delta!=0:raise AssertionError('State reset failed deterministic saved-input comparison')
        projected=project(fitted,head_from_left,cameras)
        rec=Recording(session/'frame/sets.bin');_,stamp,images=next(rec.frames(report['seconds']-.000001,report['seconds']+.000001))
        if stamp!=report['stamp']:raise AssertionError('Capture timestamp mismatch')
        reference=json.loads((Path(protocol['case_root'])/case['name']/'report.json').read_text())
        metrics=[];panels=[]
        for v in report['views']:
            name=v['camera'];image=images[name][0];observed=np.asarray(v['result']['points_raw']);native=projected[name]
            ncnn=np.asarray(next(r['model_points'] for r in reference['before_verifier'] if r['camera']==name))
            center=np.asarray(v['roi_center']);size=v['roi_size']
            x0=max(0,int(center[0]-size*.65));x1=min(image.shape[1],int(center[0]+size*.65));y0=max(0,int(center[1]-size*.65));y1=min(image.shape[0],int(center[1]+size*.65))
            raw=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
            columns=[raw,draw(image,observed,(0,255,0)),draw(image,native,(255,255,0)),draw(image,ncnn,(255,0,255))]
            panel=cv2.hconcat([cv2.resize(c[y0:y1,x0:x1],(384,384)) for c in columns]);cv2.putText(panel,name,(8,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1);panels.append(panel)
            metrics.append(dict(camera=name,native_vs_mercury_px=stats(np.linalg.norm(native-observed,axis=1)),native_vs_ncnn_px=stats(np.linalg.norm(native-ncnn,axis=1)),mercury_vs_ncnn_px=stats(np.linalg.norm(observed-ncnn,axis=1)),projected_native=native.tolist(),raw_sha256=hashlib.sha256(image.tobytes()).hexdigest()))
        cv2.imwrite(str(args.output/(case['name']+'-review.jpg')),cv2.vconcat(panels))
        reports.append(dict(case=case,stamp=stamp,seconds=report['seconds'],fit=fitted,first_fit_ms=elapsed,reset_max_delta_m=reset_delta,views=metrics))
        print(case['name'],fitted['reprojection_error'],[round(v['native_vs_ncnn_px']['median'],2) for v in metrics],flush=True)
    summary=dict(scope='Eight independent snapshots; fixed90mm native hand size; solver recreated per capture; no tracking or identity claim. No physical ground truth.',legend='Each camera row: raw / green raw Mercury / cyan constrained native Mercury / magenta existing ncnn; all share fixed source crop.',observation_sha256=hashlib.sha256(args.observations.read_bytes()).hexdigest(),dll_sha256=hashlib.sha256(args.dll.read_bytes()).hexdigest(),source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.glob('*.py')},reports=reports)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
