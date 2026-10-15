"""One-event causal ablation of a native finger-stability multiplier.

This overrides one learned curl only to isolate its existing temporal-weight path;
it is not a new inference estimate or a deployable prediction policy.
"""
import argparse
from copy import deepcopy
from contextlib import ExitStack
import hashlib,json,sys
from pathlib import Path
import cv2,numpy as np
from frame_data import Recording,load_calibration
from mercury_native_solver import NativeSolver,camera_transforms,pack_views
from probe_mercury_solver import project
from probe_mercury_keypoints import draw
from replay_mercury_solver import to_world
from run_mesh_trial import pose_matrix


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','upstream','dll','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seconds',type=float,required=True);p.add_argument('--side',type=int,choices=(0,1),required=True);p.add_argument('--finger',type=int,choices=range(4),required=True)
    a=p.parse_args();sys.path.insert(0,str(a.upstream.resolve()));a.output.mkdir(parents=True,exist_ok=False)
    parent=json.loads((a.reference/'protocol.json').read_text());source=Path(parent['source']);origin=json.loads((Path(json.loads((source/'protocol.json').read_text())['source'])/'protocol.json').read_text())
    assert hashlib.sha256((source/'frames.jsonl').read_bytes()).hexdigest()==parent['source_frames_sha256']
    base={}
    with Path(origin['baseline']).open() as f:
        for line in f:
            row=json.loads(line);base[row['capture_raw_ns']]=row['capture_pose']
    rec=Recording(Path(origin['session'])/'frame/sets.bin');cameras=load_calibration(Path(origin['session'])/'calibration');pose,head_from_left=camera_transforms(cameras)
    counts=0;previous_camera=None;output=None
    with ExitStack() as stack:
        solvers=[stack.enter_context(NativeSolver(a.dll,pose,a.side)) for _ in range(2)]
        refs=stack.enter_context((a.reference/'frames.jsonl').open());sources=stack.enter_context((source/'frames.jsonl').open())
        for line in refs:
            row=json.loads(line);observed=json.loads(next(sources));assert row['stamp']==observed['stamp']
            if row['seconds']>a.seconds+.000001:break
            ref=next((f for f in row['fits'] if f['side']==a.side and f['branch']=='box'),None)
            if ref is None:continue
            world=pose_matrix(base[row['stamp']]['capture']);current=world.copy();current[:3,3]*=.001;current=current@head_from_left
            views={v['camera']:v['model'] for v in observed['observations'] if v['side']==a.side and v['branch']=='box'};reset=ref['reset_reason'] is not None
            event=abs(row['seconds']-a.seconds)<.000001;results=[];priors=[];diagnostics=[];packets=[]
            if event:assert not reset,'Selected event must have active temporal stability'
            for index,solver in enumerate(solvers):
                if not reset:solver.transport(np.linalg.inv(current)@previous_camera)
                test=deepcopy(views)
                if event:
                    priors.append(solver.fingers().tolist());prior_sum=float(solver.fingers()[a.finger].sum(dtype=np.float32))
                    if index:
                        for v in test.values():v['curls'][a.finger+1]=prior_sum
                    diagnostics.append(solver.finger_stability(test).tolist());packets.append(pack_views(test,a.side))
                fit=solver.run(test,reset=reset,smoothing=2)
                if not event or index==0:assert fit==ref['fit'],'Baseline history changed'
                results.append(dict(fit=fit,angles=solver.fingers().tolist(),world_joints_mm=to_world(fit,head_from_left,world).tolist(),projected={k:v.tolist() for k,v in project(fit,head_from_left,cameras).items()}))
            counts+=1;previous_camera=current
            if event:
                np.testing.assert_array_equal(priors[0],priors[1]);changed=np.flatnonzero(packets[0]!=packets[1]);expected=[121*v+111+2*(a.finger+1) for v in range(2)];assert changed.tolist()==expected
                assert diagnostics[1][a.finger][5]==1
                output=dict(seconds=row['seconds'],stamp=row['stamp'],side=a.side,finger=a.finger,exact_prior_fits=counts-1,prior_angles=priors[0],diagnostics=diagnostics,changed_packet_indices=changed.tolist(),arms=dict(original=results[0],full_selected_finger_stability=results[1]),scope='Post-hoc one-event causal ablation, no future history. Only selected finger learned curl values replaced by identical prior sum; native direct curl costs/initialization are disabled, isolating current temporal multiplier. Variances and all image/depth/crop inputs unchanged.',dll_sha256=hashlib.sha256(a.dll.read_bytes()).hexdigest());break
    if output is None:raise ValueError('Exact eligible event not found')
    _,stamp,images=next(rec.frames(a.seconds-.000001,a.seconds+.000001));assert stamp==output['stamp'];panels=[]
    for name in ('slam_left','slam_right'):
        image=images[name][0];columns=[cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)]+[draw(image,r['projected'][name],(255,255,0) if a.side==0 else (0,140,255)) for r in output['arms'].values()]
        panel=cv2.hconcat([cv2.resize(c,(512,round(512*image.shape[0]/image.shape[1]))) for c in columns]);cv2.putText(panel,name+' raw / original / one-finger weight1',(5,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);panels.append(panel)
    cv2.imwrite(str(a.output/'review.jpg'),cv2.vconcat(panels));(a.output/'summary.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(dict(prior=output['prior_angles'][a.finger],after={k:v['angles'][a.finger] for k,v in output['arms'].items()},residual={k:v['fit']['reprojection_error'] for k,v in output['arms'].items()},exact_prior_fits=output['exact_prior_fits'])))


if __name__=='__main__':main()
