"""Offline stationary truth: distinguish cold initialization and regularization."""
import argparse,hashlib,json,shutil
from contextlib import ExitStack
from pathlib import Path
import numpy as np
from mercury_forward import NativeHandGenerator
from mercury_native_solver import NativeSolver,ML_TO_XR
from mercury_synthetic_seed import seed_known_synthetic_hand
from test_mercury_native_solver import POSE,observations_from_joints


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dll','reference','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    protocol=dict(frames=60,poses={'open':[-.1,-.05,-.05],'closed':[-1.2,-1.2,-.8]},sides=[0,1],conditions=['clean','gaussian'],smoothing=[0,2],initialization=['cold','truth_once'],size_m=.09,wrist_internal=[.70710678,0,0,.70710678,.04,-.03,-.5],noise='Same declared independent XY0.5pixel/curl0.4rad noise as gesture response. Seed20261005+100*side+6 for open and +30 for closed. Same packets shared across four arms.',first_step='All arms retain native first_frame=true so first fit has no temporal regularization. Truth-only arm initializes native state once before frame0; subsequent frames use only observations and own history.',purpose='Diagnostic of initialization versus fitting mismatch, not an oracle-based tracking candidate. Fixed before generating results.',reference=str(a.reference.resolve()))
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');snapshot=a.output/'snapshot';snapshot.mkdir()
    for name in ['probe_mercury_initialization.py','mercury_synthetic_seed.py','mercury_forward.py','mercury_native_solver.py','test_mercury_native_solver.py']:shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    shutil.copy2(a.dll.parent/'build.json',a.output/'build.json');gen=NativeHandGenerator(a.dll);reports=[];exact=0
    for side in (0,1):
        for pose,values in protocol['poses'].items():
            angles=np.tile(values,(4,1));truth=gen.generate(protocol['wrist_internal'],angles,side)[ML_TO_XR,:3]
            for condition in protocol['conditions']:
                case=f'side{side}-{pose}-{condition}';rng=np.random.default_rng(20261005+100*side+(6 if pose=='open' else 30));records={}
                if pose=='open':reference=[json.loads(line) for line in (a.reference/f'side{side}-ramp6-{condition}.jsonl').read_text().splitlines()][:60]
                with ExitStack() as stack:
                    solvers={}
                    for smoothing in (0,2):
                        for initial in ('cold','truth_once'):
                            name=f'{initial}-smoothing{smoothing}';solver=stack.enter_context(NativeSolver(a.dll,POSE,side))
                            if initial=='truth_once':seed_known_synthetic_hand(solver,protocol['wrist_internal'],angles)
                            solvers[name]=(solver,smoothing);records[name]=[]
                    stream=stack.enter_context((a.output/(case+'.jsonl')).open('x'))
                    for index in range(60):
                        views=observations_from_joints(truth,side)
                        for view in views.values():
                            view['curls'][1:5]=angles.sum(1).tolist();view['curls'][5:]=[0.]*5
                            if condition=='gaussian':
                                view['points_crop']=(np.asarray(view['points_crop'])+rng.normal(0,.5,(21,2))).tolist();view['curls'][1:5]=(angles.sum(1)+rng.normal(0,.4,4)).tolist();view['curls'][6:10]=[.15]*4
                        if pose=='open':assert views==reference[index]['views']
                        row=dict(index=index,truth=truth.tolist(),angles=angles.tolist(),views=views,arms={})
                        for name,(solver,smoothing) in solvers.items():
                            fit=solver.run(views,smoothing=smoothing,depth_weight=0);joints=np.asarray(fit['joints_left_camera_m'])[ML_TO_XR];fingers=solver.fingers();error=np.linalg.norm(joints-truth,axis=1)*1000
                            record=dict(fit=fit,angles=fingers.tolist(),max_joint_error_mm=float(error.max()),mean_joint_error_mm=float(error.mean()),max_angle_error_rad=float(abs(fingers-angles).max()))
                            row['arms'][name]=record;records[name].append(record)
                            if pose=='open' and name=='cold-smoothing2':
                                np.testing.assert_array_equal(joints,reference[index]['arms']['native']['joints_m']);np.testing.assert_array_equal(fingers,reference[index]['arms']['native']['angles']);exact+=1
                        stream.write(json.dumps(row,allow_nan=False)+'\n')
                report=dict(case=case,arms={name:dict(first_max_error_mm=values[0]['max_joint_error_mm'],final_max_error_mm=values[-1]['max_joint_error_mm'],settled_p95_max_error_mm=float(np.percentile([r['max_joint_error_mm'] for r in values[40:]],95)),final_max_angle_error_rad=values[-1]['max_angle_error_rad'],final_reprojection_error=values[-1]['fit']['reprojection_error']) for name,values in records.items()})
                reports.append(report);print(json.dumps(report),flush=True)
    summary=dict(protocol=protocol,reports=reports,exact_previous_control_fits=exact,fit_count=1920,dll_sha256=hashlib.sha256(a.dll.read_bytes()).hexdigest(),source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.glob('*.py')})
    assert exact==240;(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
