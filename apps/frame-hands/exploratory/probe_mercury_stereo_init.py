"""Compare stereo-ray startup with unchanged cold controls on frozen inputs."""
import argparse,hashlib,json,shutil
from contextlib import ExitStack
from pathlib import Path
import numpy as np
from mercury_native_solver import NativeSolver,ML_TO_XR
from mercury_stereo_init import initialize_from_stereo
from test_mercury_native_solver import POSE


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dll','reference','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();source=json.loads((a.reference/'summary.json').read_text());a.output.mkdir(parents=True,exist_ok=False)
    protocol=dict(reference=str(a.reference.resolve()),conditions=[r['case'] for r in source['reports']],arms=['cold-smoothing0','stereo-smoothing0','cold-smoothing2','stereo-smoothing2'],frames_per_case=60,observations='Exact frozen inputs from previous initialization study. Candidate sees only packet, side, calibration and fixed90mm size. Truth arrays are used exclusively after fitting to score.',policy='At frame0 only, triangulate wrist and four MCP joints, rigidly fit native default palm, keep native default finger articulation and first-frame stability behavior. If geometry fails, retain original cold startup unchanged.',thresholds=dict(palm_confidence_min=.5,ray_sine_min=.02,ray_distance_m=[.1,1.5],ray_gap_max_m=.01,palm_rms_max_m=.015,palm_second_first_singular_ratio_min=.02),selection='One declared geometry policy, no parameter sweep or outcome-based threshold changes.',limitation='Synthetic model geometry and isolated startup, not live accuracy or acquisition. No oracle state is supplied to the candidate.')
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');snapshot=a.output/'snapshot';snapshot.mkdir()
    for name in ('probe_mercury_stereo_init.py','mercury_stereo_init.py','mercury_native_solver.py'):shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    shutil.copy2(a.dll.parent/'build.json',a.output/'build.json');reports=[];exact=0
    for case in source['reports']:
        side=int(case['case'][4]);rows=[json.loads(line) for line in (a.reference/(case['case']+'.jsonl')).read_text().splitlines()];records={name:[] for name in protocol['arms']};initializations={}
        with ExitStack() as stack:
            solvers={name:stack.enter_context(NativeSolver(a.dll,POSE,side)) for name in records};stream=stack.enter_context((a.output/(case['case']+'.jsonl')).open('x'))
            for row in rows:
                index=row['index'];views=row['views'];result=dict(index=index,views=views,truth=row['truth'],arms={})
                for name,solver in solvers.items():
                    if index==0 and name.startswith('stereo'):initializations[name]=initialize_from_stereo(solver,views)
                    fit=solver.run(views,smoothing=int(name[-1]),depth_weight=0)
                    if name.startswith('cold'):
                        assert fit==row['arms'][name]['fit'];exact+=1
                    # Ground truth is read only for scoring AFTER the optimizer call.
                    points=np.asarray(fit['joints_left_camera_m'])[ML_TO_XR];truth=np.asarray(row['truth']);error=np.linalg.norm(points-truth,axis=1)*1000
                    record=dict(fit=fit,angles=solver.fingers().tolist(),max_joint_error_mm=float(error.max()),wrist_error_mm=float(error[0]),mean_joint_error_mm=float(error.mean()))
                    records[name].append(record);result['arms'][name]=record
                stream.write(json.dumps(result,allow_nan=False)+'\n')
        report=dict(case=case['case'],initializations=initializations,arms={name:dict(first_max_error_mm=r[0]['max_joint_error_mm'],final_max_error_mm=r[-1]['max_joint_error_mm'],settled_p95_max_error_mm=float(np.percentile([x['max_joint_error_mm'] for x in r[40:]],95)),final_wrist_error_mm=r[-1]['wrist_error_mm']) for name,r in records.items()})
        reports.append(report);print(json.dumps(report),flush=True)
    assert exact==960
    summary=dict(protocol=protocol,reports=reports,exact_original_control_fits=exact,total_fits=1920,dll_sha256=hashlib.sha256(a.dll.read_bytes()).hexdigest(),source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.glob('*.py')});(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
