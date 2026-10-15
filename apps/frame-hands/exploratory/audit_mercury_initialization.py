"""Check initialization experiment metrics and paired observation provenance."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from mercury_forward import NativeHandGenerator
from mercury_native_solver import ML_TO_XR


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('result','dll','parity','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();summary=json.loads((args.result/'summary.json').read_text());protocol=summary['protocol'];parity=json.loads((args.parity/'summary.json').read_text())
    assert hashlib.sha256(args.dll.read_bytes()).hexdigest()==summary['dll_sha256']==parity['dll_sha256']
    assert parity['counts']['exact_total']==4464
    for name,digest in summary['source_sha256'].items():assert hashlib.sha256((args.result/'snapshot'/name).read_bytes()).hexdigest()==digest
    generator=NativeHandGenerator(args.dll);reports=[];count=0;exact=0;accepted=0
    for case in summary['reports']:
        side=int(case['case'][4]);pose=case['case'].split('-')[1];condition=case['case'].split('-')[2]
        angles=np.tile(protocol['poses'][pose],(4,1));truth=generator.generate(protocol['wrist_internal'],angles,side)[ML_TO_XR,:3]
        rows=[json.loads(line) for line in (args.result/(case['case']+'.jsonl')).read_text().splitlines()];assert len(rows)==60
        if pose=='open':reference=[json.loads(line) for line in (Path(protocol['reference'])/f'side{side}-ramp6-{condition}.jsonl').read_text().splitlines()][:60]
        decomposition={name:[] for name in case['arms']}
        for index,row in enumerate(rows):
            assert row['index']==index;np.testing.assert_array_equal(truth,row['truth']);np.testing.assert_array_equal(angles,row['angles'])
            if pose=='open':
                assert row['views']==reference[index]['views']
                np.testing.assert_array_equal(np.asarray(row['arms']['cold-smoothing2']['fit']['joints_left_camera_m'])[ML_TO_XR],reference[index]['arms']['native']['joints_m']);exact+=1
            if index==0:
                for initial in ('cold','truth_once'):assert row['arms'][initial+'-smoothing0']==row['arms'][initial+'-smoothing2']
            for name,record in row['arms'].items():
                joints=np.asarray(record['fit']['joints_left_camera_m'])[ML_TO_XR];fingers=np.asarray(record['angles']);errors=np.linalg.norm(joints-truth,axis=1)*1000
                assert abs(errors.max()-record['max_joint_error_mm'])<1e-10
                assert abs(errors.mean()-record['mean_joint_error_mm'])<1e-10
                assert abs(abs(fingers-angles).max()-record['max_angle_error_rad'])<1e-10
                relative=(joints-joints[0])-(truth-truth[0]);decomposition[name].append(dict(wrist_mm=float(errors[0]),wrist_relative_max_mm=float(np.linalg.norm(relative,axis=1).max()*1000)))
                count+=1;accepted+=bool(record['fit']['native_active'] and record['fit']['passes_native_reprojection_gate'])
        arms={}
        for name,claimed in case['arms'].items():
            values=[row['arms'][name] for row in rows]
            metrics=dict(first_max_error_mm=values[0]['max_joint_error_mm'],final_max_error_mm=values[-1]['max_joint_error_mm'],settled_p95_max_error_mm=float(np.percentile([r['max_joint_error_mm'] for r in values[40:]],95)),final_max_angle_error_rad=values[-1]['max_angle_error_rad'],final_reprojection_error=values[-1]['fit']['reprojection_error'])
            assert metrics==claimed
            arms[name]=dict(**metrics,first_decomposition=decomposition[name][0],final_decomposition=decomposition[name][-1])
        reports.append(dict(case=case['case'],arms=arms))
    assert count==1920 and exact==240
    result=dict(verified_fits=count,accepted_fits=accepted,exact_previous_synthetic_control_fits=exact,exact_recorded_control_fits=parity['counts']['exact_total'],reports=reports,interpretation='In this declared geometry family, truth-once clean starts retain the exact pose whereas normal initialization may leave errors. Zero stability sometimes helps convergence but can worsen noisy fits. Truth initialization is unavailable in real tracking; this diagnoses a failure mechanism, not an implementable tracking improvement.',next='Inspect native depth-based initial palm guess and evaluate a current-observation stereo palm initializer without truth access, including degeneracy and real recorded controls.')
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='reports'}))


if __name__=='__main__':main()
