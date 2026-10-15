"""Freeze a bounded paired projection-derivative benchmark from completed host comparisons."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = args.private_root
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent
    for original, name in [(base/'length-tools/evaluate-rig-pixels-jac-aarch64', 'evaluate-rig-pixels'),
                           (base/'rig-pixels-viewer-v2/cameras.txt', 'cameras.txt'),
                           (source/'workflow_guard.py', 'workflow_guard.py'),
                           (source/'run_native_jacobian_check.py', 'run_native_jacobian_check.py')]:
        shutil.copyfile(original, root/name)
    datasets = []
    review = []
    items = [('generic_synthetic', base/'rig-generic-v1/generic.fhrig', base/'rig-pixels-synthetic-v7/staged-input.txt')]
    items += [(f'palm_side{s}', base/f'palm-profile-v1/side{s}.fhrig', base/f'palm-profile-viewer-v2/side{s}-pixel-input.txt') for s in (0, 1)]
    for name, profile, inputs in items:
        outputs = {mode: np.loadtxt(base/'rig-jacobian-comparison-v2'/name/(mode+'.txt')) for mode in ('finite', 'analytic', 'hybrid')}
        count = len(outputs['finite'])
        delta = {mode: np.max(np.linalg.norm((out[:,36:]-outputs['finite'][:,36:]).reshape(-1,21,3), axis=2), axis=1) for mode,out in outputs.items()}
        indices = sorted(set(map(int, np.linspace(0,count-1,32,dtype=int).tolist() +
                                 [np.argmax(d) for d in delta.values()] +
                                 [i for d in delta.values() for i in np.flatnonzero(d>.05)])))
        profile_name = name+'.fhrig'
        input_name = name+'-input.txt'
        shutil.copyfile(profile, root/profile_name)
        lines = inputs.read_text().splitlines()
        (root/input_name).write_text('\n'.join(lines[i] for i in indices)+'\n')
        item = dict(name=name, profile=profile_name, input=input_name, cases=len(indices),
                    source_indices=indices, input_source_sha256=sha(inputs))
        limits = np.loadtxt(profile, skiprows=1, max_rows=20)[:,6:8]
        for mode, out in outputs.items():
            if out.shape!=(count,99) or not np.isfinite(out).all() or np.any(out[:,5]>out[:,4]+1e-9):
                raise ValueError('Invalid output or objective')
            if not ((out[:,16:36]>=limits[:,0]-1e-8)&(out[:,16:36]<=limits[:,1]+1e-8)).all():
                raise ValueError('Joint limit violation')
        for mode in ('finite','hybrid'):
            filename = name+'-'+mode+'-reference.txt'
            np.savetxt(root/filename, outputs[mode][indices], fmt='%.17g')
            item[mode+'_reference'] = filename
        datasets.append(item)
        if name=='generic_synthetic':
            truth = np.load(base/'rig-pixels-synthetic-v7/reference.npz')['truth']
            groups = np.array(json.loads((base/'rig-pixels-synthetic-v7/groups.json').read_text()))
            for mode,out in outputs.items():
                error = np.linalg.norm(out[:,36:].reshape(-1,21,3)-truth,axis=2)
                np.testing.assert_array_equal(out[groups=='missing_thumb',8:],out[groups=='missing_dummy',8:])
                if np.any(out[-2:,0]!=0) or np.any(out[-2:,2:4]!=0):
                    raise ValueError('Missing stereo treated as fitted')
                review.append(dict(mode=mode, case325_group=str(groups[325]),
                                   case325_rmse_mm=float(np.sqrt(np.mean(error[325]**2))),case325_cost=float(out[325,5]),
                                   group_pose_rmse_mm={str(g):float(np.sqrt(np.mean(error[groups==g]**2))) for g in dict.fromkeys(groups)}))
    (root/'outlier-review.json').write_text(json.dumps(review,indent=2)+'\n')
    manifest = dict(scope=__doc__,selection='32 uniform per dataset plus largest deviations and every case above .05mm in analytic or hybrid comparison',
                    datasets=datasets,sha256={p.name:sha(p) for p in root.iterdir() if p.is_file()})
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    snapshot=root/'source-audit'
    snapshot.mkdir()
    for name in ('fit_rig_pixels.h','evaluate_rig_pixels.cpp','compare_rig_jacobian.py','run_native_jacobian_check.py',
                 'prepare_rig_jacobian_check.py','test_projection_jacobian.py','evaluate_rig_camera_jacobian.cpp'):
        shutil.copyfile(source/name,snapshot/name)
    print(json.dumps(dict(cases={i['name']:i['cases'] for i in datasets},manifest_sha256=sha(root/'manifest.json')),indent=2))

if __name__=='__main__':
    main()
