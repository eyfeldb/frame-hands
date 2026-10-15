"""Compare staged Godot skinning output against frozen independent geometry."""
import argparse, json
from pathlib import Path
import numpy as np
from export_profile_mesh import sha

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run/'manifest.json').read_text())
    for name, digest in manifest['sha256'].items():
        if sha(args.run/name) != digest:
            raise ValueError('Changed fixture: '+name)
    actual = json.loads((args.run/'godot.json').read_text())
    expected = np.load(args.run/'expected.npz')
    report = dict(cases=len(actual['cases']))
    if report['cases'] != manifest['cases']:
        raise ValueError('Missing cases')
    for field in ('vertices', 'landmarks'):
        values = np.asarray([row[field] for row in actual['cases']])
        if values.shape != expected[field].shape or not np.isfinite(values).all():
            raise ValueError('Invalid geometry: '+field)
        error_mm = np.linalg.norm(values-expected[field], axis=-1)*1000
        report[field] = dict(max_mm=float(error_mm.max()), p95_mm=float(np.percentile(error_mm,95)))
        if error_mm.max() > .001:
            raise ValueError('Geometry disagreement: '+field)
    for field in ('wrong_profile_side_rejected', 'invalid_angles_rejected'):
        if actual.get(field) is not True:
            raise ValueError('Missing rejection control: '+field)
        report[field] = True
    (args.run/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
if __name__ == '__main__':
    main()
