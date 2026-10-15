"""Guarded ABBA pixel-refinement benchmark using frozen observations; no camera or XR."""
import argparse
import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path
from workflow_guard import native_workflow


def rows(path):
    result = [list(map(float, line.split())) for line in path.read_text().splitlines()]
    if not result or any(len(row) != 99 or not all(map(math.isfinite, row)) for row in result):
        raise ValueError('Incomplete or nonfinite fit output')
    return result


def stats(values):
    values = sorted(values)
    def percentile(p):
        index = (len(values) - 1) * p
        lo = int(index)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)
    return dict(p50=percentile(.5), p95=percentile(.95), maximum=max(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.bundle.resolve()
    manifest = root / 'manifest.json'
    metadata = json.loads(manifest.read_text())
    if platform.machine() != 'aarch64':
        raise ValueError('Native ARM64 only')
    for name, expected in metadata['sha256'].items():
        if Path(name).name != name or hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Changed bundle file: ' + name)
    for item in metadata['datasets']:
        if not 1 <= item['cases'] <= 512:
            raise ValueError('Unbounded case count')
        for key in ('profile', 'input', 'finite_reference', 'hybrid_reference'):
            if item[key] not in metadata['sha256']:
                raise ValueError('Unverified input')
    if args.output.exists():
        raise FileExistsError(args.output)
    with native_workflow():
        args.output.mkdir(parents=True)
        results = []
        for item in metadata['datasets']:
            runs = []
            for index, mode in enumerate(('finite', 'hybrid', 'hybrid', 'finite')):
                output = args.output / f"{item['name']}-{index}-{mode}.txt"
                subprocess.run([str(root / 'evaluate-rig-pixels'), str(root / item['profile']),
                                str(root / 'cameras.txt'), str(root / item['input']), str(output.resolve()), mode],
                               check=True, capture_output=True, text=True, timeout=30)
                actual = rows(output)
                expected = rows(root / item[mode + '_reference'])
                if len(actual) != item['cases'] or len(expected) != item['cases']:
                    raise ValueError('Changed case count')
                worst = 0.
                for a, b in zip(actual, expected):
                    if a[:2] != b[:2] or a[8] != b[8] or a[5] > a[4] + 1e-9:
                        raise ValueError('Side, sufficiency or objective invariant changed')
                    worst = max(worst, max(abs(x-y) for x,y in zip(a[36:], b[36:])))
                if worst > .05:
                    raise ValueError('Host/native discrepancy exceeds .05mm: ' + str(worst))
                runs.append(dict(mode=mode, max_host_native_coordinate_difference_mm=worst,
                                 pixel_fit_us=stats([row[7] for row in actual])))
            results.append(dict(name=item['name'], cases=item['cases'], runs=runs))
        report = dict(scope=__doc__, results=results,
                      manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                      environment=dict(machine=platform.machine(), kernel=platform.release(),
                                       python=platform.python_version(), os_release=Path('/etc/os-release').read_text()),
                      limitations=['Pixel-refinement stage only, excluding initialization, detector, capture and rendering.',
                                   'ABBA order, no warmup exclusion. Same saved inputs and binary in each pair.',
                                   'Numerical equivalence to matching host mode is not physical pose accuracy.',
                                   'No XR, new recording or accepted-runtime changes. Wearer mode not observed.'])
        (args.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
