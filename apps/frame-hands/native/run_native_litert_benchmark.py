"""Run a bounded, exclusive native CPU benchmark using saved crops only."""
import argparse, json, signal
from pathlib import Path
from litert_benchmark_core import require_native, verify_bundle, benchmark, sha
from workflow_guard import native_workflow

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require_native()
    if args.output.exists(): raise RuntimeError('Refusing to overwrite results')
    def timeout(*_): raise TimeoutError('Native benchmark exceeded 180 second wall bound')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(180)
    try:
        with native_workflow():
            manifest = verify_bundle(args.bundle)
            for name in ('run_native_litert_benchmark.py', 'litert_benchmark_core.py', 'workflow_guard.py'):
                if sha(Path(__file__).with_name(name)) != manifest['files']['scripts/'+name]:
                    raise RuntimeError('Executed source differs from frozen bundle: '+name)
            result = benchmark(args.bundle, args.bundle/'runtime', 'Frame-native saved-crop CPU component benchmark')
            result['runner_sha256'] = sha(Path(__file__))
            result['core_sha256'] = sha(Path(__file__).with_name('litert_benchmark_core.py'))
            with args.output.open('x') as handle: json.dump(result, handle, indent=2)
    finally: signal.alarm(0)
    print(json.dumps({k:result[k] for k in ('scope','reference_within_tolerance','repeat_exact')}))
if __name__ == '__main__': main()
