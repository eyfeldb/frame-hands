"""Saved-crop LiteRT benchmark core. No camera, XR, or tracking runtime access."""
import hashlib, json, platform, sys, time
from pathlib import Path

OUTPUTS = {'Identity': ('screen', (21, 3)), 'Identity_1': ('presence', ()),
           'Identity_2': ('right', ()), 'Identity_3': ('world', (21, 3))}
TOLERANCES = {'screen': 0.05, 'world': 0.0001, 'presence': 0.0001, 'right': 0.0001}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify_bundle(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path) != expected:
            raise RuntimeError('Bundle integrity failure: ' + name)
    return manifest

def require_native():
    if sys.platform != 'linux' or platform.machine() != 'aarch64':
        raise RuntimeError('Requires native Linux aarch64; no host bypass')
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('Pinned runtime requires Python 3.12')
    libc, version = platform.libc_ver()
    if libc != 'glibc' or tuple(map(int, version.split('.'))) < (2, 34):
        raise RuntimeError('Pinned runtime requires glibc >= 2.34 (audited package maximum)')

def telemetry():
    result = {'platform': platform.platform(), 'python': platform.python_version(),
              'libc': platform.libc_ver()}
    if hasattr(__import__('os'), 'sched_getaffinity'):
        result['affinity'] = sorted(__import__('os').sched_getaffinity(0))
    for name in ('/etc/os-release',):
        try: result[name] = Path(name).read_text()
        except OSError: pass
    result['sysfs'] = {}
    for pattern in ('/sys/class/thermal/thermal_zone*/temp',
                    '/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq'):
        import glob
        for name in glob.glob(pattern):
            try: result['sysfs'][name] = Path(name).read_text().strip()
            except OSError: pass
    return result

def benchmark(root, runtime, scope, threads=(1, 3), seconds=120):
    manifest = verify_bundle(root)
    sys.path.insert(0, str(runtime.resolve()))
    import numpy as np
    from ai_edge_litert.interpreter import Interpreter
    import ai_edge_litert.interpreter as module
    from importlib.metadata import version
    if not Path(module.__file__).resolve().is_relative_to(runtime.resolve()):
        raise RuntimeError('Interpreter resolved outside specified runtime')
    if not Path(np.__file__).resolve().is_relative_to(runtime.resolve()):
        raise RuntimeError('NumPy resolved outside specified runtime')
    if version('ai-edge-litert') != '2.2.0' or np.__version__ != '2.5.3':
        raise RuntimeError('Unexpected runtime versions')
    before = telemetry(); deadline = time.monotonic() + seconds
    def check_time():
        if time.monotonic() > deadline: raise TimeoutError('Benchmark deadline exceeded')
    def stats(values):
        return {key: float(value) for key, value in zip(('median', 'p95', 'max'),
                (np.median(values), np.percentile(values, 95), np.max(values)))}
    inputs = []; preparation_ms = []
    for record in manifest['records']:
        check_time(); start = time.perf_counter()
        pixels = np.fromfile(root / record['file'], dtype=np.uint8).reshape(224, 224)
        inputs.append(np.repeat((pixels.astype(np.float32) * np.float32(1/255))[None, :, :, None], 3, axis=3))
        preparation_ms.append((time.perf_counter() - start) * 1000)
    passes = []; first = {}; repeat_delta = {}; reference_delta = {}
    for workers in threads:
        for pass_index, name in enumerate(('lite', 'full', 'full', 'lite')):
            check_time()
            interpreter = Interpreter(model_path=str(root / 'models' / (name + '.tflite')), num_threads=workers)
            interpreter.allocate_tensors()
            ins = interpreter.get_input_details(); outs = interpreter.get_output_details()
            if len(ins) != 1 or ins[0]['shape'].tolist() != [1,224,224,3] or ins[0]['dtype'] != np.float32:
                raise RuntimeError('Unexpected input schema')
            if {d['name']: d['shape'].tolist() for d in outs} != {'Identity':[1,63], 'Identity_1':[1,1], 'Identity_2':[1,1], 'Identity_3':[1,63]}:
                raise RuntimeError('Unexpected output schema')
            def invoke(x):
                check_time(); start = time.perf_counter()
                interpreter.set_tensor(ins[0]['index'], x); interpreter.invoke()
                outputs = {d['name']: interpreter.get_tensor(d['index']).copy() for d in outs}
                elapsed = (time.perf_counter() - start) * 1000
                values = {}
                for output, (key, shape) in OUTPUTS.items():
                    array = outputs[output].reshape(shape)
                    if not np.isfinite(array).all(): raise RuntimeError('Nonfinite model output')
                    values[key] = array.tolist()
                return values, elapsed
            for _ in range(5): invoke(inputs[0])
            timings = []; predictions = []
            label = str(workers) + ':' + name
            reference_delta.setdefault(label, {key: 0.0 for key in TOLERANCES})
            for record, x in zip(manifest['records'], inputs):
                prediction, elapsed = invoke(x); timings.append(elapsed); predictions.append(prediction)
                for key in TOLERANCES:
                    delta = float(np.max(np.abs(np.asarray(prediction[key]) - np.asarray(record['reference'][name][key]))))
                    reference_delta[label][key] = max(reference_delta[label][key], delta)
            if label not in first: first[label] = predictions
            else:
                repeat_delta[label] = {key: max(float(np.max(np.abs(np.asarray(a[key])-np.asarray(b[key])))) for a,b in zip(first[label], predictions)) for key in TOLERANCES}
            passes.append({'threads':workers, 'model':name, 'pass':pass_index,
                           'invoke_and_copy_ms':stats(timings), 'timings_ms':timings, 'predictions':predictions})
            del interpreter
    return {'scope':scope, 'environment_before':before, 'environment_after':telemetry(),
            'litert':version('ai-edge-litert'), 'numpy':np.__version__, 'backend':'CPU compatibility Interpreter (default XNNPACK)',
            'manifest_sha256':sha(root/'manifest.json'), 'crop_indices':[r['index'] for r in manifest['records']],
            'preparation_including_file_read_ms':stats(preparation_ms), 'passes':passes,
            'cross_arch_reference_max_abs':reference_delta, 'repeat_max_abs':repeat_delta,
            'tolerances':TOLERANCES,
            'reference_within_tolerance':all(v <= TOLERANCES[k] for d in reference_delta.values() for k,v in d.items()),
            'repeat_exact':all(v == 0 for d in repeat_delta.values() for v in d.values()),
            'limitations':['Saved selected crops; not complete tracking or accuracy.',
                           'CPU timings exclude detection, crop extraction, stereo fitting and presentation.',
                           'Reference agreement is numerical compatibility, not physical truth.']}
