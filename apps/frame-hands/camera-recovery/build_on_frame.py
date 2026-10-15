"""Build/test an unprivileged isolated reader candidate; never install or run capture."""
from contextlib import ExitStack
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from prepare import prepare

base=Path('/home/steamos/devkit-game/FrameHands')
here=Path(__file__).resolve().parent
output=base/'experiments/camera-buffer-v1'
with ExitStack() as stack:
    for path in [base.parent/'FrameEyes/workflow.lock',base/'trial.lock']:
        lock=stack.enter_context(path.open('a'))
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prepare(base/'upstream',output)
    common=['cc','-O2','-g','-Wall','-Wextra','-Wno-unused-parameter']
    components=[str(output/'camd/tp.c'),str(output/'camd/xrcams.c')]
    test=output/'test-buffer-pool'
    subprocess.run(common+['-I'+str(output),str(here/'test_buffer_pool.c'),*components,'-lm','-o',str(test)],check=True,timeout=60)
    regression=subprocess.run([str(test)],check=True,capture_output=True,text=True,timeout=10)
    sampler=output/'test-camera-sampler'
    subprocess.run(common+['-I'+str(output),str(base/'probe/test_camera_sampler.c'),*components,'-lm','-o',str(sampler)],check=True,timeout=60)
    sample_result=subprocess.run([str(sampler)],check=True,capture_output=True,text=True,timeout=10)
    binary=output/'frame-camera-reader-buffer-v1'
    subprocess.run(common+['-static','-o',str(binary),str(output/'camd/camd.c'),*components,'-lm'],check=True,timeout=60)
    subprocess.run([str(binary),'--help'],check=True,capture_output=True,timeout=5)
    result=dict(binary=str(binary),binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                source_manifest=json.loads((output/'source-manifest.json').read_text()),
                compiler=subprocess.check_output(['cc','--version'],text=True).splitlines()[0],
                regressions=regression.stdout,sampler_regression=sample_result.stdout,
                installed_reader_sha256=hashlib.sha256(Path('/var/lib/frame-hands/frame-camera-reader').read_bytes()).hexdigest(),
                hardware_validation='pending; no capture performed')
    (output/'build.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
