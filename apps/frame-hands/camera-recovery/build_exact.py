"""Isolated v3 build and numerical tests; no privileges, capture or installation."""
from contextlib import ExitStack
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
from prepare_exact import prepare_exact
base=Path('/home/steamos/devkit-game/FrameHands')
here=Path(__file__).resolve().parent
out=base/'experiments/camera-exact-v3'
with ExitStack() as stack:
    for path in (base.parent/'FrameEyes/workflow.lock',base/'trial.lock'):
        f=stack.enter_context(path.open('a'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prepare_exact(base/'upstream',out)
    common=['cc','-O2','-g','-Wall','-Wextra','-Wno-unused-parameter','-Wno-unused-function']
    parts=[str(out/'camd/tp.c'),str(out/'camd/xrcams.c')]
    results={}
    for name,source in [('exact-discovery',here/'test_exact_discovery.c'),('direct-queue',here/'test_direct_queue.c'),('direct-reader',here/'test_direct_reader.c'),('buffer-pool',here/'test_buffer_pool.c'),('camera-sampler',base/'probe/test_camera_sampler.c')]:
        binary=out/('test-'+name)
        subprocess.run(common+['-I'+str(out),str(source),*([] if name=='exact-discovery' else parts),'-lm','-o',str(binary)],check=True,timeout=60)
        results[name]=subprocess.check_output([str(binary)],text=True,timeout=10)
    binary=out/'frame-camera-reader-exact-v3'
    subprocess.run(common+['-static','-o',str(binary),str(out/'camd/camd.c'),*parts,'-lm'],check=True,timeout=60)
    subprocess.run([str(binary),'--help'],check=True,capture_output=True,timeout=5)
    result=dict(binary=str(binary),binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),source_manifest=json.loads((out/'source-manifest.json').read_text()),compiler=subprocess.check_output(['cc','--version'],text=True).splitlines()[0],regressions=results,hardware_validation='No capture or installation; metadata-only discovery pending')
    (out/'build.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
