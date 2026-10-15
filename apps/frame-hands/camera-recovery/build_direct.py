"""Build exact-queue reader beside both predecessors; no installation or capture."""
from contextlib import ExitStack
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
from prepare_direct import prepare_direct

base=Path('/home/steamos/devkit-game/FrameHands')
here=Path(__file__).resolve().parent
output=base/'experiments/camera-direct-v2'
with ExitStack() as stack:
    for path in [base.parent/'FrameEyes/workflow.lock',base/'trial.lock']:
        f=stack.enter_context(path.open('a'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prepare_direct(base/'upstream',output)
    common=['cc','-O2','-g','-Wall','-Wextra','-Wno-unused-parameter']
    components=[str(output/'camd/tp.c'),str(output/'camd/xrcams.c')]
    results={}
    for name,source in [('direct-queue',here/'test_direct_queue.c'),
                        ('direct-reader',here/'test_direct_reader.c'),
                        ('buffer-pool',here/'test_buffer_pool.c'),
                        ('camera-sampler',base/'probe/test_camera_sampler.c')]:
        binary=output/('test-'+name)
        subprocess.run(common+['-I'+str(output),str(source),*components,'-lm','-o',str(binary)],check=True,timeout=60)
        results[name]=subprocess.check_output([str(binary)],text=True,timeout=10)
    binary=output/'frame-camera-reader-direct-v2'
    subprocess.run(common+['-static','-o',str(binary),str(output/'camd/camd.c'),*components,'-lm'],check=True,timeout=60)
    subprocess.run([str(binary),'--help'],check=True,capture_output=True,timeout=5)
    result=dict(binary=str(binary),binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                source_manifest=json.loads((output/'source-manifest.json').read_text()),
                compiler=subprocess.check_output(['cc','--version'],text=True).splitlines()[0],
                regressions=results,installed_reader_sha256=hashlib.sha256(Path('/var/lib/frame-hands/frame-camera-reader').read_bytes()).hexdigest(),
                hardware_validation='Queue metadata query succeeded; candidate capture pending')
    (output/'build.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
