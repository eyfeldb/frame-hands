"""Build a separate ROI source with per-frame audit flush; never replace accepted binaries."""
from contextlib import ExitStack
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

BASE=Path('/home/steamos/devkit-game/FrameHands')
ORIGINAL=BASE/'experiments/recovery-v8-joint'
DEST=BASE/'experiments/ume-mesh-source-v1'
EXPECTED='83b50f1cfed74622eda463ded86e971a87eeaa6eca1e6abece672aafe012fbb8'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    with ExitStack() as stack:
        for path in (BASE.parent/'FrameEyes/workflow.lock',BASE/'trial.lock'):
            guard=stack.enter_context(path.open('a'));fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for proc in Path('/proc').glob('[0-9]*'):
            try:name=(proc/'comm').read_text().strip()
            except (OSError,UnicodeError):continue
            if name in {'ft-camd','ft-hands','ft-ringplay','frame-camera-re'} or name.startswith(('ft-hands-','ft-handreplay')):
                raise RuntimeError('Active optical process; build deferred')
        if sha(ORIGINAL/'ft-hands-joint')!=EXPECTED:raise ValueError('Unexpected baseline tracker')
        if DEST.exists():raise FileExistsError('Refusing to overwrite an existing isolated source build')
        manifest=json.loads((ORIGINAL/'manifest.json').read_text())
        for name,digest in manifest['files'].items():
            if sha(ORIGINAL/name)!=digest:raise ValueError('Source manifest mismatch: '+name)
        for name,digest in manifest['models'].items():
            if sha(BASE/name)!=digest:raise ValueError('Model manifest mismatch: '+name)
        DEST.mkdir()
        shutil.copytree(ORIGINAL/'upstream',DEST/'upstream')
        main=DEST/'upstream/track/main.cpp'
        text=main.read_text()
        old='            frame_audit << "end\\n";'
        if text.count(old)!=1:raise ValueError('Unexpected audit writer')
        main.write_text(text.replace(old,old+'\n            frame_audit.flush(); // Live ROI consumer needs each complete set immediately.'))
        source=DEST/'upstream/track';ncnn=BASE/'deps/ncnn-install'
        binary=DEST/'ft-hands-ume-roi'
        command=['nice','-n','15','c++','-O2','-g','-Wall','-Wextra','-Wno-unused-parameter','-Wno-psabi','-std=c++17','-fopenmp','-I'+str(ncnn/'include/ncnn'),'-o',str(binary),*[str(source/(name+'.cpp')) for name in ('main','calib','nets','tracker','io','record','pinch')],str(ncnn/'lib/libncnn.a'),'-ljsoncpp','-lpthread']
        subprocess.run(command,check=True,timeout=120)
        digest=sha(binary)
        # Private launcher copies preserve all camera checks, locks and recording.
        launch=DEST/'launchers';(launch/'pc-input').mkdir(parents=True)
        shutil.copytree(BASE/'probe',launch/'probe',ignore=shutil.ignore_patterns('__pycache__'))
        recording=(BASE/'recording.py').read_text()
        if recording.count("    for path in (base/'upstream/models/ncnn').glob('*'):")!=1:raise ValueError('Unexpected recording snapshot source')
        recording=recording.replace("    for path in (base/'upstream/models/ncnn').glob('*'):","    files['launchers/trial.py']=tracker.parent/'launchers/trial.py'\n    files['launchers/run_source.py']=tracker.parent/'launchers/pc-input/run_source.py'\n    files['tools/recording.py']=tracker.parent/'launchers/recording.py'\n    for path in (base/'upstream/models/ncnn').glob('*'):")
        (launch/'recording.py').write_text(recording)
        trial=(BASE/'trial.py').read_text()
        old_path="BASE/'experiments/recovery-v8-joint/ft-hands-joint'"
        if trial.count(old_path)!=1 or trial.count(EXPECTED)!=1:raise ValueError('Unexpected trial launcher')
        trial=trial.replace(old_path,"BASE/'experiments/ume-mesh-source-v1/ft-hands-ume-roi'").replace(EXPECTED,digest)
        (launch/'trial.py').write_text(trial)
        shutil.copy2(BASE/'pc-input/run_source.py',launch/'pc-input/run_source.py')
        # Relative eye lock in the copied run_source needs the original absolute path.
        runner=launch/'pc-input/run_source.py'
        body=runner.read_text().replace("trial.parent.parent/'FrameEyes/workflow.lock'", "Path('/home/steamos/devkit-game/FrameEyes/workflow.lock')")
        body=body.replace("if ready and not sys.stdin.buffer.read(1):break", "if ready:\n                command=sys.stdin.buffer.read(1)\n                if not command or command==b'Q':break")
        runner.write_text(body)
        if sha(ORIGINAL/'ft-hands-joint')!=EXPECTED:raise ValueError('Baseline changed')
        files={str(p.relative_to(DEST)):sha(p) for p in (DEST/'upstream').rglob('*') if p.is_file()}
        result=dict(baseline_sha256=EXPECTED,binary_sha256=digest,change='Flush complete diagnostic 2D observation after each camera set; no model/identity/filter changes',
                    files=files,models=manifest['models'],compiler=subprocess.check_output(['c++','--version'],text=True).splitlines()[0],live_validated=False)
        (DEST/'manifest.json').write_text(json.dumps(dict(files=files,models=manifest['models']),indent=2)+'\n')
        (DEST/'trial-build.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)

if __name__=='__main__':main()
