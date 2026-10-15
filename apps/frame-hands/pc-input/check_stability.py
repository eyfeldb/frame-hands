"""Run synthetic and recorded-pose checks; export metrics/hashes only."""
import hashlib
import json
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parent
workspace=root.parents[2]
binary=root/'build/test-stability.exe'
report={'scope':'PC diagnostic candidate. Replay of published poses, not fresh inference, ground truth, hardware cadence or user acceptance.',
        'sources':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['stability.h','test_stability.cpp']},
        'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'recordings':{}}
for clip in ['20261003T190304Z','20261003T182412Z','20261003T203217Z']:
    path=workspace/'.local/hand-tracking/recovery-v3-results'/f'{clip}-filter.poses'
    result=subprocess.run([str(binary),str(path)],check=True,capture_output=True,text=True)
    report['recordings'][clip]={'pose_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'result':result.stdout.splitlines()}
(root/'evidence/stability-preview.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
