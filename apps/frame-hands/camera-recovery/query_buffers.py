"""Compile/run a metadata-only unprivileged query, holding both workflow locks."""
from contextlib import ExitStack
import fcntl
import json
from pathlib import Path
import subprocess
import time

base=Path('/home/steamos/devkit-game/FrameHands')
here=Path(__file__).resolve().parent
with ExitStack() as stack:
    for path in [base.parent/'FrameEyes/workflow.lock',base/'trial.lock']:
        f=stack.enter_context(path.open('a'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    binary=base/'experiments/camera-buffer-v1/query-buffers'
    subprocess.run(['cc','-O2','-Wall','-Wextra','-Werror',str(here/'query_buffers.c'),'-o',str(binary)],check=True,timeout=30)
    result=subprocess.run([str(binary)],check=True,capture_output=True,text=True,timeout=5)
    path=here/('query-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'.json')
    rows=[json.loads(line) for line in result.stdout.splitlines()]
    path.write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(dict(evidence=str(path),rows=rows)))
