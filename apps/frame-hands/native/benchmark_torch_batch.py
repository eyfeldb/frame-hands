"""Native CPU batching feasibility; measures the unchanged pretrained backbone."""
import argparse,fcntl,json,subprocess,sys,time
from pathlib import Path
import numpy as np,torch
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--upstream',type=Path,required=True);p.add_argument('--fixture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
locks=[]
for name in ('FrameEyes','FrameHands'):
 f=(Path('/home/steamos/devkit-game')/name/'workflow.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
if subprocess.run(['systemctl','is-active','--quiet','frame-eyes-reader-brightness.service']).returncode==0:raise RuntimeError('Eye workflow active')
from workflow_guard import require_no_optical_processes
require_no_optical_processes()
sys.path.insert(0,str(Path(__file__).resolve().parent/'code/exploratory'))
from live_backend import Backend
backend=Backend(a.upstream,{});model=backend.model._feature_extractor._image_backbone
x=torch.from_numpy(np.fromfile(a.fixture/'input-0.bin',dtype='<f4').reshape(8,1,96,96));expected=np.fromfile(a.fixture/'output-0.bin',dtype='<f4').reshape(8,72,6,6)
report={'scope':'Synthetic CPU image-feature batching feasibility, not complete native tracking.','torch':torch.__version__,'arms':{}}
with torch.inference_mode():
 for threads in (1,4):
  torch.set_num_threads(threads)
  for batch in (1,8):
   value=x[:batch]
   actual=model(value).numpy();error=float(abs(actual-expected[:batch]).max())
   for _ in range(4):model(value)
   times=[]
   for _ in range(20):
    start=time.perf_counter();model(value);times.append((time.perf_counter()-start)*1000)
   report['arms'][f'threads{threads}_batch{batch}']={'max_abs_error':error,'ms_p50_p95_max':np.percentile(times,[50,95,100]).tolist()}
   print(report['arms'],flush=True)
a.output.write_text(json.dumps(report,indent=2)+'\n')
