"""Paired serial/grouped Vulkan backbone probe on saved tensors; no camera or XR."""
import argparse,fcntl,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
import ncnn

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--grouped',type=Path,required=True);p.add_argument('--original',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 locks=[]
 for name in ('FrameEyes','FrameHands'):
  path=Path('/home/steamos/devkit-game')/name/'workflow.lock'
  lock=path.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
 if subprocess.run(['systemctl','is-active','--quiet','frame-eyes-reader-brightness.service']).returncode==0:raise RuntimeError('Eye workflow active')
 from workflow_guard import require_no_optical_processes
 require_no_optical_processes()
 manifest=json.loads((a.grouped/'manifest.json').read_text());batch=manifest['batch']
 for name,sha in manifest['sha256'].items():assert hashlib.sha256((a.grouped/name).read_bytes()).hexdigest()==sha
 nets={}
 for name,path in [('serial',a.original/'backbone'),('grouped',a.grouped/'grouped')]:
  net=ncnn.Net();net.opt.num_threads=1;net.opt.use_vulkan_compute=True;net.opt.use_fp16_packed=False;net.opt.use_fp16_storage=False;net.opt.use_fp16_arithmetic=False
  assert net.load_param(str(path)+'.ncnn.param')==0;assert net.load_model(str(path)+'.ncnn.bin')==0;nets[name]=net
 def once(name,x):
  result=[]
  for sample in ([x] if name=='grouped' else [v[None] for v in x]):
   with nets[name].create_extractor() as ex:
    assert ex.input('in0',ncnn.Mat(np.ascontiguousarray(sample)).clone())==0
    ret,value=ex.extract('out0');assert ret==0;result.append(np.array(value).reshape(-1,72,6,6).copy())
  return np.concatenate(result)
 errors={k:[] for k in nets}
 for i in range(4):
  x=np.fromfile(a.grouped/f'input-{i}.bin',dtype='<f4').reshape(batch,96,96);expected=np.fromfile(a.grouped/f'output-{i}.bin',dtype='<f4').reshape(batch,72,6,6)
  for name in nets:
   actual=once(name,x);assert np.isfinite(actual).all();errors[name].append(float(abs(actual-expected).max()))
 for _ in range(3):
  for name in nets:once(name,x)
 times={k:[] for k in nets}
 for i in range(20):
  for name in (('serial','grouped') if i%2==0 else ('grouped','serial')):
   start=time.perf_counter();once(name,x);times[name].append((time.perf_counter()-start)*1000)
 report={'batch':batch,'scope':'Alternating paired synthetic backbone execution, not full pose or live latency.','max_abs_errors':errors,'times_ms':times,'ms_p50_p95_max':{k:np.percentile(v,[50,95,100]).tolist() for k,v in times.items()}}
 a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
