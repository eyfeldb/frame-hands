"""Offline Frame neural-block parity/timing probe. No cameras, XR, or input injection."""
import argparse,fcntl,hashlib,json,os,platform,subprocess,time
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('models',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--iterations',type=int,default=30);p.add_argument('--threads',type=int,default=4)
 a=p.parse_args();assert 1<=a.iterations<=200 and 1<=a.threads<=8
 if a.output.exists():raise FileExistsError(a.output)
 root=Path('/home/steamos/devkit-game');locks=[]
 for path in (root/'FrameEyes/workflow.lock',root/'FrameHands/workflow.lock'):
  if path.exists():
   lock=path.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
 if subprocess.run(['systemctl','is-active','--quiet','frame-eyes-reader-brightness.service']).returncode==0:raise RuntimeError('Eye reader active')
 from workflow_guard import require_no_optical_processes
 require_no_optical_processes()
 import ncnn,onnxruntime as ort
 manifest=json.loads((a.models/'manifest.json').read_text())
 for name,digest in manifest['sha256'].items():assert hashlib.sha256((a.models/name).read_bytes()).hexdigest()==digest,name
 report={'scope':'Saved fixed neural-block inputs. Not full tracking latency, quality, power, compositor contention, or standalone acceptance.', 'platform':platform.platform(),'ncnn':ncnn.__version__,'onnxruntime':ort.__version__,'threads':a.threads,'iterations':a.iterations,'gpu_count':ncnn.get_gpu_count(),'arms':{}}
 for arm in ('ort_cpu','ncnn_cpu','ncnn_vulkan_fp32','ncnn_vulkan_fp16_storage'):
  report['arms'][arm]={}
  for name,spec in manifest['blocks'].items():
   if arm=='ort_cpu':
    opt=ort.SessionOptions();opt.intra_op_num_threads=a.threads;opt.inter_op_num_threads=1
    session=ort.InferenceSession(str(a.models/(name+'.onnx')),sess_options=opt,providers=['CPUExecutionProvider'])
    def run(x):return session.run(None,{'input':x})[0]
   else:
    net=ncnn.Net();net.opt.num_threads=a.threads;net.opt.use_vulkan_compute='vulkan' in arm
    net.opt.use_fp16_storage=arm.endswith('fp16_storage');net.opt.use_fp16_packed=arm.endswith('fp16_storage');net.opt.use_fp16_arithmetic=False;net.opt.use_bf16_storage=False
    assert net.load_param(str(a.models/(name+'.ncnn.param')))==0
    assert net.load_model(str(a.models/(name+'.ncnn.bin')))==0
    def run(x):
     with net.create_extractor() as ex:
      assert ex.input('in0',ncnn.Mat(x[0]).clone())==0
      ret,value=ex.extract('out0');assert ret==0
      return np.array(value).reshape(spec['output_shape']).copy()
   errors=[]
   for i in range(4):
    x=np.fromfile(a.models/f'{name}.input-{i}.bin',dtype='<f4').reshape(spec['input_shape'])
    expected=np.fromfile(a.models/f'{name}.output-{i}.bin',dtype='<f4').reshape(spec['output_shape'])
    actual=run(x);assert np.isfinite(actual).all();errors.append(float(np.max(abs(actual-expected))))
   x=np.fromfile(a.models/f'{name}.input-0.bin',dtype='<f4').reshape(spec['input_shape'])
   for _ in range(5):run(x)
   times=[]
   for _ in range(a.iterations):
    t=time.perf_counter();run(x);times.append((time.perf_counter()-t)*1000)
   report['arms'][arm][name]={'max_abs_errors':errors,'ms_p50_p95_max':np.percentile(times,[50,95,100]).tolist(),'finite':True}
   print(arm,name,report['arms'][arm][name],flush=True)
   if arm!='ort_cpu':net.clear()
  weights={'backbone':8,'fusion':2,'temporal':10,'regressor':10}
  key='serial_two_hand_neural_only_ms_estimate' if set(manifest['blocks'])==set(weights) else 'serial_measured_blocks_only_ms_estimate'
  report['arms'][arm][key]=sum(weights[name]*report['arms'][arm][name]['ms_p50_p95_max'][0] for name in manifest['blocks'])
 a.output.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
