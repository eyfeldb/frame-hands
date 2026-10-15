"""Native Linux ARM candidate: ncnn neural blocks plus accepted CPU geometry/state.

No CUDA requirement, IPC, camera, or driver access. Source model remains pinned.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from identity_pipeline import ImageIdentityPipeline
from chirality_evidence import ChiralityEvidence
from feature_reuse import PreparedRegressor
from live_backend import Backend

class BlockBank:
 def __init__(self,directory,mode='hybrid_fp32',threads=4):
  import ncnn
  self.ncnn=ncnn;self.directory=Path(directory);self.manifest=json.loads((self.directory/'manifest.json').read_text());self.nets={};self.counts={};self.elapsed={}
  if mode not in ('cpu','hybrid_fp32','hybrid_fp16','hybrid_fp16_arithmetic'):raise ValueError(mode)
  if mode!='cpu' and ncnn.get_gpu_count()<1:raise RuntimeError('No Vulkan GPU; refusing silent fallback')
  for name in self.manifest['blocks']:
   for suffix in ('.ncnn.param','.ncnn.bin'):
    f=self.directory/(name+suffix)
    if hashlib.sha256(f.read_bytes()).hexdigest()!=self.manifest['sha256'][f.name]:raise ValueError('Changed model '+str(f))
   net=ncnn.Net();net.opt.num_threads=threads;net.opt.use_vulkan_compute=name=='backbone' and mode!='cpu'
   half=name=='backbone' and mode in ('hybrid_fp16','hybrid_fp16_arithmetic')
   net.opt.use_fp16_storage=half;net.opt.use_fp16_packed=half;net.opt.use_fp16_arithmetic=half and mode=='hybrid_fp16_arithmetic';net.opt.use_bf16_storage=False
   if net.load_param(str(self.directory/(name+'.ncnn.param'))) or net.load_model(str(self.directory/(name+'.ncnn.bin'))):raise RuntimeError('ncnn load failed')
   self.nets[name]=net;self.counts[name]=0;self.elapsed[name]=0.
 def run(self,name,x):
  if x.device.type!='cpu' or x.dtype!=torch.float32:raise ValueError('Native blocks require CPU FP32 tensors')
  spec=self.manifest['blocks'][name]
  if list(x.shape[1:])!=spec['input_shape'][1:]:raise ValueError('Unexpected block shape')
  start=time.perf_counter();outputs=[]
  for sample in x.detach().numpy():
   with self.nets[name].create_extractor() as ex:
    if ex.input('in0',self.ncnn.Mat(np.ascontiguousarray(sample)).clone()):raise RuntimeError('ncnn input failed')
    status,out=ex.extract('out0')
    if status:raise RuntimeError('ncnn inference failed')
    value=np.array(out).reshape(spec['output_shape'][1:]).copy()
    if not np.isfinite(value).all():raise ValueError('Nonfinite native block')
    outputs.append(value)
  self.counts[name]+=len(outputs);self.elapsed[name]+=(time.perf_counter()-start)*1000
  return torch.from_numpy(np.stack(outputs))
 def bind(self,model):
  model._feature_extractor._image_backbone=NativeBlock(self,'backbone')
  model._feature_extractor._multi_view_fusion=NativeBlock(self,'fusion')
  model._temporal._temporal_module=NativeBlock(self,'temporal')
  model._regressor_k._pose_regression_layers=NativeBlock(self,'regressor')

class NativeBlock(nn.Module):
 def __init__(self,bank,name):super().__init__();self.bank=bank;self.name=name
 def forward(self,x):return self.bank.run(self.name,x)

class NativeIdentityPipeline(ImageIdentityPipeline):
 def __init__(self,upstream,blocks,mode='hybrid_fp32',threads=4,early_hypothesis=False,require_stereo_exit=False):
  if require_stereo_exit and not early_hypothesis:raise ValueError('Stereo exit requires early hypotheses')
  started=time.perf_counter();self.bank=BlockBank(blocks,mode,threads)
  evidence_type=ChiralityEvidence
  if early_hypothesis:
   from early_evidence import EarlyHypothesisEvidence
   evidence_type=EarlyHypothesisEvidence
  self.evidence=evidence_type(upstream,{},optimized=True,device='cpu')
  self.backend=Backend(upstream,{},threads=threads,association='temporal')
  if early_hypothesis:
   self.evidence.require_stereo_exit=require_stereo_exit;self.evidence.ownership=self.backend.ownership
  self.bank.bind(self.evidence.backend.model);self.bank.bind(self.backend.model)
  self.backend.prepared_regressor=PreparedRegressor(self.backend.model,self.backend.skeleton,device='cpu')
  self.ready=False;self.last_debug={};self.reset('Native model preparation')
  self.preparation_seconds=time.perf_counter()-started
