"""Pack independent image hypotheses as convolution groups; preserve per-image math.

Experimental fixed-batch export, never changes weights within a group or pixels.
"""
import argparse,copy,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE.parent/'exploratory'))
from live_backend import Backend

def grouped_copy(module,count):
 result=copy.deepcopy(module)
 def replace(parent):
  for name,child in list(parent.named_children()):
   if isinstance(child,nn.Conv2d):
    out=nn.Conv2d(child.in_channels*count,child.out_channels*count,child.kernel_size,child.stride,child.padding,child.dilation,child.groups*count,child.bias is not None,child.padding_mode)
    with torch.no_grad():
     out.weight.copy_(child.weight.repeat(count,1,1,1))
     if child.bias is not None:out.bias.copy_(child.bias.repeat(count))
    setattr(parent,name,out)
   elif isinstance(child,nn.BatchNorm2d):
    out=nn.BatchNorm2d(child.num_features*count,child.eps,child.momentum,child.affine,child.track_running_stats)
    with torch.no_grad():
     if child.affine:out.weight.copy_(child.weight.repeat(count));out.bias.copy_(child.bias.repeat(count))
     if child.track_running_stats:out.running_mean.copy_(child.running_mean.repeat(count));out.running_var.copy_(child.running_var.repeat(count));out.num_batches_tracked.copy_(child.num_batches_tracked)
    setattr(parent,name,out)
   else:
    if list(child.parameters(recurse=False)):raise TypeError('Unhandled parametric layer '+type(child).__name__)
    replace(child)
 replace(result);return result.eval()

class PackedBackbone(nn.Module):
 def __init__(self,backbone,count):super().__init__();self.block=grouped_copy(backbone,count)
 def forward(self,x):return self.block(x)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--upstream',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--batch',type=int,default=8);a=p.parse_args()
 if a.batch not in (2,4,8):p.error('Supported batches2,4,8')
 a.output.mkdir(parents=True,exist_ok=False)
 import pnnx
 b=Backend(a.upstream,{});original=b.model._feature_extractor._image_backbone;candidate=PackedBackbone(original,a.batch).eval();gen=torch.Generator().manual_seed(92949)
 with torch.inference_mode():
  x=torch.rand(a.batch,1,96,96,generator=gen);packed=x.reshape(1,a.batch,96,96)
  reference=original(x);actual=candidate(packed).reshape(reference.shape)
  error=float((reference-actual).abs().max());assert torch.allclose(reference,actual,atol=3e-6,rtol=3e-4),error
  pnnx.export(candidate,(a.output/'grouped.pt').as_posix(),(packed,),fp16=False)
  cases=[]
  for i in range(4):
   x=torch.rand(a.batch,1,96,96,generator=gen);expected=original(x).numpy()
   x.numpy().tofile(a.output/f'input-{i}.bin');expected.tofile(a.output/f'output-{i}.bin')
   cases.append({'index':i,'input_shape':[a.batch,96,96],'output_shape':list(expected.shape)})
  manifest={'batch':a.batch,'torch_grouping_max_abs_error':error,'cases':cases,'scope':'Synthetic independent-image packing equivalence, not end-to-end pose parity or achieved live latency.'}
  manifest['sha256']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in a.output.iterdir() if f.is_file()}
  (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
