"""Export pinned UmeTrack neural blocks; no cameras, weights committed, or live changes."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'exploratory'))
from live_backend import Backend

class NamedBlock(torch.nn.Module):
 def __init__(self,block):
  super().__init__();self.block=block
 def forward(self,x):return self.block(x)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--upstream',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 import pnnx,onnxruntime as ort
 b=Backend(a.upstream,{});m=b.model;g=torch.Generator().manual_seed(7153)
 with torch.inference_mode():
  skel=m._skeleton_enc(b.skeleton.joint_rotation_axes,b.skeleton.joint_rest_positions)
  image=torch.rand(1,1,96,96,generator=g)
  feat=m._feature_extractor._image_backbone(image)
  blocks={'backbone':(m._feature_extractor._image_backbone,image),
   'fusion':(m._feature_extractor._multi_view_fusion,torch.cat([feat,feat],1)),
   'temporal':(m._temporal._temporal_module,torch.cat([torch.zeros(1,m._temporal._nc_memory,*feat.shape[-2:]),feat],1)),
   'regressor':(m._regressor_k._pose_regression_layers,torch.cat([feat,skel.expand(1,*skel.shape[1:])],1))}
  manifest={'scope':'Neural-block export and fixed-tensor equivalence only; not complete tracking or a performance/quality claim.',
   'weights_sha256':hashlib.sha256((a.upstream/'pretrained_models/pretrained_weights.torch').read_bytes()).hexdigest(),
   'torch':torch.__version__,'ort':ort.__version__,'pnnx':'20260526','skeleton_shape':list(skel.shape),'blocks':{},
   'spatial_ftl_ratio':m._feature_extractor._ftl_ratio,'temporal_ftl_ratio':m._temporal._temporal_ftl_ratio,
   'use_unscaled_as_canonical':m._feature_extractor._use_unscaled_as_canonical,'memory_channels':m._temporal._nc_memory,
   'output_ranges':m._regressor_k._output_index_ranges}
  np.save(a.output/'skeleton.npy',skel.numpy());np.save(a.output/'wrist_points.npy',m._regressor_k._left_wrist_sample_points.numpy())
  for name,(module,x) in blocks.items():
   module.eval();path=a.output/name
   torch.onnx.export(module,(x,),str(path)+'.onnx',dynamo=False,opset_version=17,input_names=['input'],output_names=['output'])
   pnnx.export(NamedBlock(module),path.as_posix()+'.pt',(x,),fp16=False)
   session=ort.InferenceSession(str(path)+'.onnx',providers=['CPUExecutionProvider'])
   cases=[]
   for i in range(4):
    value=x if i==0 else torch.rand(x.shape,generator=g)* (1 if name=='backbone' else 2)- (0 if name=='backbone' else 1)
    target=module(value).numpy();actual=session.run(None,{'input':value.numpy()})[0]
    err=float(np.max(np.abs(target-actual)));assert np.allclose(target,actual,atol=3e-5,rtol=3e-4),(name,err)
    value.numpy().astype('<f4').tofile(str(path)+f'.input-{i}.bin');target.astype('<f4').tofile(str(path)+f'.output-{i}.bin')
    cases.append({'max_abs_error':err})
   manifest['blocks'][name]={'input_shape':list(x.shape),'output_shape':list(target.shape),'cases':cases}
   print(name,manifest['blocks'][name],flush=True)
  manifest['sha256']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in a.output.iterdir() if f.is_file()}
  (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
