"""Paired numerical and synthetic-pose microbenchmark for fixed-hand contraction."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE.parent/'exploratory'))
from live_backend import Backend
from cached_skinning import FixedHandSkinner


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('upstream','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--native-guard',action='store_true');a=p.parse_args()
    from contextlib import nullcontext
    if a.native_guard:
        from workflow_guard import native_workflow
        guard=native_workflow()
    else:guard=nullcontext()
    with guard:
        b=Backend(a.upstream,{},threads=1);cached=FixedHandSkinner(b.hand)
        g=torch.Generator().manual_seed(9351);rows=[]
        with torch.inference_mode():
            for batch in (1,2,8,16):
                angles=(torch.rand(batch,22,generator=g)-.5)*2
                wrist=torch.eye(4).repeat(batch,1,1)
                from rotations import so3_exp_map
                wrist[:,:3,:3]=so3_exp_map(torch.randn(batch,3,generator=g))
                wrist[:,:3,3]=(torch.rand(batch,3,generator=g)-.5)*6000
                wrist[1::2,:,0]*=-1
                hand=b.hand._replace(**{k:v.unsqueeze(0).expand(batch,*v.shape) for k,v in b.hand._asdict().items() if isinstance(v,torch.Tensor)})
                old=lambda:b.landmarks(hand,angles,wrist)
                new=lambda:cached.landmarks(hand,angles,wrist)
                error=float((old()-new()).abs().max())
                if error>.01:raise AssertionError(('landmark',batch,error))
                times={'old':[],'new':[]}
                for _ in range(10):old();new()
                for i in range(100):
                    for name,call in ((('old',old),('new',new)) if i%2==0 else (('new',new),('old',old))):
                        start=time.perf_counter();call();times[name].append((time.perf_counter()-start)*1000)
                rows.append(dict(kind='landmarks',batch=batch,max_error_mm=error,median_ms={k:float(np.median(v)) for k,v in times.items()}))
            for side in (0,1):
                old=lambda:b.mesh(angles[0],wrist[0],side)
                new=lambda:cached.mesh(angles[0],wrist[0],side)
                error=float(np.abs(old()-new()).max())
                if error>.01:raise AssertionError(('mesh',side,error))
                times={'old':[],'new':[]}
                for _ in range(10):old();new()
                for i in range(100):
                    for name,call in ((('old',old),('new',new)) if i%2==0 else (('new',new),('old',old))):
                        start=time.perf_counter();call();times[name].append((time.perf_counter()-start)*1000)
                rows.append(dict(kind='mesh',side=side,max_error_mm=error,median_ms={k:float(np.median(v)) for k,v in times.items()}))
        changed=b.hand._replace(joint_rest_positions=b.hand.joint_rest_positions.clone())
        try:cached.landmarks(changed,angles,wrist)
        except ValueError:pass
        else:raise AssertionError('Changed skeleton not rejected')
        result=dict(scope='Synthetic CPU FP32 contraction parity and alternating microbenchmark; not live tracking.',torch=torch.__version__,rows=rows,source_sha256={name:hashlib.sha256((BASE/name).read_bytes()).hexdigest() for name in ('cached_skinning.py','benchmark_skinning.py')})
        a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
