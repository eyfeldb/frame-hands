"""Offline fixed-confidence and mirror-consistency screen; no runtime policy."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from train_chirality import local_images,public_images,predict,metrics
from tiny_chirality import load_classifier
from distill_encoder import sha


def screen(model,images,labels,strong):
    x=torch.from_numpy(images).cuda()
    q=predict(model,x);mirror=predict(model,x.flip(-1))
    consistent=np.abs(q+mirror-1)<=.05
    eligible=(np.maximum(q,1-q)>=.98)&(np.maximum(mirror,1-mirror)>=.98)&consistent
    wrong=(q>=.5)!=(labels>=.5)
    checked=eligible&strong
    return dict(images=len(q),strong_reference_images=int(strong.sum()),eligible_images=int(eligible.sum()),eligible_strong_reference_images=int(checked.sum()),eligible_unknown_reference_images=int((eligible&~strong).sum()),wrong_eligible_strong_reference_images=int((checked&wrong).sum()),wrong_fraction=float(wrong[checked].mean()) if checked.any() else None,unfiltered=metrics(q,labels,strong)),q,mirror,eligible,wrong


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('run','train','train-evidence','development','development-evidence','public'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if any('143912Z' in str(v) for v in (a.train,a.development)):p.error('Latest recording reserved')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    _,_,_,_,th=local_images(a.train,a.train_evidence)
    dx,dy,ds,keys,dh=local_images(a.development,a.development_evidence)
    public,_=public_images(a.public,th|dh)
    px=public['validation'];px=np.concatenate((px,px[:,:,:,::-1]))
    py=np.concatenate((np.zeros(len(px)//2),np.ones(len(px)//2)))
    model=load_classifier(a.run/'best.torch')
    local,q,m,eligible,wrong=screen(model,dx,dy,ds)
    public_result,*_=screen(model,px,py,np.ones(len(px),bool))
    with (a.run/'mirror-screen-trace.jsonl').open('w') as out:
        for i,key in enumerate(keys):out.write(json.dumps(dict(**key,q=float(q[i]),mirrored_q=float(m[i]),eligible=bool(eligible[i]),strong_reference=bool(ds[i]),teacher_disagreement=bool(wrong[i])))+'\n')
    result=dict(scope='Fixed mirror-consistency screen only; no overlap/detector/temporal policy, no physical labels for local recording, no native timing or integration.',checkpoint_sha256=sha(a.run/'best.torch'),source_sha256=sha(Path(__file__)),policy=dict(confidence_both_views=.98,mirror_complement_error=.05),local_development=local,public_sequence_validation=public_result)
    (a.run/'mirror-screen.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
