"""Matched image-only/target-conditioned identity training; full pose model unchanged."""
import argparse,copy,json,math,random,time
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from target_chirality import ARCH,TargetChirality,condition
from distill_encoder import sha
from distill_pose_encoder import augment
from train_chirality import metrics


@torch.no_grad()
def evaluate(model,x,y,strong,conditioned):
    model.eval();values=[];mirrors=[]
    for chunk in x.split(128):
        if not conditioned:chunk=torch.cat((chunk[:,:1],torch.zeros_like(chunk[:,1:])),dim=1)
        values.append(model(chunk).sigmoid().cpu().numpy());mirrors.append(model(chunk.flip(-1)).sigmoid().cpu().numpy())
    q=np.concatenate(values);qm=np.concatenate(mirrors)
    eligible=(np.maximum(q,1-q)>=.98)&(np.maximum(qm,1-qm)>=.98)&(np.abs(q+qm-1)<=.05)
    selected=eligible&strong;wrong=(q>=.5)!=(y>=.5)
    return dict(**metrics(q,y,strong),mirror_screen=dict(eligible=int(eligible.sum()),eligible_strong=int(selected.sum()),unknown_reference=int((eligible&~strong).sum()),wrong_strong=int((wrong&selected).sum()),wrong_fraction=float(wrong[selected].mean()) if selected.any() else None))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False);seed=3295;torch.set_num_threads(4);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed);random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    manifest=json.loads((a.data/'manifest.json').read_text());data={}
    for name,spec in manifest['datasets'].items():
        path=a.data/(name+'.npz')
        if sha(path)!=spec['archive_sha256']:raise ValueError('Changed conditioning data')
        with np.load(path,allow_pickle=False) as archive:data[name]={k:archive[k] for k in archive.files}
    def device(name,key):return torch.from_numpy(data[name][key]).cuda()
    strong=data['train']['strong'];x=device('train','images')[strong];uv=device('train','points')[strong];y=device('train','labels')[strong]
    px=device('public_train','images');puv=device('public_train','points')
    dx=condition(device('development','images'),device('development','points'));dy=data['development']['labels'];ds=data['development']['strong']
    vx=condition(device('public_validation','images'),device('public_validation','points'));vx=torch.cat((vx,vx.flip(-1)))
    vy=np.concatenate((np.zeros(len(vx)//2),np.ones(len(vx)//2)));vs=np.ones(len(vx),bool)
    initial=TargetChirality().cuda();models={k:copy.deepcopy(initial) for k in ('image_only','target_conditioned')}
    opts={k:torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001) for k,model in models.items()}
    sched={k:torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=60,eta_min=.00003) for k,opt in opts.items()}
    steps=max(100,math.ceil(max(len(x),len(px))/32));history=[];best={k:float('inf') for k in models};selected={}
    g=torch.Generator(device='cuda').manual_seed(seed);started=time.perf_counter()
    report=dict(scope=__doc__,seed=seed,epochs=60,steps_per_epoch=steps,batch_size=64,learning_rate=.002,parameters=sum(p.numel() for p in initial.parameters()),architecture=ARCH,data_manifest_sha256=sha(a.data/'manifest.json'),torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),tf32=False,selection='Minimum strong-local-development BCE at epoch0 and every5epochs; public validation excluded from selection; latest local holdout unused.',augmentation='Equal32local/32public; same random image/rotation/flip augmentation for both arms; target point jitter sigma2px/drop10%; only target arm receives wrist/all-points Gaussian maps sigma3px.',source_sha256={name:sha(Path(__file__).parent/name) for name in ('target_chirality.py','train_target_chirality.py')})
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    for epoch in range(61):
        losses={k:0. for k in models}
        if epoch:
            for model in models.values():model.train()
            for _ in range(steps):
                li=torch.randint(len(x),(32,),device='cuda',generator=g);pi=torch.randint(len(px),(32,),device='cuda',generator=g)
                images=torch.cat((x[li],px[pi]));points=torch.cat((uv[li],puv[pi]));target=torch.cat((y[li],torch.zeros(32,device='cuda')))
                value=condition(augment(images,g),points,g)
                flip=torch.rand(64,device='cuda',generator=g)<.5
                value=torch.where(flip[:,None,None,None],value.flip(-1),value);target=torch.where(flip,1-target,target)
                theta=torch.zeros((64,2,3),device='cuda');angle=(torch.rand(64,device='cuda',generator=g)-.5)*(math.pi/6)
                theta[:,0,0]=theta[:,1,1]=angle.cos();theta[:,0,1]=-angle.sin();theta[:,1,0]=angle.sin()
                value=F.grid_sample(value,F.affine_grid(theta,value.shape,align_corners=False),align_corners=False)
                for name,model in models.items():
                    sample=value if name=='target_conditioned' else torch.cat((value[:,:1],torch.zeros_like(value[:,1:])),dim=1)
                    opt=opts[name];opt.zero_grad(set_to_none=True);loss=F.binary_cross_entropy_with_logits(model(sample),target)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses[name]+=loss.item()
            for scheduler in sched.values():scheduler.step()
        row=dict(epoch=epoch,seconds=time.perf_counter()-started,loss={k:v/steps for k,v in losses.items()})
        if epoch%5==0:
            for name,model in models.items():
                conditioned=name=='target_conditioned';local=evaluate(model,dx,dy,ds,conditioned);public=evaluate(model,vx,vy,vs,conditioned)
                row[name]=dict(local=local,public=public)
                if local['cross_entropy']<best[name]:
                    best[name]=local['cross_entropy'];selected[name]=row[name]|dict(epoch=epoch)
                    torch.save(dict(architecture=ARCH,conditioned=conditioned,epoch=epoch,state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}),a.output/(name+'.torch'))
        history.append(row);(a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n');print(json.dumps(row),flush=True)
    summary=dict(scope='Offline teacher-label/public validation evidence, not native speed or live identity validation.',selected=selected,checkpoint_sha256={k:sha(a.output/(k+'.torch')) for k in models},seconds=time.perf_counter()-started)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
