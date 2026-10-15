"""Controlled local-only versus mixed public/local feature distillation."""
import argparse,hashlib,json,random,sys,time
from pathlib import Path
import numpy as np
import torch
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'exploratory'))
from live_backend import Backend
from student_encoder import ARCH,load_student
from distill_encoder import sha
from distill_pose_encoder import MonoHead,collect,augment,get_targets,evaluate,verify_head


def load_public(directory,excluded):
    manifest=json.loads((directory/'manifest.json').read_text())
    if not manifest['training_only'] or manifest['commit']!='355df534a1c761ac416427925485b311f22ce50e':raise ValueError('Unexpected public split/version')
    images=[];seen=set();overlap=0
    for row in manifest['records']:
        if not row['sequence'].endswith('/training'):raise ValueError('Non-training public record')
        path=directory/row['directory']/'training.npz'
        if sha(path)!=row['archive_sha256']:raise ValueError('Changed public archive')
        with np.load(path,allow_pickle=False) as archive:
            crops=archive['images']
            if crops.ndim!=4 or crops.shape[-2:]!=(96,96) or not np.isfinite(crops).all() or crops.min()<0 or crops.max()>1:raise ValueError('Invalid crop archive')
            for image in crops.reshape(-1,96,96):
                for flip in (False,True):
                    value=np.ascontiguousarray(image[:,::-1] if flip else image,dtype=np.float32)
                    quantized=np.rint(value*255).astype(np.uint8)
                    digest=hashlib.sha256(quantized.tobytes()).hexdigest()
                    if digest in excluded:overlap+=1;continue
                    if digest in seen:continue
                    seen.add(digest);images.append(value)
    if overlap:raise ValueError('Public images duplicate local split after uint8 comparison')
    if not images:raise ValueError('No public training crops')
    return np.stack(images)[:,None],dict(manifest_sha256=sha(directory/'manifest.json'),unique_crops=len(images),cross_split_duplicates=overlap,record_count=manifest['record_count'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('upstream','train','development','public','checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--epochs',type=int,default=40);p.add_argument('--batch',type=int,default=64)
    a=p.parse_args()
    if a.train.resolve()==a.development.resolve() or any('143912Z' in str(v) for v in (a.train,a.development)):p.error('Invalid local split or reserved held-out recording')
    if a.batch<2 or a.batch%2 or a.epochs<1:p.error('Positive epochs and even batch required')
    a.output.mkdir(parents=True,exist_ok=False)
    seed=4087
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    backend=Backend(a.upstream,{})
    head=MonoHead(backend)
    train,k,rows,hashes=collect(a.train,3);dev,dk,drows,dhashes=collect(a.development,1)
    if hashes & dhashes:raise ValueError('Local split duplicates')
    public,public_info=load_public(a.public,hashes|dhashes)
    x=torch.from_numpy(train).cuda().float()/255.;px=torch.from_numpy(public).cuda()
    k=torch.from_numpy(k).cuda();dx=torch.from_numpy(dev).cuda().float()/255.;dk=torch.from_numpy(dk).cuda()
    parity=verify_head(head,backend,x[:4],k[:4])
    y,raw=get_targets(head,x,k,a.batch);dy,draw=get_targets(head,dx,dk,a.batch)
    scales=(y.square().mean().clamp_min(1e-12),raw.std(0).clamp_min(.02))
    del y,raw,train,public,dev
    arms={name:load_student(a.upstream,a.checkpoint) for name in ('local_control','mixed_public')}
    opts={name:torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=.0001) for name,model in arms.items()}
    schedulers={name:torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs,eta_min=.00001) for name,opt in opts.items()}
    steps=(len(x)+a.batch-1)//a.batch
    manifest=dict(scope=__doc__,seed=seed,epochs=a.epochs,batch=a.batch,steps_per_epoch=steps,architecture=ARCH,
        initialization_sha256=sha(a.checkpoint),teacher_weights_sha256=sha(a.upstream/'pretrained_models/pretrained_weights.torch'),train_recording=a.train.name,development_recording=a.development.name,
        local_train_images=len(x),development_images=len(dx),public=public_info,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),tf32=False,
        train_transport_sha256=sha(a.train/'transport.bin'),development_transport_sha256=sha(a.development/'transport.bin'),
        loss='Feature MSE divided by fixed LOCAL training teacher energy; no public pose labels used as targets',
        arms='Control gets64 local crops; mixed gets the same first32 local crops plus32 public crops. Equal optimizer steps and batch sizes. All encoder layers trainable; BN running statistics update normally.',
        augmentation='Same photometric function as prior ablation: gain,offset,box3blur,noise; both mirrored hypotheses for all sources',
        checkpoint_selection='Minimum local-development single-view zero-memory standardized pose loss, including initial checkpoint and evaluated every5epochs',
        screening_gate='Development matched mean vertex median<=3mm,p95<=10mm,side/view changes<=1%; no held-out evaluation unless passed',
        head_parity=parity,source_sha256={name:sha(BASE/name) for name in ('distill_mixed_encoder.py','distill_pose_encoder.py','student_encoder.py')})
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest),flush=True)
    initial=evaluate(arms['local_control'],head,dx,dk,dy,draw,scales,a.batch)
    print('initial',initial,flush=True)
    history=[];best={name:initial['pose_standardized_mse'] for name in arms};best_epoch={name:0 for name in arms}
    for name,model in arms.items():
        torch.save(dict(architecture=ARCH,epoch=0,state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}),a.output/(name+'-best.torch'))
    g=torch.Generator(device='cuda').manual_seed(seed)
    started=time.perf_counter()
    for epoch in range(1,a.epochs+1):
        for model in arms.values():model.train()
        totals={name:0. for name in arms}
        for step in range(steps):
            local_ids=torch.randint(len(x),(a.batch,),device='cuda',generator=g)
            public_ids=torch.randint(len(px),(a.batch//2,),device='cuda',generator=g)
            value=augment(torch.cat((x[local_ids],px[public_ids])),g)
            with torch.no_grad():target=head.model._feature_extractor._image_backbone(value)
            samples={'local_control':(value[:a.batch],target[:a.batch]),'mixed_public':(torch.cat((value[:a.batch//2],value[a.batch:])),torch.cat((target[:a.batch//2],target[a.batch:])))}
            for name,model in arms.items():
                xx,yy=samples[name];opt=opts[name];opt.zero_grad(set_to_none=True)
                loss=(model(xx)-yy).square().mean()/scales[0]
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step();totals[name]+=float(loss.detach())
        row=dict(epoch=epoch,train={name:v/steps for name,v in totals.items()},seconds=time.perf_counter()-started)
        if epoch%5==0 or epoch==a.epochs:
            row['development']={name:evaluate(model,head,dx,dk,dy,draw,scales,a.batch) for name,model in arms.items()}
            for name,model in arms.items():
                score=row['development'][name]['pose_standardized_mse']
                if score<best[name]:
                    best[name]=score;best_epoch[name]=epoch
                    torch.save(dict(architecture=ARCH,epoch=epoch,state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}),a.output/(name+'-best.torch'))
        for scheduler in schedulers.values():scheduler.step()
        history.append(row);(a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n');print(json.dumps(row),flush=True)
    if any(p.grad is not None for p in head.parameters()):raise ValueError('Teacher accumulated gradients')
    result=dict(scope='Controlled diversity experiment; full causal replay still required. No native runtime change.',initial_development=initial,best_epoch=best_epoch,best_development_pose_loss=best,
        seconds=time.perf_counter()-started,checkpoints_sha256={name:sha(a.output/(name+'-best.torch')) for name in arms})
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
