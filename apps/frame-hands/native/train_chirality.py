"""Train a separate cheap handedness classifier; preserve full pose model."""
import argparse,hashlib,json,math,random,sys,time
from pathlib import Path
import cv2,numpy as np,torch
import torch.nn.functional as F
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE.parent/'exploratory'))
from replay_mesh_trial import packets
from distill_encoder import sha
from distill_pose_encoder import augment
from tiny_chirality import ARCH,TinyChirality


def local_images(session,evidence_path):
    evidence={r['stamp']:{(v['camera'],v['source_id']):v for v in r['views']} for line in evidence_path.open() if (r:=json.loads(line))}
    images=[];labels=[];strong=[];keys=[];hashes=set()
    for meta,blobs in packets(session/'transport.bin'):
        if meta.get('waiting') or meta['stamp'] not in evidence:continue
        for crop,blob in zip(meta['crops'],blobs):
            row=evidence[meta['stamp']].get((crop['camera'],crop['id']))
            if row is None:continue
            image=cv2.imdecode(np.frombuffer(blob,np.uint8),0)
            if image is None or image.shape!=(96,96):raise ValueError('Invalid local crop')
            images.append(image);labels.append(int(row['right']>=.5));strong.append(bool(row['agreement'] and max(row['right'],1-row['right'])>=.85))
            keys.append(dict(stamp=meta['stamp'],camera=crop['camera'],source_id=crop['id'],teacher_right=row['right'],teacher_agreement=row['agreement']))
            hashes.add(hashlib.sha256(image.tobytes()).hexdigest())
            hashes.add(hashlib.sha256(np.ascontiguousarray(image[:,::-1]).tobytes()).hexdigest())
    return np.stack(images)[:,None].astype(np.float32)/255.,np.asarray(labels,np.float32),np.asarray(strong,bool),keys,hashes


def public_images(root,excluded):
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['commit']!='355df534a1c761ac416427925485b311f22ce50e' or not manifest['training_only']:raise ValueError('Changed public split')
    groups={}
    for row in manifest['records']:groups.setdefault(row['group'],set()).add(row['sequence'])
    validation={sequence for sequences in groups.values() for i,sequence in enumerate(sorted(sequences)) if i%4==0}
    splits={'train':[],'validation':[]};seen={key:set() for key in splits}
    for row in manifest['records']:
        if not row['sequence'].endswith('/training'):raise ValueError('Unexpected testing record')
        label='validation' if row['sequence'] in validation else 'train'
        path=root/row['directory']/'training.npz'
        if sha(path)!=row['archive_sha256']:raise ValueError('Changed public crop archive')
        with np.load(path,allow_pickle=False) as archive:
            for image in archive['images'].reshape(-1,96,96):
                image=np.ascontiguousarray(image,dtype=np.float32)
                digest=hashlib.sha256(np.rint(image*255).astype(np.uint8).tobytes()).hexdigest()
                mirror=hashlib.sha256(np.ascontiguousarray(np.rint(image[:,::-1]*255).astype(np.uint8)).tobytes()).hexdigest()
                if digest in excluded or mirror in excluded:raise ValueError('Public/local duplicate')
                if digest in seen[label]:continue
                seen[label].update((digest,mirror));splits[label].append(image)
    if seen['train']&seen['validation']:raise ValueError('Public train/validation duplicate')
    return {key:np.stack(images)[:,None] for key,images in splits.items()},sorted(validation)


@torch.no_grad()
def predict(model,x,batch=128):
    model.eval();return torch.cat([model(chunk).sigmoid() for chunk in x.split(batch)]).cpu().numpy()


def metrics(q,labels,strong=None):
    if strong is None:strong=np.ones(len(q),bool)
    score=np.maximum(q,1-q);wrong=(q>=.5)!=(labels>=.5)
    rows={}
    for threshold in (.5,.9,.95,.98,.99):
        selected=(score>=threshold)&strong
        rows[str(threshold)]=dict(count=int(selected.sum()),wrong=int((wrong&selected).sum()),error_rate=float(wrong[selected].mean()) if selected.any() else None)
    valid=np.clip(q[strong],1e-6,1-1e-6);target=labels[strong]
    return dict(images=len(q),strong_reference_images=int(strong.sum()),cross_entropy=float(-(target*np.log(valid)+(1-target)*np.log(1-valid)).mean()),thresholds=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('train','train-evidence','development','development-evidence','public','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--epochs',type=int,default=60);a=p.parse_args()
    if a.epochs<1 or a.train.resolve()==a.development.resolve() or any('143912Z' in str(v) for v in (a.train,a.development)):p.error('Invalid split/epochs')
    a.output.mkdir(parents=True,exist_ok=False)
    seed=3295;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    lx,ly,ls,lk,lh=local_images(a.train,a.train_evidence)
    dx,dy,ds,dk,dh=local_images(a.development,a.development_evidence)
    if lh&dh:raise ValueError('Local recording duplicate')
    public,validation_sequences=public_images(a.public,lh|dh)
    x=torch.from_numpy(lx[ls]).cuda();y=torch.from_numpy(ly[ls]).cuda();dx=torch.from_numpy(dx).cuda();px=torch.from_numpy(public['train']).cuda()
    pv=torch.from_numpy(public['validation']).cuda();pv=torch.cat((pv,pv.flip(-1)));py=np.concatenate((np.zeros(len(pv)//2),np.ones(len(pv)//2)))
    model=TinyChirality().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs,eta_min=.00003)
    steps=max(100,math.ceil(max(len(x),len(px))/32))
    manifest=dict(scope=__doc__,architecture=ARCH,seed=seed,epochs=a.epochs,batch=64,steps_per_epoch=steps,learning_rate=.002,parameters=sum(p.numel() for p in model.parameters()),
        public_manifest_sha256=sha(a.public/'manifest.json'),public_validation_sequences=validation_sequences,public_train_unmirrored_images=len(px),public_validation_images=len(pv),local_train_strong_images=len(x),local_development_images=len(dx),
        labels='Public canonical-left=0, horizontal mirror=1; local labels from independent teacher q>=.85 or<=.15 with pose/sigma agreement. Teacher labels are not physical truth.',
        augmentation='Per batch32local/32public; random horizontal flips/complemented labels; prior photometric augmentation; rotation+-15degrees, zero padding',
        selection='Minimum local-development cross entropy on strong teacher-reference images, every5epochs. Public validation sequences never train; latest local recording never read.',
        policy_probe='Predeclared0.98confidence and mirrored-probe consistency within0.05, plus existing overlap/raw detector checks; not runtime enabled',
        torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),tf32=False,source_sha256={name:sha(BASE/name) for name in ('train_chirality.py','tiny_chirality.py')},
        train_recording=a.train.name,development_recording=a.development.name,train_transport_sha256=sha(a.train/'transport.bin'),development_transport_sha256=sha(a.development/'transport.bin'),train_evidence_sha256=sha(a.train_evidence),development_evidence_sha256=sha(a.development_evidence))
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest),flush=True)
    history=[];best=float('inf');best_epoch=None;g=torch.Generator(device='cuda').manual_seed(seed);started=time.perf_counter()
    for epoch in range(1,a.epochs+1):
        model.train();total=0.
        for step in range(steps):
            li=torch.randint(len(x),(32,),device='cuda',generator=g);pi=torch.randint(len(px),(32,),device='cuda',generator=g)
            value=torch.cat((x[li],px[pi]));target=torch.cat((y[li],torch.zeros(32,device='cuda')))
            flip=torch.rand(64,device='cuda',generator=g)<.5
            value=torch.where(flip[:,None,None,None],value.flip(-1),value);target=torch.where(flip,1-target,target)
            value=augment(value,g)
            theta=torch.zeros((64,2,3),device='cuda');angle=(torch.rand(64,device='cuda',generator=g)-.5)*(math.pi/6)
            theta[:,0,0]=theta[:,1,1]=angle.cos();theta[:,0,1]=-angle.sin();theta[:,1,0]=angle.sin()
            value=F.grid_sample(value,F.affine_grid(theta,value.shape,align_corners=False),align_corners=False)
            opt.zero_grad(set_to_none=True);loss=F.binary_cross_entropy_with_logits(model(value),target)
            if not torch.isfinite(loss):raise ValueError('Nonfinite classifier loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();total+=loss.item()
        row=dict(epoch=epoch,train_bce=total/steps,seconds=time.perf_counter()-started)
        if epoch%5==0 or epoch==a.epochs:
            local=metrics(predict(model,dx),dy,ds);public_score=metrics(predict(model,pv),py)
            row.update(local_development=local,public_validation=public_score)
            if local['cross_entropy']<best:
                best=local['cross_entropy'];best_epoch=epoch
                torch.save(dict(architecture=ARCH,epoch=epoch,state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}),a.output/'best.torch')
        scheduler.step();history.append(row);(a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n');print(json.dumps(row),flush=True)
    result=dict(scope='Image chirality research only; scores uncalibrated and teacher labels imperfect; no live integration.',best_epoch=best_epoch,best_local_development_cross_entropy=best,checkpoint_sha256=sha(a.output/'best.torch'),seconds=time.perf_counter()-started)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
