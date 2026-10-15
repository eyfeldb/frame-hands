"""Controlled photometric-only versus pose-aware encoder distillation.

Offline development on saved crops. Frozen teacher and downstream layers; no
camera access, live integration, spatial augmentation or temporal training.
"""
import argparse, hashlib, json, random, sys, time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / 'exploratory'))
from live_backend import Backend
from replay_mesh_trial import packets
from feature_reuse import PreparedRegressor
from student_encoder import ARCH, load_student
from distill_encoder import sha


class MonoHead(torch.nn.Module):
    """Frozen zero-memory, single-view teacher head; raw outputs avoid SVD gradients."""
    def __init__(self, backend):
        super().__init__()
        self.model = backend.model.eval().cuda().requires_grad_(False)
        with torch.no_grad():
            self.register_buffer('skeleton', self.model._skeleton_enc(backend.skeleton.joint_rotation_axes.cuda(), backend.skeleton.joint_rest_positions.cuda()))

    def forward(self, features, intrinsics):
        from lib.models.model_utils import compute_singlev_xfs
        fused = self.model._feature_extractor.compute_singlev_features(features, compute_singlev_xfs(intrinsics))
        memory = features.new_zeros((len(features), self.model._temporal._nc_memory, *features.shape[-2:]))
        _, fused = self.model._temporal.forward_one_step(memory, fused)
        return self.model._regressor_k._pose_regression_layers(torch.cat((fused, self.skeleton.expand(len(features), -1, -1, -1)), 1)).flatten(1)


def collect(session, stride):
    images, intrinsics, rows, seen = [], [], [], set()
    n = 0
    for meta, blobs in packets(session / 'transport.bin'):
        if meta.get('waiting'):
            continue
        n += 1
        if (n - 1) % stride:
            continue
        for crop, blob in zip(meta['crops'], blobs):
            image = cv2.imdecode(np.frombuffer(blob, np.uint8), 0)
            k = np.asarray(crop['intrinsics'], np.float32)
            if image is None or image.shape != (96, 96) or k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0:
                raise ValueError('Invalid saved crop/intrinsics')
            for flip in (False, True):
                value = np.ascontiguousarray(image[:, ::-1] if flip else image)
                digest = hashlib.sha256(value.tobytes()).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest); images.append(value); intrinsics.append(k)
                rows.append(dict(stamp=meta['stamp'], camera=crop['camera'], id=crop['id'], flip=flip, sha256=digest))
    return np.stack(images)[:, None], np.stack(intrinsics), rows, seen


def augment(x, generator):
    n = len(x)
    gain = .7 + .6 * torch.rand((n, 1, 1, 1), device=x.device, generator=generator)
    offset = .12 * torch.rand((n, 1, 1, 1), device=x.device, generator=generator) - .06
    blend = .5 * torch.rand((n, 1, 1, 1), device=x.device, generator=generator)
    blurred = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode='replicate'), 3, stride=1)
    noise = .01 * torch.randn(x.shape, device=x.device, generator=generator)
    return ((x * (1 - blend) + blurred * blend) * gain + offset + noise).clamp(0, 1)


@torch.no_grad()
def verify_head(head, backend, x, k):
    from lib.models.regressor import decode_wrist_xfs_svd
    from lib.models.umetrack_model import InputFrameData, InputFrameDesc
    model = head.model
    features = model._feature_extractor._image_backbone(x)
    raw = head(features, k)
    prepared = PreparedRegressor(model, backend.skeleton, device='cuda')
    indices = torch.arange(len(x))
    data = InputFrameData(x[:, 0].cpu(), k.cpu(), torch.eye(4).repeat(len(x), 1, 1))
    desc = InputFrameDesc(torch.stack((indices, indices+1), 1), indices, torch.zeros(len(x), dtype=torch.bool), torch.zeros(len(x), dtype=torch.long))
    result = prepared.regress(data, desc, features)
    ranges = model._regressor_k._output_index_ranges
    a, b = ranges['wrist_xfs']
    wrist = decode_wrist_xfs_svd(raw[:, a:b], model._regressor_k._left_wrist_sample_points).cpu()
    a, b = ranges['joint_angles']
    angles = raw[:, a:b].cpu()
    a, b = ranges['landmark_uncertainty_sigmas']
    sigma = F.softplus(raw[:, a:b]).clamp_min(1e-5).cpu()
    errors = dict(wrist_max_abs=float((wrist-result.wrist_xfs).abs().max()), angles_max_abs=float((angles-result.joint_angles[:, :20]).abs().max()), sigma_max_abs=float((sigma-result.landmark_uncertainty_sigmas).abs().max()))
    if max(errors.values()) > 2e-6:
        raise ValueError(f'Head mismatch {errors}')
    return errors


@torch.no_grad()
def get_targets(head, x, k, batch):
    features, raw = [], []
    for start in range(0, len(x), batch):
        y = head.model._feature_extractor._image_backbone(x[start:start+batch])
        features.append(y)
        raw.append(head(y, k[start:start+batch]))
    return torch.cat(features), torch.cat(raw)


@torch.no_grad()
def evaluate(student, head, x, k, y, raw, scales, batch):
    student.eval()
    sums = dict(feature_nmse=0., pose_standardized_mse=0., wrist_points_mse=0., angles_mse=0.)
    for start in range(0, len(x), batch):
        sl = slice(start, start+batch)
        features = student(x[sl]); pose = head(features, k[sl]); delta = pose-raw[sl]; n = len(pose)
        sums['feature_nmse'] += float((features-y[sl]).square().mean()/scales[0])*n
        sums['pose_standardized_mse'] += float((delta/scales[1]).square().mean())*n
        sums['wrist_points_mse'] += float(delta[:, 20:41].square().mean())*n
        sums['angles_mse'] += float(delta[:, :20].square().mean())*n
    return {key: value/len(x) for key, value in sums.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'train', 'development', 'checkpoint', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch', type=int, default=64)
    a = p.parse_args()
    if a.train.resolve()==a.development.resolve() or any('143912Z' in str(v) for v in (a.train, a.development)):
        p.error('Distinct recordings required; latest accepted recording is reserved')
    if min(a.epochs, a.batch) < 1:
        p.error('Positive training limits required')
    a.output.mkdir(parents=True, exist_ok=False)
    seed = 8921
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    backend=Backend(a.upstream,{})
    head=MonoHead(backend)
    train,k,rows,hashes=collect(a.train,3)
    dev,dk,drows,dhashes=collect(a.development,1)
    if hashes & dhashes: raise ValueError('Cross-split duplicate crop')
    x=torch.from_numpy(train).cuda().float()/255.; k=torch.from_numpy(k).cuda()
    dx=torch.from_numpy(dev).cuda().float()/255.; dk=torch.from_numpy(dk).cuda()
    head_check=verify_head(head,backend,x[:4],k[:4])
    y,raw=get_targets(head,x,k,a.batch)
    dy,draw=get_targets(head,dx,dk,a.batch)
    scales=(y.square().mean().clamp_min(1e-12),raw.std(0).clamp_min(.02))
    arms={name:load_student(a.upstream,a.checkpoint) for name in ('photometric','photometric_pose')}
    optimizers={name:torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=.0001) for name,model in arms.items()}
    schedulers={name:torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs,eta_min=.00001) for name,opt in optimizers.items()}
    manifest=dict(scope=__doc__, torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),tf32=False,
        architecture=ARCH, seed=seed, initial_checkpoint_sha256=sha(a.checkpoint), teacher_weights_sha256=sha(a.upstream/'pretrained_models/pretrained_weights.torch'),
        train_recording=a.train.name,development_recording=a.development.name, train_transport_sha256=sha(a.train/'transport.bin'), development_transport_sha256=sha(a.development/'transport.bin'),
        train_images=len(x),development_images=len(dx),cross_split_duplicate_images=0,epochs=a.epochs,batch=a.batch,learning_rate=.0002,
        augmentation='Shared samples in both arms: gain[.7,1.3], offset[-.06,.06], box3 blur blend[0,.5], noise std.01; unchanged geometry',
        loss='feature normalized MSE; pose arm adds 0.1 times raw62-dimensional MSE standardized by TRAIN teacher std (floor .02)',
        checkpoint_selection='Minimum development single-view zero-memory standardized pose loss, both arms, evaluated every5epochs',
        screening_gate='For held-out test eligibility: development matched mean vertex median<=3mm, p95<=10mm, side/view changed frames<=1%; wearer acceptance still required',
        head_parity=head_check,raw_scales=scales[1].cpu().tolist(), source_sha256={p.name:sha(p) for p in (Path(__file__),BASE/'student_encoder.py')})
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest),flush=True)
    for label,r in (('train',rows),('development',drows)):
        (a.output/(label+'-crops.jsonl')).write_text(''.join(json.dumps(v)+'\n' for v in r))
    history=[];best={name:float('inf') for name in arms};best_epoch={}
    for name,model in arms.items():
        report=evaluate(model,head,dx,dk,dy,draw,scales,a.batch)
        print('initial',name,report,flush=True)
    g=torch.Generator(device='cuda').manual_seed(seed)
    started=time.perf_counter()
    for epoch in range(1,a.epochs+1):
        for model in arms.values():model.train()
        order=torch.randperm(len(x),device='cuda',generator=g)
        totals={name:0. for name in arms}
        for ids in order.split(a.batch):
            value=augment(x[ids],g)
            with torch.no_grad():
                features=head.model._feature_extractor._image_backbone(value)
                targets=head(features,k[ids])
            for name,model in arms.items():
                opt=optimizers[name];opt.zero_grad(set_to_none=True)
                actual=model(value)
                loss=(actual-features).square().mean()/scales[0]
                if name=='photometric_pose':
                    loss=loss+.1*((head(actual,k[ids])-targets)/scales[1]).square().mean()
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
                totals[name]+=float(loss.detach())*len(ids)
        row=dict(epoch=epoch,train={name:total/len(x) for name,total in totals.items()},seconds=time.perf_counter()-started)
        if epoch%5==0 or epoch==a.epochs:
            row['development']={name:evaluate(model,head,dx,dk,dy,draw,scales,a.batch) for name,model in arms.items()}
            for name,model in arms.items():
                score=row['development'][name]['pose_standardized_mse']
                if score<best[name]:
                    best[name]=score;best_epoch[name]=epoch
                    torch.save(dict(architecture=ARCH,epoch=epoch,state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}),a.output/(name+'-best.torch'))
        for scheduler in schedulers.values():scheduler.step()
        history.append(row)
        print(json.dumps(row),flush=True)
        (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    if any(p.grad is not None for p in head.parameters()):raise ValueError('Frozen teacher accumulated gradients')
    summary=dict(scope='Controlled feature/pose distillation; full causal replay required, not promoted.',seconds=time.perf_counter()-started,best_epoch=best_epoch,best_development_pose_loss=best,
        checkpoints_sha256={name:sha(a.output/(name+'-best.torch')) for name in arms})
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
