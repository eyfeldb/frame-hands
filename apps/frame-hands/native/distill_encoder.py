"""Offline teacher-feature distillation. No device/camera access or runtime changes.

Whole recordings define train/development sets. The latest accepted recording
is deliberately not an input. Teacher agreement is not physical ground truth.
"""
import argparse, hashlib, json, math, random, sys, time
from pathlib import Path
import cv2
import numpy as np
import torch
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / 'exploratory'))
from live_backend import Backend
from replay_mesh_trial import packets
from student_encoder import ARCH, make_student


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def collect(session, stride):
    images, rows, seen = [], [], set()
    count = 0
    for meta, blobs in packets(session / 'transport.bin'):
        if meta.get('waiting'):
            continue
        count += 1
        if (count - 1) % stride:
            continue
        for crop, blob in zip(meta['crops'], blobs):
            image = cv2.imdecode(np.frombuffer(blob, np.uint8), 0)
            if image is None or image.shape != (96, 96):
                raise ValueError('Unexpected crop')
            for flip in (False, True):
                value = np.ascontiguousarray(image[:, ::-1] if flip else image)
                digest = hashlib.sha256(value.tobytes()).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                images.append(value)
                rows.append(dict(stamp=meta['stamp'], camera=crop['camera'], id=crop['id'], flip=flip, sha256=digest))
    if not images:
        raise ValueError('Empty split')
    return np.stack(images)[:, None], rows, seen


@torch.no_grad()
def targets(model, images, batch):
    result = []
    for offset in range(0, len(images), batch):
        x = torch.from_numpy(images[offset:offset + batch]).to('cuda', dtype=torch.float32) / 255.
        result.append(model(x).cpu())
    return torch.cat(result)


@torch.no_grad()
def evaluate(model, x, y, batch):
    model.eval()
    squared = 0.
    maximum = 0.
    for offset in range(0, len(x), batch):
        delta = model(x[offset:offset + batch]) - y[offset:offset + batch]
        squared += delta.square().sum().item()
        maximum = max(maximum, delta.abs().max().item())
    mse = squared / y.numel()
    energy = y.square().mean().item()
    return dict(mse=mse, normalized_mse=mse / max(energy, 1e-12), rmse=math.sqrt(mse), max_abs=maximum)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'train', 'development', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--train-stride', type=int, default=3)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch', type=int, default=64)
    a = p.parse_args()
    if a.train.resolve() == a.development.resolve() or min(a.train_stride, a.epochs, a.batch) < 1:
        p.error('Distinct whole recordings and positive limits required')
    if '143912Z' in str(a.train) or '143912Z' in str(a.development):
        p.error('Latest accepted recording is reserved for final held-out evaluation')
    if not torch.cuda.is_available():
        p.error('This development-only trainer requires the PC CUDA environment')
    a.output.mkdir(parents=True, exist_ok=False)
    seed = 7153
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    backend = Backend(a.upstream, {})
    teacher = backend.model._feature_extractor._image_backbone.eval().cuda()
    teacher.requires_grad_(False)
    student = make_student(a.upstream, teacher).cuda()
    train, train_rows, train_hashes = collect(a.train, a.train_stride)
    development, dev_rows, dev_hashes = collect(a.development, 1)
    overlap = train_hashes & dev_hashes
    if overlap:
        raise ValueError(f'{len(overlap)} duplicate crops across recording splits')
    manifest = dict(scope=__doc__, seed=seed, architecture=ARCH, teacher_architecture='resnet_layers_2352-f32',
        torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), tf32=False,
        teacher_weights_sha256=sha(a.upstream / 'pretrained_models/pretrained_weights.torch'),
        train_recording=a.train.name, development_recording=a.development.name,
        train_transport_sha256=sha(a.train / 'transport.bin'), development_transport_sha256=sha(a.development / 'transport.bin'),
        train_stride=a.train_stride, train_images=len(train), development_images=len(development), duplicate_crops=0,
        trainable_parameters=sum(v.numel() for v in student.parameters()), teacher_parameters=sum(v.numel() for v in teacher.parameters()),
        optimizer='AdamW', learning_rate=.001, weight_decay=.0001, epochs=a.epochs, batch=a.batch,
        initialization='Leading-channel teacher slices; output72 channels preserved; all student layers trainable',
        loss='Feature MSE divided by fixed training teacher mean squared energy; frozen downstream network',
        selection='Lowest whole-recording development feature MSE; no held-out143912Z use')
    (a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for label, rows in (('train', train_rows), ('development', dev_rows)):
        with (a.output / (label + '-crops.jsonl')).open('w') as f:
            for row in rows:
                f.write(json.dumps(row) + '\n')
    print(json.dumps(manifest), flush=True)
    train_y = targets(teacher, train, a.batch).cuda()
    dev_y = targets(teacher, development, a.batch).cuda()
    train_x = torch.from_numpy(train).to('cuda', dtype=torch.float32) / 255.
    dev_x = torch.from_numpy(development).to('cuda', dtype=torch.float32) / 255.
    scale = train_y.square().mean().detach().clamp_min(1e-12)
    optimizer = torch.optim.AdamW(student.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs, eta_min=.00003)
    initial = evaluate(student, dev_x, dev_y, a.batch)
    print('initial', initial, flush=True)
    best, best_epoch = float('inf'), None
    history = []
    started = time.perf_counter()
    for epoch in range(1, a.epochs + 1):
        student.train()
        order = torch.randperm(len(train_x), device='cuda')
        total = 0.
        for ids in order.split(a.batch):
            optimizer.zero_grad(set_to_none=True)
            loss = (student(train_x[ids]) - train_y[ids]).square().mean() / scale
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 10.)
            optimizer.step()
            total += loss.item() * len(ids)
        dev = evaluate(student, dev_x, dev_y, a.batch)
        row = dict(epoch=epoch, train_normalized_mse=total / len(train_x), development=dev, elapsed_seconds=time.perf_counter() - started)
        history.append(row)
        print(json.dumps(row), flush=True)
        if dev['mse'] < best:
            best, best_epoch = dev['mse'], epoch
            torch.save(dict(architecture=ARCH, epoch=epoch, state_dict={k: v.detach().cpu().clone() for k, v in student.state_dict().items()}), a.output / 'best.torch')
        scheduler.step()
        (a.output / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
    summary = dict(initial_development=initial, best_epoch=best_epoch, best_development=history[best_epoch-1]['development'],
        checkpoint_sha256=sha(a.output / 'best.torch'), seconds=time.perf_counter()-started,
        scope='Feature imitation on a different recording; no physical accuracy, native speed or live acceptance established')
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
