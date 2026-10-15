"""Offline UmeTrack stereo reconstruction on recorded Frame images. No XR or device I/O."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
from frame_data import Recording, read_audit, load_calibration, crop_camera, capture_poses, Ownership, NAMES
from prepare_umetrack import COMMIT

# Corresponding landmarks only. UmeTrack has palm center instead of MediaPipe thumb CMC.
COMMON = [(5, 0), (6, 2), (7, 3), (0, 4), (8, 5), (9, 6), (10, 7), (1, 8),
          (11, 9), (12, 10), (13, 11), (2, 12), (14, 13), (15, 14), (16, 15), (3, 16),
          (17, 17), (18, 18), (19, 19), (4, 20)]
EDGES = [(5, 6), (6, 7), (7, 0), (5, 8), (8, 9), (9, 10), (10, 1), (8, 11), (11, 12),
         (12, 13), (13, 2), (11, 14), (14, 15), (15, 16), (16, 3), (14, 17), (17, 18), (18, 19), (19, 4), (5, 17)]
MP_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
            (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def warp(source, destination, image):
    y, x = np.mgrid[:96, :96]
    pixels = np.stack((x, y), -1)
    world = destination.eye_to_world(destination.window_to_eye(pixels))
    eye = source.world_to_eye(world)
    uv = source.eye_to_window(eye).astype(np.float32)
    uv[eye[..., 2] <= 0] = -1
    return cv2.remap(image, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR)


def draw_hand(image, points, edges, label, color):
    if not np.isfinite(points).all() or np.max(np.abs(points)) > 10000:
        return
    p = np.rint(points).astype(int)
    for a, b in edges:
        cv2.line(image, tuple(p[a]), tuple(p[b]), color, 2, cv2.LINE_AA)
    for point in p:
        cv2.circle(image, tuple(point), 3, color, -1)
    cv2.putText(image, label, (int(np.clip(p[5, 0], 5, image.shape[1]-200)), int(np.clip(p[5, 1], 80, image.shape[0]-20))),
                cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)


def preview(images, observation, predictions, cameras, seconds, output, crops):
    panels = []
    for name in NAMES:
        raw = cv2.cvtColor(images[name][0], cv2.COLOR_GRAY2BGR)
        original = raw.copy()
        alternate = raw.copy()
        for v in observation['views']:
            if v['camera'] != name:
                continue
            hand = observation['hands'].get(v['id'])
            label = ('R' if hand['right'] > .5 else 'L') if hand else '?'
            draw_hand(original, v['points'], MP_EDGES, f'{label} ID{v["id"]}', (255, 220, 0))
        for p in predictions:
            pixels = cameras[name].eye_to_window(cameras[name].world_to_eye(np.array(p['landmarks_world_mm'])))
            draw_hand(alternate, pixels, EDGES, f'{p["side"]} ID{p["source_id"]}', (0, 255, 90))
        row = []
        for panel, title in [(raw, 'RAW'), (original, 'RECORDED MediaPipe'), (alternate, 'EXPERIMENT UmeTrack')]:
            cv2.rectangle(panel, (0, 0), (panel.shape[1], 45), (0, 0, 0), -1)
            cv2.putText(panel, f'{seconds:.3f}s {name} {title}', (10, 29), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
            row.append(cv2.resize(panel, (528, 512)))
        panels.append(np.hstack(row))
    if not cv2.imwrite(str(output), np.vstack(panels)):
        raise OSError('Could not save preview')
    if crops:
        tiles = [cv2.resize(c, (192, 192), interpolation=cv2.INTER_NEAREST) for c in crops]
        cv2.imwrite(str(output.with_name(output.stem+'-crops.png')), np.hstack(tiles))


def quantiles(values):
    return dict(zip(('p50', 'p95', 'max'), map(float, np.percentile(values, [50, 95, 100])))) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', type=float, default=0)
    parser.add_argument('--end', type=float, default=1e9)
    parser.add_argument('--ownership', choices=('baseline', 'fresh'), default='fresh')
    parser.add_argument('--memory', choices=('off', 'world'), default='world')
    parser.add_argument('--scale', type=float, default=1.)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--preview-every', type=float, default=1.)
    args = parser.parse_args()
    if not 0 <= args.start < args.end or not .5 <= args.scale <= 1.5 or not 1 <= args.threads <= 32 or args.preview_every <= 0:
        parser.error('Invalid interval, scale, threads or preview cadence')
    if args.output.exists():
        parser.error('Output must be a new directory to preserve previous experiments')
    rev = subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if rev != COMMIT:
        raise ValueError('Unpinned UmeTrack source')
    sys.path.insert(0, str(args.upstream.resolve()))
    from lib.models.model_loader import load_pretrained_model
    from lib.models.umetrack_model import InputFrameData, InputFrameDesc, InputSkeletonData
    from lib.common.hand import HandModel, scaled_hand_model, LANDMARK
    from lib.common.hand_skinning import skin_landmarks
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    weights = args.upstream/'pretrained_models/pretrained_weights.torch'
    if sha(weights) != '72eab0ea2c5c4518188ede4cf62e98f62eb1e4ef13628c0f20ad3fc56d1bf18d':
        raise ValueError('Weights changed')
    model = load_pretrained_model(str(weights)).eval()
    raw_hand = json.loads((args.upstream/'dataset/generic_hand_model.json').read_text())
    hand = HandModel(**{k: torch.tensor(v) if isinstance(v, list) else v for k, v in raw_hand.items()})
    hand = scaled_hand_model(hand, args.scale)
    skeleton = InputSkeletonData(hand.joint_rotation_axes.float(), hand.joint_rest_positions.float()*.001)
    frame = args.session/'frame'
    audits = read_audit(frame/'live-views.txt')
    recording = Recording(frame/'sets.bin')
    camera_head = load_calibration(frame/'snapshot/device/persist')
    poses = capture_poses(args.session)
    ownership = Ownership()
    args.output.mkdir(parents=True)
    manifest = dict(version=1, upstream_commit=rev, weights_sha256=sha(weights),
        baseline_commit='f9e2367', adapter_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        upstream_adapted_sha256={n:sha(args.upstream/n) for n in ('lib/common/hand_skinning.py','lib/models/model_loader.py','dataset/generic_hand_model.json')}, args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        environment=dict(python=platform.python_version(), platform=platform.platform(), torch=torch.__version__,
                         numpy=np.__version__, opencv=cv2.__version__, device='CPU'),
        calibration_sha256={p.name:sha(p) for p in (frame/'snapshot/device/persist').glob('*.json')},
        audit_sha256=sha(frame/'live-views.txt'), image_bytes=(frame/'sets.bin').stat().st_size,
        landmark_order=[x.value for x in LANDMARK],
        scope='Offline pose backend with recorded detector ROIs; no independent detection, no calibrated confidence, no XR output; not accuracy ground truth')
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    previous = {}
    elapsed, residuals, motions, lengths = [], [], [], []
    counters = dict(frames=0, unmatched_audit=0, missing_pose=0, model_hands=0, stereo_hands=0,
                    baseline_duplicate_sets=0, reassigned_hands=0, crop_rejected=0, memory_used=0)
    next_preview = args.start
    begin = time.perf_counter()
    with (args.output/'predictions.jsonl').open('x') as output:
        for seconds, stamp, images in recording.frames(args.start, args.end):
            counters['frames'] += 1
            observation = audits.get(stamp)
            if observation is None:
                counters['unmatched_audit'] += 1
                previous = {}
                continue
            world = poses.get(observation['mono_ns'])
            has_pose = world is not None
            if not has_pose:
                counters['missing_pose'] += 1
                world = np.eye(4)
            cameras = {name: c.copy(camera_to_world_xf=world@c.camera_to_world_xf) for name, c in camera_head.items()}
            sides = [int(h['right'] > .5) for h in observation['hands'].values()]
            counters['baseline_duplicate_sets'] += len(sides) == 2 and len(set(sides)) == 1
            assigned, candidates, side_scores = ownership.assign(observation, args.ownership)
            crops, intrinsics, extrinsics, ranges, included, view_lists, use_memory = [], [], [], [], [], [], []
            t0 = time.perf_counter()
            for side, ident in sorted(assigned.items()):
                views = sorted(candidates[ident], key=lambda v: NAMES.index(v['camera']))
                first = len(crops)
                valid_views = []
                for v in views:
                    name = v['camera']
                    if name not in images or images[name][0].shape != (cameras[name].height, cameras[name].width):
                        continue
                    try:
                        crop = crop_camera(cameras[name], v['points'], side, world)
                    except ValueError:
                        counters['crop_rejected'] += 1
                        continue
                    crops.append(warp(cameras[name], crop, images[name][0]))
                    intrinsics.append(crop.uv_to_window_matrix())
                    xf = np.linalg.inv(crop.camera_to_world_xf)
                    xf[:3, 3] *= .001
                    extrinsics.append(xf)
                    valid_views.append(v)
                if len(crops) == first:
                    continue
                ranges.append((first, len(crops)))
                included.append((side, ident))
                view_lists.append(valid_views)
                old = previous.get(side)
                memory = bool(args.memory == 'world' and has_pose and old and old['has_pose'] and old['source_id'] == ident and stamp-old['stamp'] <= 100_000_000)
                use_memory.append(memory)
                counters['memory_used'] += memory
            predictions = []
            if included:
                frame_data = InputFrameData(torch.tensor(np.stack(crops)/255., dtype=torch.float32),
                                            torch.tensor(np.stack(intrinsics), dtype=torch.float32),
                                            torch.tensor(np.stack(extrinsics), dtype=torch.float32))
                indices = torch.tensor([s for s, _ in included])
                desc = InputFrameDesc(torch.tensor(ranges), indices, torch.tensor(use_memory), indices)
                with torch.inference_mode():
                    inferred = model.regress_pose_use_skeleton(frame_data, desc, skeleton)
                    for i, (side, ident) in enumerate(included):
                        wrist = inferred.wrist_xfs[i].clone()
                        wrist[:3, 3] *= 1000
                        skin_wrist = wrist.clone()
                        if side:
                            skin_wrist[:, 0] *= -1
                        points = skin_landmarks(hand, inferred.joint_angles[i], skin_wrist).numpy()
                        if not np.isfinite(points).all():
                            raise ValueError('Nonfinite model prediction')
                        head_points = (np.linalg.inv(world)[:3, :3]@points.T).T+np.linalg.inv(world)[:3, 3]
                        disagreement = []
                        for v in view_lists[i]:
                            c = cameras[v['camera']]
                            uv = c.eye_to_window(c.world_to_eye(points))
                            disagreement.extend(np.linalg.norm(uv[[a for a, b in COMMON]]-v['points'][[b for a, b in COMMON]], axis=1).tolist())
                        residuals.extend(disagreement)
                        segment_lengths = np.linalg.norm(points[[b for a, b in EDGES]]-points[[a for a, b in EDGES]], axis=1)
                        lengths.append(segment_lengths)
                        p = dict(side='R' if side else 'L', source_id=ident, stamp=stamp, has_pose=has_pose,
                                 memory=use_memory[i], views=[v['camera'] for v in view_lists[i]], right_score=side_scores[ident],
                                 landmarks_world_mm=points.tolist(), landmarks_head_mm=head_points.tolist(),
                                 wrist_world_mm=wrist.numpy().tolist(), joint_angles=inferred.joint_angles[i].numpy().tolist(),
                                 uncertainty_raw=inferred.landmark_uncertainty_sigmas[i].numpy().tolist(),
                                 disagreement_px=quantiles(disagreement), segment_lengths_mm=segment_lengths.tolist())
                        old = previous.get(side)
                        if old and old['source_id'] == ident and stamp-old['stamp'] <= 100_000_000 and has_pose and old['has_pose']:
                            motions.extend(np.linalg.norm(points-np.array(old['landmarks_world_mm']), axis=1).tolist())
                        predictions.append(p)
                        counters['model_hands'] += 1
                        counters['stereo_hands'] += len(view_lists[i]) == 2
                        if ident in observation['hands']:
                            counters['reassigned_hands'] += side != int(observation['hands'][ident]['right'] > .5)
            ms = (time.perf_counter()-t0)*1000
            if included:
                elapsed.append(ms)
            previous = {int(p['side'] == 'R'):p for p in predictions}
            output.write(json.dumps(dict(seconds=seconds, capture_raw_ns=stamp, capture_mono_ns=observation['mono_ns'],
                                         has_capture_pose=has_pose, processing_ms=ms, hands=predictions), allow_nan=False)+'\n')
            if seconds >= next_preview and set(NAMES).issubset(images):
                preview(images, observation, predictions, cameras, seconds, args.output/f'{seconds:08.3f}.jpg', crops)
                next_preview = seconds+args.preview_every
            if counters['frames'] % 150 == 0:
                print(f'{seconds:.2f}s: {counters["model_hands"]} hands, CPU p50 {np.median(elapsed) if elapsed else 0:.1f} ms', flush=True)
    report = dict(counters=counters, processing_ms=quantiles(elapsed), processing_ms_after_first=quantiles(elapsed[1:]),
                  mean_processing_fps=1000/np.mean(elapsed) if elapsed else None,
                  detector_model_disagreement_px=quantiles(residuals), consecutive_world_landmark_motion_mm=quantiles(motions),
                  segment_length_std_mm=np.std(lengths, axis=0).tolist() if lengths else [],
                  wall_seconds=time.perf_counter()-begin,
                  limitations=['ROIs and stereo associations come from the existing detector, so detector misses remain.',
                               'Disagreement is against MediaPipe, not physical ground truth; lower is not necessarily better.',
                               'Reported availability counts model outputs, not correct visible hands.',
                               'Uncertainty values are uncalibrated on Frame; no confidence of 1 is manufactured.',
                               'No end-to-end, GPU, headset or Maestro validation. Fixed generic skeleton scale.',
                               'Head poses are recorded SteamVR estimates, keyed by exact capture timestamps; missing poses reset memory.'])
    (args.output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
