"""Read-only, timestamp-exact adapters for private FHSET01 and diagnostic traces."""
import itertools
import json
from pathlib import Path
import struct
import numpy as np

SET = struct.Struct('<8sII')
CAM = struct.Struct('<16sIIQQ')
NAMES = ('slam_left', 'slam_right')


def read_audit(path):
    result = {}
    current = None
    for line in (path.read() if hasattr(path, 'read') else Path(path).read_text()).splitlines():
        f = line.split()
        if not f:
            continue
        if f[0] == 'set':
            if current is not None:
                raise ValueError('Unclosed audit set')
            current = dict(raw_ns=int(f[1]), mono_ns=int(f[2]), count=int(f[3]), hands={}, views=[])
        elif f[0] == 'hand' and current is not None:
            ident = int(f[1])
            if ident in current['hands']:
                raise ValueError('Duplicate audit identity')
            current['hands'][ident] = dict(right=float(f[2]), stereo=int(f[3]))
        elif f[0] == 'view' and current is not None:
            if len(f) != 47:
                raise ValueError('Malformed landmark row')
            v = dict(camera=f[1], id=int(f[2]), presence=float(f[3]), right=float(f[4]),
                     points=np.array(list(map(float, f[5:]))).reshape(21, 2))
            if not np.isfinite(v['points']).all() or not 0 <= v['presence'] <= 1 or not 0 <= v['right'] <= 1:
                raise ValueError('Invalid audit observation')
            current['views'].append(v)
        elif f[0] == 'end' and current is not None:
            if current['count'] != len(current['hands']) or current['raw_ns'] in result:
                raise ValueError('Incomplete or duplicate audit set')
            result[current['raw_ns']] = current
            current = None
        else:
            raise ValueError('Unexpected audit row')
    if current is not None:
        raise ValueError('Truncated audit')
    return result


class Recording:
    def __init__(self, path):
        self.path = Path(path)
        self.index = []
        length = self.path.stat().st_size
        with self.path.open('rb') as stream:
            while stream.tell() < length:
                offset = stream.tell()
                header = stream.read(SET.size)
                if len(header) != SET.size:
                    raise ValueError('Truncated set header')
                magic, count, size = SET.unpack(header)
                if magic != b'FHSET01\0' or not 1 <= count <= 4:
                    raise ValueError('Invalid set header')
                metadata = stream.read(count*CAM.size)
                if len(metadata) != count*CAM.size:
                    raise ValueError('Truncated camera header')
                rows = [CAM.unpack_from(metadata, i*CAM.size) for i in range(count)]
                if any(not 1 <= w <= 4096 or not 1 <= h <= 4096 for _, w, h, _, _ in rows):
                    raise ValueError('Invalid dimensions')
                if size != SET.size+count*CAM.size+sum(w*h for _, w, h, _, _ in rows) or offset+size > length:
                    raise ValueError('Invalid record size or truncated pixels')
                names = [n.rstrip(b'\0').decode('ascii') for n, *_ in rows]
                if len(set(names)) != count:
                    raise ValueError('Duplicate camera')
                stamp = min(row[3] for row in rows)
                if self.index and stamp <= self.index[-1][0]:
                    raise ValueError('Nonmonotonic capture time')
                self.index.append((stamp, offset+SET.size+count*CAM.size, rows))
                stream.seek(offset+size)
        if not self.index:
            raise ValueError('Empty recording')

    def frames(self, start, end):
        first = self.index[0][0]
        with self.path.open('rb') as stream:
            for stamp, offset, rows in self.index:
                seconds = (stamp-first)/1e9
                if not start <= seconds < end:
                    continue
                stream.seek(offset)
                images = {}
                for name, w, h, capture, dequeue in rows:
                    pixels = stream.read(w*h)
                    if len(pixels) != w*h:
                        raise ValueError('Recording changed during replay')
                    images[name.rstrip(b'\0').decode('ascii')] = (np.frombuffer(pixels, np.uint8).reshape(h, w), capture)
                yield seconds, stamp, images


def pose(value, scale=1):
    x, z = np.array(value['plus_x']), np.array(value['plus_z'])
    out = np.eye(4)
    out[:3, :3] = np.column_stack((x, np.cross(z, x), z))
    out[:3, 3] = np.array(value['position'])*scale
    if not np.allclose(out[:3, :3].T@out[:3, :3], np.eye(3), atol=1e-5):
        raise ValueError('Nonrigid calibration')
    return out


def load_calibration(persist):
    from lib.common.camera import Fisheye62CameraModel
    device = json.loads((persist/'device_config.json').read_text())
    rig = json.loads((persist/'xrservice.json').read_text())
    # Device head translation, if nonzero, also uses metres.
    head_from_cam0 = np.linalg.inv(pose(device['head'], 1000))@pose(device['cv']['cad_from_cal'], 1000)
    cameras = {}
    for c in rig['cameras']:
        if c['sourceCamera'] not in NAMES:
            continue
        kb = next(v for v in c['intrinsics'] if v['cameraModel'] == 'kb')
        cameras[c['sourceCamera']] = Fisheye62CameraModel(c['width'], c['height'],
            (kb['fx'], kb['fy']), (kb['cx'], kb['cy']),
            (kb['k1'], kb['k2'], kb['k3'], kb['k4'], 0, 0, 0, 0),
            head_from_cam0@pose(c['extrinsics']))
    if set(cameras) != set(NAMES):
        raise ValueError('Both forward calibrations are required')
    return cameras


def unproject(camera, pixels):
    q = (np.asarray(pixels)-camera.c)/camera.f
    radius = np.linalg.norm(q, axis=-1)
    theta = radius.copy()
    k = np.array(camera.distort[:4])
    for _ in range(12):
        powers = np.stack([theta**p for p in (2, 4, 6, 8)], -1)
        derivative = 1 + powers@(k*np.array([3, 5, 7, 9]))
        theta -= (theta*(1+powers@k)-radius)/derivative
    if not np.isfinite(theta).all() or np.any(theta < 0) or np.any(theta >= np.pi):
        raise ValueError('Invalid fisheye inverse')
    return np.concatenate((q*np.divide(np.sin(theta), radius, out=np.ones_like(theta), where=radius>1e-12)[..., None],
                           np.cos(theta)[..., None]), -1)


def crop_camera(camera, pixels, side, world_from_head, margin=1.25):
    from lib.common.camera import PinholePlaneCameraModel
    rays = unproject(camera, pixels)@camera.camera_to_world_xf[:3, :3].T
    z = rays.mean(0)
    z /= np.linalg.norm(z)
    down = world_from_head[:3, :3]@np.array([0., -1., 0.])
    x = np.cross(down, z)
    if np.linalg.norm(x) < .01:
        raise ValueError('Crop upright axis degenerate')
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    rotation = np.column_stack((x*(-1 if side else 1), y, z))
    local = rays@rotation
    if np.any(local[:, 2] <= .05):
        raise ValueError('Crop spans camera horizon')
    extent = np.max(np.abs(local[:, :2]/local[:, 2:]))
    focal = 47.5/(extent*margin)
    if not 5 <= focal <= 2000:
        raise ValueError('Crop extent invalid')
    xf = camera.camera_to_world_xf.copy()
    xf[:3, :3] = rotation
    return PinholePlaneCameraModel(96, 96, focal, (47.5, 47.5), [], xf)


class Ownership:
    """Joint injective assignment with bounded history; no spatial left/right heuristic."""
    def __init__(self):
        self.previous = {}

    def assign(self, observation, mode='fresh'):
        candidates = {}
        for v in observation['views']:
            if v['camera'] in NAMES and v['presence'] >= .5:
                candidates.setdefault(v['id'], []).append(v)
        ids = sorted(candidates)
        if len(ids) > 4:
            raise ValueError('Unexpected number of candidate hands')
        scores = {}
        now = observation['raw_ns']
        for ident, views in candidates.items():
            q = np.average([v['right'] for v in views], weights=[v['presence'] for v in views])
            scores[ident] = float(np.clip(q, .001, .999))
        if mode == 'baseline':
            result = {}
            for side in (0, 1):
                matches = [i for i in ids if i in observation['hands'] and int(observation['hands'][i]['right'] > .5) == side]
                if len(matches) == 1:
                    result[side] = matches[0]
        else:
            best, result = -np.inf, {}
            for allocation in itertools.product((-1, 0, 1), repeat=len(ids)):
                assigned = [s for s in allocation if s >= 0]
                if len(set(assigned)) != len(assigned):
                    continue
                score = 0.
                for ident, side in zip(ids, allocation):
                    if side < 0:
                        score += np.log(.45)
                        continue
                    probability = scores[ident] if side else 1-scores[ident]
                    old = self.previous.get(ident)
                    continuing = old and old[0] == side and now-old[1] <= 100_000_000
                    # An ambiguous new detection has no anatomical side yet.
                    # Injectivity alone must not manufacture two opposite hands.
                    if probability < (.40 if continuing else .60):
                        score = -np.inf
                        break
                    score += np.log(probability)
                    if continuing:
                        score += .25
                if score > best:
                    best = score
                    result = {side: ident for ident, side in zip(ids, allocation) if side >= 0}
        self.previous = {ident: (side, now) for side, ident in result.items()}
        return result, candidates, scores


def capture_poses(session):
    """Join sampler poses to source monotonic capture by exact relay packet sequence.

    Pose is the recorded SteamVR capture-time estimate, not optical ground truth.
    Never align independent clocks by subtracting their initial timestamps.
    """
    seq_to_capture = {}
    relay_size = 3280
    with (session/'relay-trace.bin').open('rb') as stream:
        while data := stream.read(relay_size):
            if len(data) != relay_size or struct.unpack_from('<8sII', data) != (b'FHRELAY1', 1, relay_size):
                raise ValueError('Malformed relay trace')
            if data[40:48] != b'FHREAD1\0' or data[2712:2720] != b'FHPCVR1\0':
                raise ValueError('Malformed relay payload')
            source_capture = struct.unpack_from('<Q', data, 56+24)[0]
            seq = struct.unpack_from('<Q', data, 2712+16)[0]
            seq_to_capture[seq] = source_capture
    result = {}
    with (session/'sampler-trace.bin').open('rb') as stream:
        while data := stream.read(3136):
            if len(data) != 3136 or struct.unpack_from('<8sII', data) != (b'FHTRACE1', 1, 3136):
                raise ValueError('Malformed sampler trace')
            capture_flags = struct.unpack_from('<I', data, 32)[0]
            if capture_flags & 3 != 3:
                continue
            seq = struct.unpack_from('<Q', data, 704+16)[0]
            stamp = seq_to_capture.get(seq)
            if not stamp or stamp in result:
                continue
            xf = np.eye(4)
            xf[:3, :] = np.array(struct.unpack_from('<12f', data, 88)).reshape(3, 4)
            xf[:3, 3] *= 1000
            if np.isfinite(xf).all() and np.allclose(xf[:3, :3].T@xf[:3, :3], np.eye(3), atol=1e-3):
                result[stamp] = xf
    return result


def warp(source, destination, image):
    import cv2
    y,x=np.mgrid[:96,:96]
    world=destination.eye_to_world(destination.window_to_eye(np.stack((x,y),-1)))
    eye=source.world_to_eye(world)
    uv=source.eye_to_window(eye).astype(np.float32)
    uv[eye[...,2]<=0]=-1
    return cv2.remap(image,uv[...,0],uv[...,1],cv2.INTER_LINEAR)
