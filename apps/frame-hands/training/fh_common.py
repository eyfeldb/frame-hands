"""Shared, read-only helpers for Frame Hands model training on the PC.

Reproduces the installed tracker's conventions (tracker-rig-v9: track/calib.cpp,
track/nets.cpp) in NumPy so recorded sessions can be replayed, labelled and used to
evaluate candidate models with the same crop geometry the device uses.

Nothing here opens a camera, touches the headset, or writes outside the paths it is given.
"""
from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SET = struct.Struct('<8sII')
CAM = struct.Struct('<16sIIQQ')
CAMERA_NAMES = ('slam_left', 'slam_right')
PALM_SIZE, HAND_SIZE = 192, 224
ROI_LANDMARKS = (0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18)


# ----------------------------------------------------------------------------- calibration

def _pose(d, scale):
    x = np.array(d['plus_x'], float)
    z = np.array(d['plus_z'], float)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = x, y, z
    T[:3, 3] = np.array(d['position'], float) * scale
    return T


@dataclass
class Camera:
    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    k: np.ndarray          # k1..k4
    R: np.ndarray          # 3x3, camera axes as columns in the head frame
    origin: np.ndarray     # camera centre in the head frame (m)

    def _theta_d(self, t):
        t2 = t * t
        return t * (1 + t2 * (self.k[0] + t2 * (self.k[1] + t2 * (self.k[2] + t2 * self.k[3]))))

    def project_cam(self, p):
        """Camera-frame points (N,3) -> pixels (N,2). Matches Camera::project_cam."""
        p = np.atleast_2d(np.asarray(p, float))
        r = np.hypot(p[:, 0], p[:, 1])
        theta = np.arctan2(r, p[:, 2])
        s = np.where(r > 1e-12, self._theta_d(theta) / np.where(r > 1e-12, r, 1), 0.0)
        return np.stack((self.fx * p[:, 0] * s + self.cx, self.fy * p[:, 1] * s + self.cy), -1)

    def unproject(self, uv):
        """Pixels (N,2) -> unit rays in the camera frame (N,3). Newton as in Camera::unproject."""
        uv = np.atleast_2d(np.asarray(uv, float))
        mx, my = (uv[:, 0] - self.cx) / self.fx, (uv[:, 1] - self.cy) / self.fy
        td = np.hypot(mx, my)
        t = td.copy()
        for _ in range(8):
            t2 = t * t
            df = 1 + t2 * (3 * self.k[0] + t2 * (5 * self.k[1] + t2 * (7 * self.k[2] + t2 * 9 * self.k[3])))
            t = np.clip(t - (self._theta_d(t) - td) / df, 0.0, math.pi)
        s = np.where(td > 1e-12, np.sin(t) / np.where(td > 1e-12, td, 1), 1.0)
        return np.stack((mx * s, my * s, np.cos(t)), -1)

    def ray(self, uv):
        """Pixels -> unit rays in the head frame."""
        return self.unproject(uv) @ self.R.T

    def project(self, head_pts):
        """Head-frame points (N,3) -> pixels (N,2) and depth along the optical axis (N,)."""
        d = np.atleast_2d(np.asarray(head_pts, float)) - self.origin
        c = d @ self.R           # R^T d, row-wise
        return self.project_cam(c), c[:, 2]

    def off_axis_deg(self, uv):
        return np.degrees(np.arccos(np.clip(self.unproject(uv)[:, 2], -1, 1)))


def load_calibration(persist: Path) -> dict[str, Camera]:
    """Same maths as calib.cpp load_calibration, from the sanitized snapshot copies."""
    rig = json.loads((persist / 'xrservice.json').read_text())
    dev = json.loads((persist / 'device_config.json').read_text())
    cad_from_cam0 = _pose(dev['cv']['cad_from_cal'], 1.0)
    cad_from_head = _pose(dev['head'], 1.0)
    head_from_cam0 = np.linalg.inv(cad_from_head) @ cad_from_cam0
    out = {}
    for c in rig['cameras']:
        kb = next(i for i in c['intrinsics'] if i['cameraModel'] == 'kb')
        head_from_cam = head_from_cam0 @ _pose(c['extrinsics'], 1e-3)
        out[c['sourceCamera']] = Camera(
            c['sourceCamera'], c['width'], c['height'], kb['fx'], kb['fy'], kb['cx'], kb['cy'],
            np.array([kb['k1'], kb['k2'], kb['k3'], kb['k4']]), head_from_cam[:3, :3].copy(), head_from_cam[:3, 3].copy())
    return out


def triangulate(origins, dirs, weights):
    """Weighted closest point to several rays; returns (point, rms distance). Matches calib.cpp."""
    A = np.zeros((3, 3)); b = np.zeros(3)
    for o, d, w in zip(origins, dirs, weights):
        P = np.eye(3) - np.outer(d, d)
        A += w * P
        b += w * P @ o
    p = np.linalg.solve(A, b) if abs(np.linalg.det(A)) > 1e-18 else np.zeros(3)
    s = 0.0
    for o, d in zip(origins, dirs):
        off = p - o
        perp = off - d * (off @ d)
        s += perp @ perp
    return p, math.sqrt(s / len(origins))


def triangulate_hand(cams, views):
    """views: list of (Camera, pts21 (21,2), presence). Returns (pts (21,3), residual at the 11th-smallest)."""
    pts = np.zeros((21, 3)); res = np.zeros(21)
    rays = [cam.ray(p) for cam, p, _ in views]
    for k in range(21):
        pts[k], res[k] = triangulate([cam.origin for cam, _, _ in views], [r[k] for r in rays], [w for _, _, w in views])
    return pts, float(np.sort(res)[10])


# ------------------------------------------------------------------------------- crops

def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def roi_from_points(p):
    """nets.cpp roi_from_points: the landmark model's next crop from 21 image points."""
    p = np.asarray(p, float)
    w = p[0]
    m = (p[5] + p[13]) * 0.5
    m = (m + p[9]) * 0.5
    rot = wrap_angle(math.pi / 2 - math.atan2(-(m[1] - w[1]), m[0] - w[0]))
    sel = p[list(ROI_LANDMARKS)]
    center = (sel.min(0) + sel.max(0)) * 0.5
    c, s = math.cos(-rot), math.sin(-rot)
    d = sel - center
    q = np.stack((d[:, 0] * c - d[:, 1] * s, d[:, 0] * s + d[:, 1] * c), -1)
    qlo, qhi = q.min(0), q.max(0)
    mid = (qlo + qhi) * 0.5
    c2, s2 = math.cos(rot), math.sin(rot)
    center = center + np.array([mid[0] * c2 - mid[1] * s2, mid[0] * s2 + mid[1] * c2])
    w2, h2 = qhi[0] - qlo[0], qhi[1] - qlo[1]
    center = center + np.array([-h2 * -0.1 * s2, h2 * -0.1 * c2])
    return center, max(w2, h2) * 2.0, rot


def crop_matrix(center, size, rotation, out):
    """2x3 affine from crop pixels to image pixels (nets.cpp crop_matrix)."""
    c, s, k = math.cos(rotation), math.sin(rotation), size / out
    tm = np.zeros((2, 3))
    tm[0, 0], tm[0, 1], tm[1, 0], tm[1, 1] = c * k, -s * k, s * k, c * k
    tm[0, 2] = center[0] - (tm[0, 0] + tm[0, 1]) * out / 2.0
    tm[1, 2] = center[1] - (tm[1, 0] + tm[1, 1]) * out / 2.0
    return tm


def to_image(tm, xy):
    xy = np.atleast_2d(xy)
    return xy @ tm[:, :2].T + tm[:, 2]


def to_crop(tm, uv):
    """Inverse of to_image: image pixels -> crop pixels."""
    inv = np.linalg.inv(np.vstack((tm, [0, 0, 1])))
    uv = np.atleast_2d(uv)
    return uv @ inv[:2, :2].T + inv[:2, 2]


def clahe_4x4(img, clip_limit=2.0):
    """nets.cpp clahe(): OpenCV-style CLAHE with 4x4 tiles on a square uint8 crop."""
    n = img.shape[0]
    tiles = 4
    ts = n // tiles
    area = ts * ts
    clip = max(1, int(clip_limit * area / 256))
    lut = np.zeros((tiles, tiles, 256), np.uint8)
    for ty in range(tiles):
        for tx in range(tiles):
            hist = np.bincount(img[ty * ts:(ty + 1) * ts, tx * ts:(tx + 1) * ts].ravel(), minlength=256).astype(np.int64)
            excess = int(np.maximum(hist - clip, 0).sum())
            hist = np.minimum(hist, clip)
            add, residual = excess // 256, excess - (excess // 256) * 256
            hist += add
            hist[:residual] += 1
            cdf = np.cumsum(hist)
            scale = np.float32(255.0 / area)
            lut[ty, tx] = np.minimum(255, (cdf.astype(np.float32) * scale + np.float32(0.5)).astype(np.int32)).astype(np.uint8)
    ys = np.arange(n, dtype=np.float32)
    fy = (ys + 0.5) / ts - 0.5
    y0 = np.clip(np.floor(fy).astype(int), 0, tiles - 1); y1 = np.minimum(y0 + 1, tiles - 1); wy = np.clip(fy - y0, 0, 1)
    fx = (ys + 0.5) / ts - 0.5
    x0 = np.clip(np.floor(fx).astype(int), 0, tiles - 1); x1 = np.minimum(x0 + 1, tiles - 1); wx = np.clip(fx - x0, 0, 1)
    v = img
    top = lut[y0[:, None], x0[None, :], v] * (1 - wx[None, :]) + lut[y0[:, None], x1[None, :], v] * wx[None, :]
    bot = lut[y1[:, None], x0[None, :], v] * (1 - wx[None, :]) + lut[y1[:, None], x1[None, :], v] * wx[None, :]
    return (top * (1 - wy[:, None]) + bot * wy[:, None] + 0.5).astype(np.uint8)


def crop_patch(img, tm, out):
    """Bilinear warp as ncnn::warpaffine_bilinear_c1 (dst->src matrix, zero border)."""
    import cv2
    return cv2.warpAffine(img, tm.astype(np.float64), (out, out), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def model_input(patch):
    """Grey patch -> (3,n,n) float32 in 0..1, the planar input the ncnn models take."""
    f = patch.astype(np.float32) / 255.0
    return np.ascontiguousarray(np.stack((f, f, f), 0))


# ------------------------------------------------------------------------------ recordings

class Recording:
    """Index of a private FHSET01 recording; frames read on demand, never copied."""

    def __init__(self, path):
        self.path = Path(path)
        self.index = []     # (raw_ns, offset_of_pixels, rows)
        length = self.path.stat().st_size
        with self.path.open('rb') as f:
            while f.tell() < length:
                offset = f.tell()
                header = f.read(SET.size)
                if len(header) != SET.size:
                    raise ValueError('Truncated set header')
                magic, count, size = SET.unpack(header)
                if magic != b'FHSET01\0' or not 1 <= count <= 4:
                    raise ValueError('Invalid set header')
                meta = f.read(count * CAM.size)
                rows = [CAM.unpack_from(meta, i * CAM.size) for i in range(count)]
                if size != SET.size + count * CAM.size + sum(w * h for _, w, h, _, _ in rows) or offset + size > length:
                    raise ValueError('Invalid record size')
                stamp = min(r[3] for r in rows)
                self.index.append((stamp, offset + SET.size + count * CAM.size, rows))
                f.seek(offset + size)
        self.by_stamp = {s: i for i, (s, _, _) in enumerate(self.index)}
        self._f = None

    def __len__(self):
        return len(self.index)

    def stamps(self):
        return [s for s, _, _ in self.index]

    def read(self, i):
        """Set i -> {camera: uint8 (h,w)}."""
        if self._f is None:
            self._f = self.path.open('rb')
        stamp, offset, rows = self.index[i]
        self._f.seek(offset)
        images = {}
        for name, w, h, capture, dequeue in rows:
            buf = self._f.read(w * h)
            images[name.rstrip(b'\0').decode('ascii')] = np.frombuffer(buf, np.uint8).reshape(h, w)
        return images

    def read_stamp(self, raw_ns):
        return self.read(self.by_stamp[raw_ns])


def read_audit(path):
    """live-views.txt -> {raw_ns: {'mono_ns', 'hands': {id: (right, stereo)}, 'views': [dict]}}.

    Each view: camera, id, presence, right, pts (21,2) image pixels. These are the LIVE
    compact-model outputs recorded on the device, one row per processed crop.
    """
    out = {}
    cur = None
    for line in Path(path).read_text().splitlines():
        f = line.split()
        if not f:
            continue
        if f[0] == 'set':
            cur = dict(raw_ns=int(f[1]), mono_ns=int(f[2]), count=int(f[3]), hands={}, views=[])
        elif f[0] == 'hand':
            cur['hands'][int(f[1])] = (float(f[2]), int(f[3]))
        elif f[0] == 'view':
            if len(f) != 47 or cur is None:
                continue            # partial trailing row of a live file still being written
            cur['views'].append(dict(camera=f[1], id=int(f[2]), presence=float(f[3]), right=float(f[4]),
                                     pts=np.array(list(map(float, f[5:47]))).reshape(21, 2)))
        elif f[0] == 'end':
            if cur is not None:
                out[cur['raw_ns']] = cur
            cur = None
    return out


# --------------------------------------------------------------------- palm conventions

# The detector's 7 keypoints, as landmark indices (fit_palm_convention.py on 030445Z):
# wrist, index MCP, middle MCP, ring MCP, pinky MCP, thumb CMC, thumb MCP.
PALM_KP_LANDMARKS = (0, 5, 9, 13, 17, 1, 2)
PALM_EXTENT_LANDMARKS = (0, 1, 2, 5, 9, 13, 17)
# box centre = palm centroid + (right, up) * extent; box side = factor * extent (square)
PALM_BOX_OFFSET = (-0.037, -0.088)
PALM_BOX_SCALE = 1.037


def palm_box_from_landmarks(pts):
    """Image-space palm box (centre (2,), side) and 7 keypoints (7,2) in the detector's convention."""
    pts = np.asarray(pts, float)
    palm = pts[list(PALM_EXTENT_LANDMARKS)]
    cen = palm.mean(0)
    up = pts[9] - pts[0]
    up = up / (np.linalg.norm(up) + 1e-9)
    right = np.array([up[1], -up[0]])
    ext = max(np.ptp(palm @ right), np.ptp(palm @ up))
    center = cen + right * PALM_BOX_OFFSET[0] * ext + up * PALM_BOX_OFFSET[1] * ext
    return center, PALM_BOX_SCALE * ext, pts[list(PALM_KP_LANDMARKS)]


def decode_palms(boxes18, logits, anchors, threshold):
    """Raw detector outputs (N,18),(N,) -> tracker-style weighted-NMS palms in crop pixels.

    Each: dict(c (2,), s (2,), kp (7,2), score). Mirrors nets.cpp Nets::palms.
    """
    min_logit = math.log(threshold / (1 - threshold))
    idx = np.nonzero(logits > min_logit)[0]
    cands = []
    for i in idx:
        r = boxes18[i]
        cands.append(dict(c=np.array([r[0] + anchors[i, 0], r[1] + anchors[i, 1]]), s=np.array([r[2], r[3]]),
                          kp=np.array([[r[4 + 2 * k] + anchors[i, 0], r[5 + 2 * k] + anchors[i, 1]] for k in range(7)]),
                          score=1 / (1 + math.exp(-float(np.clip(logits[i], -100, 100))))))
    cands.sort(key=lambda d: -d['score'])
    used = [False] * len(cands)
    out = []
    for i, ci in enumerate(cands):
        if used[i]:
            continue
        wsum = 0.0; acc_c = np.zeros(2); acc_s = np.zeros(2); acc_kp = np.zeros((7, 2))
        for j, cj in enumerate(cands):
            if used[j]:
                continue
            if j != i:
                ix = max(0.0, min(ci['c'][0] + ci['s'][0] / 2, cj['c'][0] + cj['s'][0] / 2) - max(ci['c'][0] - ci['s'][0] / 2, cj['c'][0] - cj['s'][0] / 2))
                iy = max(0.0, min(ci['c'][1] + ci['s'][1] / 2, cj['c'][1] + cj['s'][1] / 2) - max(ci['c'][1] - ci['s'][1] / 2, cj['c'][1] - cj['s'][1] / 2))
                inter = ix * iy
                uni = ci['s'][0] * ci['s'][1] + cj['s'][0] * cj['s'][1] - inter
                if inter / (uni + 1e-9) <= 0.3:
                    continue
            used[j] = True
            w = cj['score']
            wsum += w; acc_c += cj['c'] * w; acc_s += cj['s'] * w; acc_kp += cj['kp'] * w
        out.append(dict(c=acc_c / wsum, s=acc_s / wsum, kp=acc_kp / wsum, score=ci['score']))
    return out


# ------------------------------------------------------------------------------- ncnn

def palm_anchors():
    anchors = []
    for stride, per in ((8, 2), (16, 6)):
        n = PALM_SIZE // stride
        for y in range(n):
            for x in range(n):
                for _ in range(per):
                    anchors.append(((x + 0.5) / n * PALM_SIZE, (y + 0.5) / n * PALM_SIZE))
    return np.array(anchors)


class NcnnNets:
    """The deployed palm/hand ncnn models, CPU, with the tracker's preprocessing."""

    def __init__(self, models_dir, int8=False, fp16=True, threads=1):
        import ncnn
        self.ncnn = ncnn
        suffix = '-int8.ncnn' if int8 else '.ncnn'
        self.palm = self._load(str(Path(models_dir) / ('palm' + suffix)), fp16, threads)
        self.hand = self._load(str(Path(models_dir) / ('hand' + suffix)), fp16, threads)
        self.anchors = palm_anchors()

    def _load(self, base, fp16, threads):
        net = self.ncnn.Net()
        net.opt.use_vulkan_compute = False
        net.opt.num_threads = threads
        net.opt.use_fp16_packed = net.opt.use_fp16_storage = net.opt.use_fp16_arithmetic = fp16
        if net.load_param(base + '.param') or net.load_model(base + '.bin'):
            raise RuntimeError('cannot load ' + base)
        return net

    def _run(self, net, chw, outputs):
        ex = net.create_extractor()
        ex.input('in0', self.ncnn.Mat(chw))
        res = []
        for name in outputs:
            ret, m = ex.extract(name)
            if ret != 0:
                raise RuntimeError('extract failed: ' + name)
            res.append(np.array(m).reshape(-1).copy())
        return res

    def landmarks(self, img, center, size, rotation, contrast='none'):
        """Returns dict(pts (21,2) image px, crop_pts (21,3), presence, right, world (21,3), tm)."""
        tm = crop_matrix(center, size, rotation, HAND_SIZE)
        patch = crop_patch(img, tm, HAND_SIZE)
        if contrast == 'clahe':
            patch = clahe_4x4(patch)
        screen, presence, right, world = self._run(self.hand, model_input(patch), ('out0', 'out1', 'out2', 'out3'))
        screen = screen.reshape(21, 3)
        return dict(pts=to_image(tm, screen[:, :2]), crop_pts=screen, presence=float(presence[0]),
                    right=float(right[0]), world=world.reshape(21, 3), tm=tm, patch=patch)

    def palms(self, img, center, size, rotation, threshold=0.2, contrast='clahe'):
        """MediaPipe palm detector on one tile, with the tracker's weighted NMS. Returns list of dicts."""
        tm = crop_matrix(center, size, rotation, PALM_SIZE)
        patch = crop_patch(img, tm, PALM_SIZE)
        if contrast == 'clahe':
            patch = clahe_4x4(patch)
        boxes, logits = self._run(self.palm, model_input(patch), ('out0', 'out1'))
        out = []
        for d in decode_palms(boxes.reshape(-1, 18), logits, self.anchors, threshold):
            out.append(dict(center=to_image(tm, d['c'])[0], size=d['s'] * (size / PALM_SIZE), kp=to_image(tm, d['kp']),
                            score=d['score'], crop_center=d['c'], crop_size=d['s'], crop_kp=d['kp']))
        return out, tm, patch


def palm_to_roi(palm):
    """Palm::roi for a decoded palm dict (image-pixel keypoints and size)."""
    a, b = palm['kp'][0], palm['kp'][2]
    rot = wrap_angle(math.pi / 2 - math.atan2(-(b[1] - a[1]), b[0] - a[0]))
    h = palm['size'][1]
    shift = np.array([-h * -0.5 * math.sin(rot), h * -0.5 * math.cos(rot)])
    return palm['center'] + shift, max(palm['size']) * 2.6, rot
