"""Overlay actual live 2D landmarks on their exact recorded stereo image set.

Run on Frame with its isolated Python dependencies. Images remain in the private
recording directory unless explicitly copied. Replay predictions are not used.
"""
import argparse
import json
import math
from pathlib import Path
import struct
import sys

BASE = Path('/home/steamos/devkit-game/FrameHands')
sys.path.insert(0, str(BASE/'deps/python'))
import cv2
import numpy as np

HEADER = struct.Struct('<8sII')
CAMERA = struct.Struct('<16sIIQQ')
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7),
         (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13),
         (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
         (17, 18), (18, 19), (19, 20)]


def read_audit(path):
    observations = {}
    current = None
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == 'set':
            current = dict(raw_ns=int(fields[1]), mono_ns=int(fields[2]),
                           count=int(fields[3]), hands={}, views=[])
        elif current is not None and fields[0] == 'hand':
            current['hands'][int(fields[1])] = dict(right=float(fields[2]), stereo=int(fields[3]))
        elif current is not None and fields[0] == 'view':
            if len(fields) != 47:
                raise ValueError('Invalid landmark row')
            coords = list(map(float, fields[5:]))
            if not all(math.isfinite(x) and abs(x) < 1e6 for x in coords):
                raise ValueError('Invalid landmark coordinate')
            current['views'].append(dict(camera=fields[1], hand=int(fields[2]),
                                        presence=float(fields[3]), right=float(fields[4]),
                                        points=np.array(coords).reshape(21, 2)))
        elif current is not None and fields[0] == 'end':
            if current['count'] != len(current['hands']):
                raise ValueError('Incomplete hand record')
            observations[current['raw_ns']] = current
            current = None
    return observations


def read_sets(path):
    with path.open('rb') as stream:
        while header := stream.read(HEADER.size):
            if len(header) != HEADER.size:
                raise ValueError('Truncated set header')
            magic, count, size = HEADER.unpack(header)
            if magic != b'FHSET01\0' or not 1 <= count <= 4:
                raise ValueError('Invalid set header')
            camera_bytes = stream.read(CAMERA.size*count)
            if len(camera_bytes) != CAMERA.size*count:
                raise ValueError('Truncated camera header')
            cameras = [CAMERA.unpack_from(camera_bytes, i*CAMERA.size) for i in range(count)]
            if any(not 1 <= w <= 4096 or not 1 <= h <= 4096 for _, w, h, _, _ in cameras):
                raise ValueError('Invalid camera dimensions')
            if size != HEADER.size+CAMERA.size*count+sum(w*h for _, w, h, _, _ in cameras):
                raise ValueError('Invalid record size')
            images = {}
            for name, width, height, capture, dequeue in cameras:
                pixels = stream.read(width*height)
                if len(pixels) != width*height:
                    raise ValueError('Truncated image')
                images[name.rstrip(b'\0').decode()] = (np.frombuffer(pixels, np.uint8).reshape(height, width), capture)
            yield images


def draw(images, observation, seconds, label='LIVE'):
    raw_tiles, annotated_tiles = [], []
    for name in ('slam_left', 'slam_right'):
        if name not in images:
            continue
        pixels = images[name][0]
        raw = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
        annotated = cv2.cvtColor(cv2.createCLAHE(2, (8, 8)).apply(pixels), cv2.COLOR_GRAY2BGR)
        for view in observation['views']:
            if view['camera'] != name:
                continue
            hand = observation['hands'].get(view['hand'])
            side = ('R' if hand['right'] > .5 else 'L') if hand else 'unpublished'
            color = (0, 160, 255) if side == 'R' else (255, 220, 0) if side == 'L' else (140, 140, 140)
            points = np.rint(view['points']).astype(np.int32)
            for a, b in EDGES:
                cv2.line(annotated, tuple(points[a]), tuple(points[b]), color, 3, cv2.LINE_AA)
            for point in points:
                cv2.circle(annotated, tuple(point), 4, color, -1)
            joint_label = f"{side} id {view['hand']} presence {view['presence']:.2f}"
            x = max(4, min(int(points[0, 0]), pixels.shape[1]-420))
            y = max(60, min(int(points[0, 1]), pixels.shape[0]-12))
            cv2.putText(annotated, joint_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .7, color, 2)
        for image, caption in [(raw, 'raw'), (annotated, f'{label} joints / contrast enhanced')]:
            cv2.rectangle(image, (0, 0), (image.shape[1], 42), (0, 0, 0), -1)
            cv2.putText(image, f'{seconds:.2f}s {name} {caption}', (10, 29),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1)
        raw_tiles.append(cv2.resize(raw, (528, 512)))
        annotated_tiles.append(cv2.resize(annotated, (528, 512)))
    return np.vstack([np.hstack(raw_tiles), np.hstack(annotated_tiles)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--every', type=float, default=.5)
    parser.add_argument('--audit-file', default='live-views.txt')
    parser.add_argument('--label', choices=['LIVE','REPLAY'], default='LIVE')
    args = parser.parse_args()
    if args.every < .1:
        parser.error('--every must be at least 0.1 seconds')
    if args.audit_file != 'live-views.txt' and args.label == 'LIVE':
        parser.error('Alternative observations must be labeled REPLAY')
    observations = read_audit(args.directory/args.audit_file)
    output = args.directory/('review' if args.label=='LIVE' else 'replay-review')
    output.mkdir(mode=0o700, exist_ok=True)
    matched = set()
    samples = []
    start = None
    next_sample = 0
    records = 0
    for images in read_sets(args.directory/'sets.bin'):
        records += 1
        timestamp = min(t for _, t in images.values())
        if start is None:
            start = timestamp
        observation = observations.get(timestamp)
        if observation is None:
            continue
        matched.add(timestamp)
        seconds = (timestamp-start)/1e9
        if seconds < next_sample:
            continue
        filename = f'frame-{records:05d}.jpg'
        if not cv2.imwrite(str(output/filename), draw(images, observation, seconds,label=args.label)):
            raise OSError('Could not write diagnostic image')
        samples.append(dict(image=filename, seconds=seconds, hands=observation['count'],
                            raw_capture_ns=timestamp))
        next_sample = seconds+args.every
    report = dict(recorded_sets=records, observation_source=args.label, observations=len(observations),
                  matched_observations=len(matched), unmatched_observations=len(observations)-len(matched),
                  kind='Exact timestamp overlays; visual review required, not accuracy ground truth', samples=samples)
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
