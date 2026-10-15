"""Content-free diagnostics for Frametop FHHANDS1. Run on Frame; no image recording.

Measures published observations, not display latency or tracking accuracy.
No hand coordinates are retained in the report. Requires only standard Python.
"""
import argparse
import collections
import json
import math
import os
from pathlib import Path
import struct
import time

SIZE = 2656
HEADER = struct.Struct('<8sIIQQQII16x')
HAND = struct.Struct('<IIff63fI')


def decode(data, now_ns, max_age_ms=100):
    if len(data) != SIZE:
        raise ValueError('wrong packet size')
    magic, version, size, seq, capture, publish, count, capsules = HEADER.unpack_from(data)
    if magic != b'FHHANDS1' or version != 1 or size != SIZE:
        raise ValueError('unsupported packet layout')
    if seq % 2 or not seq:
        raise ValueError('incomplete packet')
    if count > 2 or capsules > 64:
        raise ValueError('invalid counts')
    if not 0 < capture <= publish <= now_ns:
        raise ValueError('invalid timestamps')
    if max_age_ms is not None and now_ns - capture > max_age_ms * 1e6:
        raise ValueError('stale packet')
    sides, identities, stereo = [], set(), 0
    for n in range(count):
        values = HAND.unpack_from(data, HEADER.size + n * HAND.size)
        identity, flags, confidence, reserved = values[:4]
        points = values[4:67]
        if flags & ~3 or not identity or values[67] > 64:
            raise ValueError('invalid hand metadata')
        if identity in identities:
            raise ValueError('duplicate hand identity')
        identities.add(identity)
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError('invalid confidence')
        if not all(math.isfinite(v) and abs(v) < 5 for v in points):
            raise ValueError('invalid geometry')
        side = 'right' if flags & 1 else 'left'
        if side in sides:
            raise ValueError('duplicate handedness')
        sides.append(side)
        stereo += bool(flags & 2)
    return dict(seq=seq, capture=capture, publish=publish, count=count,
                stereo=stereo, sides=sides, latency_ms=(publish-capture)/1e6,
                observation_age_ms=(now_ns-capture)/1e6)


def snapshot(fd):
    # The writer uses an odd/even sequence lock. Reject a concurrent write.
    before = os.pread(fd, 8, 16)
    data = os.pread(fd, SIZE, 0)
    after = os.pread(fd, 8, 16)
    if before != after or data[16:24] != before:
        raise ValueError('concurrent write')
    return data


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    return round(values[max(0, math.ceil(len(values)*fraction)-1)], 3)


class Observations:
    """Keep slow publications in latency tails; separately count usable polls.

    A fresh publication can contain an old observation. Seeing the same packet
    repeatedly adds no inference sample and cannot keep it alive indefinitely.
    """
    def __init__(self):
        self.report = dict(schema=2, metric='capture_to_publish_not_motion_to_photon',
                           samples=0, polls=0, two_hands_samples=0, stereo_hand_samples=0,
                           hand_samples=0, left_samples=0, right_samples=0,
                           fresh_hand_polls=0, stale_polls=0, duplicate_polls=0,
                           stale_publications=0, fresh_hand_samples=0,
                           note='Tracker confidence is track age, not calibrated probability. No accuracy verdict.')
        self.rejected = collections.Counter()
        self.latencies, self.hand_latencies, self.times, self.gaps = [], [], [], []
        self.last_seq = None

    def add(self, data, now_ns):
        self.report['polls'] += 1
        try:
            # No age cutoff here: retaining bad latency is essential for p95.
            packet = decode(data, now_ns, max_age_ms=None)
            fresh = packet['observation_age_ms'] <= 100
            self.report['stale_polls'] += not fresh
            self.report['fresh_hand_polls'] += fresh and packet['count'] > 0
            if packet['seq'] == self.last_seq:
                self.report['duplicate_polls'] += 1
                return
            if self.times and packet['capture'] <= self.times[-1]:
                raise ValueError('nonincreasing capture timestamp')
            if self.times:
                self.gaps.append((packet['capture']-self.times[-1])/1e6)
            self.last_seq = packet['seq']
            self.report['samples'] += 1
            self.report['stale_publications'] += not fresh
            self.report['two_hands_samples'] += packet['count'] == 2
            self.report['hand_samples'] += packet['count']
            self.report['fresh_hand_samples'] += packet['count'] if fresh else 0
            self.report['stereo_hand_samples'] += packet['stereo']
            for side in packet['sides']:
                self.report[side+'_samples'] += 1
            self.latencies.append(packet['latency_ms'])
            if packet['count']:
                self.hand_latencies.append(packet['latency_ms'])
            self.times.append(packet['capture'])
        except ValueError as exc:
            self.rejected[str(exc)] += 1

    def finish(self, elapsed):
        def distribution(values):
            return dict(samples=len(values), p50=percentile(values,.5),
                        p95=percentile(values,.95), maximum=max(values,default=None))
        out = dict(self.report, rejected=dict(self.rejected), elapsed_seconds=round(elapsed,3))
        out['unique_updates_hz'] = round((len(self.times)-1)*1e9/(self.times[-1]-self.times[0]),3) if len(self.times)>1 else None
        out['capture_to_publish_ms'] = distribution(self.latencies)
        out['hand_capture_to_publish_ms'] = distribution(self.hand_latencies)
        out['update_gap_ms'] = distribution(self.gaps)
        out['has_live_hand_evidence'] = out['fresh_hand_samples'] > 0
        # This is evidence collection success, never the user's acceptance gate.
        out['acceptance_verified'] = False
        return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=20)
    parser.add_argument('--path', type=Path, default=Path(f'/run/user/{os.getuid()}/frametop-hands/hands'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 0 < args.seconds <= 300:
        parser.error('--seconds must be in (0,300]')
    observations = Observations()
    error = None
    start = time.monotonic()
    fd = None
    try:
        fd = os.open(args.path, os.O_RDONLY | os.O_NOFOLLOW)
        if os.fstat(fd).st_size != SIZE:
            raise ValueError('wrong mapped-file size')
        while time.monotonic() - start < args.seconds:
            try:
                observations.add(snapshot(fd), time.monotonic_ns())
            except ValueError as exc:
                observations.report['polls'] += 1
                observations.rejected[str(exc)] += 1
            time.sleep(.002)
    except (OSError, ValueError) as exc:
        error = str(exc)
    finally:
        if fd is not None:
            os.close(fd)
    report = observations.finish(time.monotonic()-start)
    if error:
        report['error'] = error
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text+'\n', encoding='utf-8')
    print(text)
    return 1 if 'error' in report or not report['samples'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
