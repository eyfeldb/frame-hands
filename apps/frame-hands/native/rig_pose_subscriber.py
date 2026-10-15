"""Bounded Frame-local rig/head-pose subscriber. No camera or XR scene startup.

Run under the supervised launcher's shared workflow lease, alongside its tracker.
Owns only its background pose helper and a NEW output directory. Python stdlib only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import queue
import subprocess
import threading
import time
from rig_pose_bridge import RigPoseBridge, IDENTITY, profile_limits, positive_integer

MAX_BYTES = 65536
MAX_AGE_NS = 250_000_000


def decode(data):
    if len(data) > MAX_BYTES:
        raise ValueError('Packet too large')
    def reject_constant(value):
        raise ValueError("Nonfinite JSON constant: "+value)
    result = json.loads(data, parse_constant=reject_constant)
    if not isinstance(result, dict):
        raise ValueError('Packet must be an object')
    return result


def atomic_json(path, value):
    data = json.dumps(value, allow_nan=False, separators=(',', ':'))+'\n'
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w', encoding='utf-8') as output:
        output.write(data)
    os.replace(temporary, path)


class PoseProcess:
    """One request at a time. A timeout makes this transport permanently unusable."""
    def __init__(self, command, stderr, timeout=.10):
        self.timeout = timeout
        self.failed = False
        self.lines = queue.Queue(maxsize=2)
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=stderr, bufsize=0)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        # A rejected zero timestamp confirms OpenVR initialization is complete.
        # Give startup its own bound; ordinary pose requests stay short.
        try:
            self.process.stdin.write(b'0\n')
            self.process.stdin.flush()
            ready = decode(self.lines.get(timeout=3.0))
            if ready.get('valid') is not False or ready.get('reason') != 'invalid_or_stale_capture_mono_ns':
                raise ValueError('Unexpected pose-helper startup response')
        except (OSError, ValueError, queue.Empty):
            self.failed = True
            self.close()
            raise RuntimeError('Pose helper failed startup handshake') from None

    def _read(self):
        while True:
            line = self.process.stdout.readline(MAX_BYTES+1)
            try:
                self.lines.put_nowait(line)
            except queue.Full:
                self.failed = True
                return
            if not line or len(line) > MAX_BYTES:
                return

    def query(self, capture):
        if self.failed or self.process.poll() is not None:
            raise RuntimeError('Pose helper unavailable')
        try:
            self.process.stdin.write((str(positive_integer(capture))+'\n').encode('ascii'))
            self.process.stdin.flush()
            line = self.lines.get(timeout=self.timeout)
            if not line:
                raise RuntimeError('Pose helper exited')
            result = decode(line)
            if result.get('valid') is True and result.get('capture_mono_ns') != capture:
                raise ValueError('Pose response belongs to another capture')
            return result
        except (OSError, ValueError, queue.Empty, RuntimeError):
            self.failed = True
            raise RuntimeError('Pose helper query failed; restart required') from None

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write(b'Q\n')
                self.process.stdin.flush()
                self.process.wait(timeout=.2)
            except (OSError, subprocess.TimeoutExpired):
                self.process.terminate()
                try:
                    self.process.wait(timeout=.5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=1)
        self.reader.join(timeout=1)
        self.process.stdin.close()
        self.process.stdout.close()


class Subscriber:
    def __init__(self, limits):
        self.bridge = RigPoseBridge(limits)
        self.sequence = 0
        self.capture = 0
        self.active_context = False
        self.last_bytes = None
        self.restart_required = False

    def invalid(self, reason, now, wall):
        self.sequence += 1
        self.active_context = False
        return dict(version=2, sequence=self.sequence, capture_ns=self.capture,
            published_wall_s=wall, age_ms=max(0,(now-self.capture)/1e6),
            context_valid=False, head_world_m=IDENTITY, hands=[], reason=reason)

    def consume(self, data, query, monotonic_ns=time.monotonic_ns, wall_time=time.time):
        if self.restart_required or data == self.last_bytes:
            return None, None
        self.last_bytes = data
        head = None
        try:
            snapshot = decode(data)
            capture = positive_integer(snapshot.get('capture_mono_ns'))
            sequence = positive_integer(snapshot.get('sequence'))
            # Never accept an old file after a producer restart. A new supervised
            # session gets a new subscriber/output directory and fresh source IDs.
            if sequence <= self.bridge.last_source_sequence:
                if capture > self.bridge.last_capture:
                    self.restart_required = True
                    raise ValueError('Producer restarted; new session required')
                return None, None
            if capture <= self.bridge.last_capture:
                return None, None
            now = monotonic_ns()
            published = positive_integer(snapshot.get('published_mono_ns'))
            if capture <= published <= now and now-capture <= MAX_AGE_NS:
                head = query(capture)
            else:
                head = dict(valid=False, reason='stale_before_query')
            packet = self.bridge.convert(snapshot, head, monotonic_ns(), wall_time())
            if packet is None:
                return None, head
            self.sequence += 1
            packet['sequence'] = self.sequence
            self.capture = capture
            self.active_context = packet['context_valid']
            return packet, head
        except (ValueError, KeyError, TypeError, OverflowError) as error:
            return self.invalid('invalid_snapshot_or_pose: '+str(error), monotonic_ns(), wall_time()), head

    def expire(self, now, wall):
        if self.active_context and now-self.capture > MAX_AGE_NS:
            return self.invalid('tracker_stalled', now, wall)
        return None


def verified_limits(profiles, viewer):
    index = decode((viewer/'profile-rigs.json').read_bytes())
    if index.get('version') != 2 or len(index.get('profiles', [])) != 2:
        raise ValueError('Invalid viewer profile index')
    limits = []
    for side, entry in enumerate(index['profiles']):
        profile = profiles/f'side{side}.fhrig'
        if entry.get('side') != side or entry.get('file') != f'side{side}.json':
            raise ValueError('Wrong viewer side')
        if hashlib.sha256(profile.read_bytes()).hexdigest() != entry.get('profile_sha256'):
            raise ValueError('Tracker/viewer anatomy mismatch')
        if hashlib.sha256((viewer/entry['file']).read_bytes()).hexdigest() != entry.get('sha256'):
            raise ValueError('Changed viewer mesh')
        limits.append(profile_limits(profile, side))
    return limits


def run(snapshot_path, output, subscriber, helper, seconds):
    """Caller supplies a NEW owned output directory and a running bounded helper."""
    started = time.monotonic()
    packets = 0
    with (output/'trace.jsonl').open('x', encoding='utf-8') as trace:
        def publish(packet, source=None, head=None):
            nonlocal packets
            atomic_json(output/'pose.json', packet)
            trace.write(json.dumps(dict(mono_ns=time.monotonic_ns(), packet=packet,
                snapshot_utf8=source.decode('utf-8', errors='replace') if source else None,
                head=head), allow_nan=False)+'\n')
            trace.flush()
            packets += 1
        publish(subscriber.invalid('starting', time.monotonic_ns(), time.time()))
        reason = 'duration_complete'
        try:
            while time.monotonic()-started < seconds:
                try:
                    with snapshot_path.open('rb') as source:
                        data = source.read(MAX_BYTES+1)
                except FileNotFoundError:
                    data = None
                if data is not None:
                    try:
                        packet, head = subscriber.consume(data, helper.query)
                    except RuntimeError:
                        publish(subscriber.invalid('pose_helper_failed',time.monotonic_ns(),time.time()),data)
                        raise
                    if packet is not None:
                        publish(packet, data, head)
                packet = subscriber.expire(time.monotonic_ns(), time.time())
                if packet is not None:
                    publish(packet)
                if helper.process.poll() is not None:
                    raise RuntimeError('Pose helper stopped')
                time.sleep(.008)
        except BaseException:
            reason = 'subscriber_failed_or_interrupted'
            raise
        finally:
            publish(subscriber.invalid(reason, time.monotonic_ns(), time.time()))
            helper.close()
    return packets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('snapshot','profiles','viewer','pose-service','openvr','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=90)
    args = parser.parse_args()
    if platform.system() != 'Linux' or platform.machine() != 'aarch64':
        raise ValueError('Live subscriber requires native Linux ARM64')
    if not 1 <= args.seconds <= 295:
        raise ValueError('Bounded 1..295 seconds required')
    limits = verified_limits(args.profiles, args.viewer)
    for path in (args.pose_service, args.openvr):
        if not path.is_absolute() or not path.is_file():
            raise ValueError('Existing absolute helper/library paths required')
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output/'pose.stderr').open('wb') as stderr:
        helper = PoseProcess([str(args.pose_service),str(args.openvr),str(args.seconds+5)], stderr)
        try:
            count = run(args.snapshot,args.output,Subscriber(limits),helper,args.seconds)
        finally:
            helper.close()
    print(json.dumps(dict(packets=count, output=str(args.output), scope='Native diagnostic pose publication only')))

if __name__ == '__main__':
    main()
