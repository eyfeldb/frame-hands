"""Read only ring/hand metadata when checking whether a trial has started."""
import os
import struct
import time
from measure import decode, snapshot, SIZE

RING_HEADER = struct.Struct('<8sIIIIQqQ16x')
RING_BYTES = 1344


def ring_ready(path, expected_pid, started_ns, require_forward=False):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            data = os.pread(fd, RING_BYTES, 0)
            size = os.fstat(fd).st_size
        finally:
            os.close(fd)
        if len(data) != RING_BYTES:
            return False
        magic, version, header, count, _, file_bytes, pid, heartbeat = RING_HEADER.unpack_from(data)
        now = time.monotonic_ns()
        if (magic != b'FHRING01' or version != 1 or header != RING_BYTES
                or not 2 <= count <= 8 or file_bytes != size or pid != expected_pid
                or not started_ns <= heartbeat <= now or now-heartbeat > 500000000):
            return False
        # Require published observations from at least two mono cameras.
        # This is transport readiness, not proof of correct camera calibration.
        ready = 0
        forward = 0
        for n in range(count):
            offset = 64+160*n
            latest = struct.unpack_from('<Q',data,offset+104)[0]
            flags = struct.unpack_from('<I',data,offset+128)[0]
            ready += latest > 0 and flags == 0
            width,height=struct.unpack_from('<II',data,offset+72)
            forward += latest > 0 and flags == 0 and (width,height)==(1056,1024)
        return ready >= 2 and (not require_forward or forward >= 2)
    except (OSError, ValueError, struct.error):
        return False


def hands_ready(path, started_ns):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            if os.fstat(fd).st_size != SIZE:
                return False
            data = snapshot(fd)
        finally:
            os.close(fd)
        packet = decode(data,time.monotonic_ns(),max_age_ms=None)
        return packet['publish'] >= started_ns and packet['capture'] >= started_ns
    except (OSError,ValueError):
        return False
