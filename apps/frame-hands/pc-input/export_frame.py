"""Bounded request/reply reader over an already authenticated adb/SSH pipe.

Only 21 optical landmarks per hand leave Frame; no camera pixels or calibration.
It does not start tracking, create sockets, inject input, or modify SteamVR.
"""
import argparse
import base64
import os
from pathlib import Path
import select
import stat
import struct
import sys
import time

SIZE = 2656


def snapshot(path):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_size != SIZE:
                return bytes(SIZE)
            before = os.pread(fd, 8, 16)
            data = os.pread(fd, SIZE, 0)
            after = os.pread(fd, 8, 16)
            if len(data) != SIZE or before != after or before != data[16:24] or not int.from_bytes(before, 'little') or before[0] & 1:
                return bytes(SIZE)
            return data
        finally:
            os.close(fd)
    except OSError:
        return bytes(SIZE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=int, default=90)
    args = parser.parse_args()
    if not 5 <= args.seconds <= 3600:
        parser.error('seconds must be 5..3600')
    end = time.monotonic() + args.seconds
    path = Path(f'/run/user/{os.getuid()}/frametop-hands/hands')
    while time.monotonic() < end:
        ready, _, _ = select.select([sys.stdin.buffer], [], [], min(1, max(0, end-time.monotonic())))
        if not ready:
            continue
        if sys.stdin.buffer.read(1) != b'R':
            break
        data = snapshot(path)
        # ADB's host shell can apply newline conversion even without a PTY.
        # One bounded ASCII frame avoids corrupting arbitrary float bytes.
        sys.stdout.buffer.write(base64.b64encode(struct.pack('<8sQ', b'FHREAD1\0', time.monotonic_ns()) + data)+b'\n')
        sys.stdout.buffer.flush()


if __name__ == '__main__':
    main()
