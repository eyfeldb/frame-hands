"""Isolated untimed source adaptation and bounded live-spool storage.

No camera ownership is acquired by preparing a launcher. Existing exact-v3
binary/geometry/lock checks are retained. Only --continuous opts into this path.
"""
import ctypes
import os
from pathlib import Path
import shutil


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Unexpected launcher source: ' + repr(old[:100]))
    return text.replace(old, new)


def prepare(original, output):
    original, output = Path(original), Path(output)
    if output.exists():
        raise ValueError('Use a fresh isolated launcher directory')
    trial = (original/'trial.py').read_text()
    source = (original/'pc-input/run_source.py').read_text()
    trial = replace_once(trial, "    args = parser.parse_args()", "    parser.add_argument('--continuous', action='store_true')\n    args = parser.parse_args()\n    if args.continuous and not (args.pc_session and args.record_session):parser.error('Continuous mode requires the PC live spool')")
    trial = replace_once(trial, 'if not 5 <= args.seconds <= (300 if args.pc_session else 90):', 'if not args.continuous and not 5 <= args.seconds <= (300 if args.pc_session else 90):')
    trial = replace_once(trial, 'if recording_requested and shutil.disk_usage(BASE).free < required_bytes:', 'if args.continuous:required_bytes=2_000_000_000\n    if recording_requested and shutil.disk_usage(BASE).free < required_bytes:')
    trial = replace_once(trial, "'--seconds',str(args.seconds+8)]", "'--seconds',str(0 if args.continuous else args.seconds+8)]")
    trial = replace_once(trial, "str(args.seconds+5)])", "('inf' if args.continuous else str(args.seconds+5))])")
    start = trial.index('            measurement=subprocess.Popen(')
    end = trial.index("            result['camera_running_at_end']", start)
    bounded = trial[start:end]
    continuous = """            if args.continuous:
                phase('tracking', detail='Optical hands running until Stop or disconnect')
                result['live_spool_only']=True
                while True:
                    if camera.poll() is not None or tracker.poll() is not None:
                        raise RuntimeError('A tracking worker stopped')
                    if shutil.disk_usage(BASE).free < 512_000_000:
                        raise RuntimeError('Low free storage; stopping live tracking')
                    time.sleep(.2)
            else:
"""
    trial = trial[:start]+continuous+''.join('    '+line if line.strip() else line for line in bounded.splitlines(keepends=True))+trial[end:]
    trial = replace_once(trial, "            result['error']=str(exc)", "            if args.continuous and str(exc).startswith('Trial interrupted by signal '):\n                result['stopped_by_owner']=True\n                result['measurement_returncode']=0\n            else:result['error']=str(exc)")
    trial = replace_once(trial, "            if private:\n                recording=private/'sets.bin'", "            if private and not args.continuous:\n                recording=private/'sets.bin'")
    trial = replace_once(trial, "            if private:save_json(private/'result.json',result)", "            if private:save_json(private/'result.json',result)\n            if private and args.continuous and result['owned_processes_stopped']:\n                for name in ('sets.bin','live-views.txt'):(private/name).unlink(missing_ok=True)")
    source = replace_once(source, '    args=parser.parse_args()', "    parser.add_argument('--continuous',action='store_true')\n    args=parser.parse_args()")
    source = replace_once(source, "if not 5<=args.seconds<=300:", "if not args.continuous and not 5<=args.seconds<=300:")
    source = replace_once(source, '        child=subprocess.Popen(command)', "        if args.continuous:command.append('--continuous')\n        child=subprocess.Popen(command)")
    source = replace_once(source, 'if time.monotonic()>deadline:', 'if not args.continuous and time.monotonic()>deadline:')
    # The controlling PC renews a short lease. EOF/Q stops immediately; a hung
    # host cannot leave an untimed camera process running indefinitely.
    source = replace_once(source, '        while child.poll() is None:', '        last_lease=time.monotonic()\n        while child.poll() is None:\n            if args.continuous and time.monotonic()-last_lease>45:raise TimeoutError("PC tracking lease expired")')
    source = replace_once(source, "                if not command or command==b'Q':break", "                if not command or command==b'Q':break\n                if command==b'H':last_lease=time.monotonic()")
    compile(trial, 'trial.py', 'exec'); compile(source, 'run_source.py', 'exec')
    shutil.copytree(original, output, ignore=shutil.ignore_patterns('__pycache__'))
    (output/'trial.py').write_text(trial)
    (output/'pc-input/run_source.py').write_text(source)


class SparseSpool:
    """Release only fully consumed blocks; append offsets and recent data stay valid."""
    def __init__(self):
        self.reclaimed={}

    def release(self, path, consumed):
        path=Path(path)
        end=max(0, int(consumed)//4096*4096)
        start=self.reclaimed.get(str(path),0)
        if end-start<4*1024*1024:return
        fd=os.open(path,os.O_RDWR)
        try:
            if end>os.fstat(fd).st_size:raise ValueError('Cannot reclaim unwritten spool data')
            lib=ctypes.CDLL(None,use_errno=True)
            call=lib.fallocate
            call.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_longlong,ctypes.c_longlong]
            call.restype=ctypes.c_int
            # PUNCH_HOLE | KEEP_SIZE: never truncate or move the writer cursor.
            if call(fd,3,start,end-start):
                error=ctypes.get_errno();raise OSError(error,os.strerror(error))
        finally:os.close(fd)
        self.reclaimed[str(path)]=end


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('original',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();prepare(a.original,a.output)
