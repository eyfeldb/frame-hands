"""Private diagnostic snapshots and strict, dependency-free recording checks."""
import hashlib
import json
from pathlib import Path
import shutil
import struct
import platform
import subprocess

SET = struct.Struct('<8sII')
CAMERA = struct.Struct('<16sIIQQ')
TRACE_SIZE = 3136
RELAY = struct.Struct('<8sIIQQII')
RELAY_SIZE = RELAY.size + 2672 + 568


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def snapshot(files, destination):
    """Explicit allowlist only; never snapshot environment, account or device secrets."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for name, source in files.items():
        target = destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        manifest[name] = dict(sha256=digest(target), bytes=target.stat().st_size)
    save_json(destination/'files.json', manifest)
    return manifest


def snapshot_frame(base, private, tracker, camera):
    files = {'bin/tracker': tracker, 'bin/camera-reader': camera,
             'launchers/trial.py': base/'trial.py',
             'launchers/run_source.py': base/'pc-input/run_source.py',
             'launchers/export_frame.py': base/'pc-input/export_frame.py',
             'tools/recording.py': base/'recording.py',
             'tools/review_recording.py': base/'review_recording.py'}
    for path in (base/'upstream/models/ncnn').glob('*'):
        if path.is_file(): files['models/ncnn/'+path.name] = path
    source = tracker.parent/'upstream'
    if not source.is_dir(): source = base/'upstream'
    for name in ('manifest.json','trial-build.json'):
        if (tracker.parent/name).is_file():files['build/'+name]=tracker.parent/name
    for folder in ('track', 'include', 'camd'):
        for path in (source/folder).rglob('*'):
            if path.is_file() and path.suffix in ('.h', '.hpp', '.cpp'):
                files['source/'+str(path.relative_to(source)).replace('\\', '/')] = path
    version=Path('/opt/steamvr/bin/version.txt')
    if version.is_file():files['runtime/steamvr-version.txt']=version
    snapshot(files, private/'snapshot')
    # load_calibration uses only these camera/rig fields. Do not copy entire device configs.
    rig = json.loads(Path('/persist/xrservice.json').read_text())
    device = json.loads(Path('/persist/device_config.json').read_text())
    def pose(value): return {key: value[key] for key in ('plus_x', 'plus_z', 'position')}
    cameras = []
    for camera in rig['cameras']:
        if camera['sourceCamera'] not in ('slam_left', 'slam_right'): continue
        cameras.append(dict(sourceCamera=camera['sourceCamera'], width=camera['width'], height=camera['height'],
                            extrinsics=pose(camera['extrinsics']),
                            intrinsics=[{key: item[key] for key in ('cameraModel','fx','fy','cx','cy','k1','k2','k3','k4')}
                                        for item in camera['intrinsics'] if item['cameraModel']=='kb']))
    if len(cameras)!=2: raise ValueError('Expected both forward camera calibrations')
    persist = private/'snapshot/device/persist'
    persist.mkdir(parents=True)
    save_json(persist/'xrservice.json', dict(cameras=cameras))
    save_json(persist/'device_config.json', dict(cv=dict(cad_from_cal=pose(device['cv']['cad_from_cal'])), head=pose(device['head'])))
    save_json(persist/'hashes.json', {p.name:digest(p) for p in persist.glob('*.json')})
    save_json(private/'snapshot/runtime.json',dict(kernel=platform.release(),architecture=platform.machine(),
              os_release=Path('/etc/os-release').read_text(),
              dynamic_dependencies=subprocess.check_output(['ldd',str(tracker)],text=True)))


def inspect_camera(directory):
    """Scan headers without loading gigabytes of pixels. Catch partial files and missing audited sets."""
    directory = Path(directory)
    path = directory/'sets.bin'
    size = path.stat().st_size
    stamps = set(); records = 0; first = last = None; cameras_seen = set()
    with path.open('rb') as stream:
        while stream.tell()<size:
            start = stream.tell(); header = stream.read(SET.size)
            if len(header)!=SET.size: raise ValueError('Truncated camera set header')
            magic, count, total = SET.unpack(header)
            if magic!=b'FHSET01\0' or not 1<=count<=4: raise ValueError('Invalid camera set')
            raw = stream.read(CAMERA.size*count)
            if len(raw)!=CAMERA.size*count: raise ValueError('Truncated camera metadata')
            cameras = [CAMERA.unpack_from(raw, CAMERA.size*i) for i in range(count)]
            if any(not 1<=w<=4096 or not 1<=h<=4096 for _,w,h,_,_ in cameras): raise ValueError('Invalid image dimensions')
            expected = SET.size+CAMERA.size*count+sum(w*h for _,w,h,_,_ in cameras)
            if total!=expected or start+total>size: raise ValueError('Truncated or malformed camera pixels')
            names = [name.rstrip(b'\0').decode('ascii') for name,_,_,_,_ in cameras]
            if len(set(names))!=count: raise ValueError('Duplicate camera in set')
            cameras_seen.update(names)
            stamp = min(capture for _,_,_,capture,_ in cameras)
            if last is not None and stamp<=last: raise ValueError('Non-increasing camera timestamps')
            if first is None: first=stamp
            last=stamp; stamps.add(stamp); records+=1
            stream.seek(start+total)
    audited=[]; current=None; hands=0
    with (directory/'live-views.txt').open() as stream:
        for line in stream:
            fields=line.split()
            if not fields: continue
            if fields[0]=='set':
                if current is not None: raise ValueError('Incomplete live audit set')
                current=(int(fields[1]), int(fields[3])); hands=0
            elif fields[0]=='hand': hands+=1
            elif fields[0]=='end':
                if current is None or hands!=current[1]: raise ValueError('Incomplete live audit hands')
                audited.append(current[0]); current=None
    if current is not None: raise ValueError('Truncated live audit')
    if not records or not audited: raise ValueError('Empty camera or live audit recording')
    missing=sum(stamp not in stamps for stamp in audited)
    return dict(sets=records, bytes=size, cameras=sorted(cameras_seen), first_raw_ns=first, last_raw_ns=last,
                duration_seconds=(last-first)/1e9, audited_sets=len(audited), audited_sets_missing_images=missing,
                structurally_complete=True, all_audited_images_saved=missing==0,
                scope='Header/length and timestamp checks; not pixel integrity or proof of lossless acquisition')


def inspect_fixed(path, kind):
    count=0; previous=None; first=None
    size=TRACE_SIZE if kind=='trace' else RELAY_SIZE
    with Path(path).open('rb') as stream:
        while data:=stream.read(size):
            if len(data)!=size: raise ValueError('Truncated '+kind+' record')
            magic,version,length,stamp=struct.unpack_from('<8sIIQ',data)
            if (magic,version,length)!=(b'FHTRACE1' if kind=='trace' else b'FHRELAY1',1,size):
                raise ValueError('Invalid '+kind+' header')
            if previous is not None and stamp<previous: raise ValueError('Non-monotonic '+kind+' time')
            if kind=='trace' and data[1272:1280]!=b'FHVIEW2\0': raise ValueError('Invalid saved view')
            if kind=='relay' and data[RELAY.size:RELAY.size+8]!=b'FHREAD1\0': raise ValueError('Invalid saved source export')
            if first is None: first=stamp
            previous=stamp; count+=1
    if not count: raise ValueError('Empty '+kind+' recording')
    return dict(records=count, bytes=count*size, first_pc_ms=first, last_pc_ms=previous, structurally_complete=True)
