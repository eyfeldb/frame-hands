"""Snapshot the accepted diagnostic build; no camera/runtime changes or private images."""
import hashlib
import json
from pathlib import Path
import tarfile

base = Path('/home/steamos/devkit-game/FrameHands')
name = '20261003-identity-accepted'
destination = base/'checkpoints'/name
destination.mkdir(parents=True, exist_ok=True)
assert not any(destination.iterdir()), 'Refusing to overwrite an existing checkpoint'
files = [base/p for p in [
    'trial.py', 'probe/measure.py', 'check_camera_geometry.py', 'apply_frame_patch.py',
    'build-viewer.sh', 'build-on-frame.sh', 'probe/readiness.py', 'probe/scene-check',
    'upstream/build/ft-hands-identity-latch', 'upstream/Makefile',
    'viewer/libframe_hands.so', 'viewer/frame_hands.gdextension',
    'viewer/openxr_action_map.tres',
]]
for pattern in ['viewer/*.gd', 'viewer/*.cpp', 'viewer/*.tscn', 'viewer/project.godot',
                'upstream/track/*.cpp', 'upstream/track/*.h', 'upstream/include/*.h',
                'upstream/camd/*.h']:
    files.extend(sorted(base.glob(pattern)))
for model in ['hand', 'palm']:
    for suffix in ['bin', 'param']:
        files.append(base/f'upstream/models/ncnn/{model}.ncnn.{suffix}')
files = sorted(set(files))
hashes = {str(p.relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
assert hashes['upstream/build/ft-hands-identity-latch']=='052d39d1971c5886976d2204bdec2789e7e1a184ca2bb06a2625948260d53682'
assert hashes['viewer/smooth_hands.gd']=='46239e2dd006d82b5a380333922024440daa35a018b00bfa380ed004ac538de2'
manifest = dict(checkpoint=name, files=hashes, mode='standalone Linux ARM64',
                accepted_trial='20261003T203217Z',
                user_feedback='Pretty great; non-index/thumb fingers less accurate, lower priority for now.',
                scope='Accepted diagnostic tracking baseline; not general app input or full goal completion',
                runtime=dict(godot='4.7.2.stable.official.ed1daf0bf', steamvr='2.17.10',
                             steamos='0.3.0/20260922.6101926', ncnn='20260526'),
                external_dependencies=['Installed capability-enabled camera reader', 'Factory calibration in /persist',
                                       'FrameCinema/godot', 'FrameHands/deps/python', 'SteamVR/OpenXR'])
manifest_path = destination/'manifest.json'
manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
archive = destination/'accepted.tar.gz'
with tarfile.open(archive, 'x:gz') as output:
    for p in files:
        output.add(p, arcname=str(p.relative_to(base)), recursive=False)
    output.add(manifest_path, arcname='checkpoint-manifest.json')
digest=hashlib.sha256(archive.read_bytes()).hexdigest()
(destination/'archive-sha256.txt').write_text(digest+'  accepted.tar.gz\n')
print(json.dumps(dict(checkpoint=str(destination), files=len(hashes), archive_bytes=archive.stat().st_size,
                      archive_sha256=digest)))
