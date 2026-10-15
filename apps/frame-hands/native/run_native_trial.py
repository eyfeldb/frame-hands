"""Supervised, bounded standalone native hand diagnostic; no system input writes.

Requires an explicit --ready for this trial. Never installs, stops unrelated apps,
or changes runtime/driver defaults. All command arguments are lists, never a shell.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import struct
import subprocess
import sys
import time
from rig_pose_subscriber import atomic_json, decode, verified_limits
from workflow_guard import native_workflow

CAMERA_SHA='cd46259265d7e1df542ea78259c02ac4f42846263127f4606fa19b9725318990'
BASE=Path(__file__).resolve().parent
SOURCES=('run_native_trial.py','workflow_guard.py','rig_pose_subscriber.py','rig_pose_bridge.py')


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def verify_tree(root):
    manifest=json.loads((root/'manifest.json').read_text())
    for name,digest in manifest['sha256'].items():
        path=(root/name).resolve()
        if root.resolve() not in path.parents or sha(path)!=digest:
            raise ValueError('Changed staged input: '+name)
    return manifest


def configuration(path):
    config=decode(path.read_bytes())
    if config.get('version')!=1:raise ValueError('Unknown trial configuration')
    for name in SOURCES:
        if config.get('source_sha256',{}).get(name)!=sha(BASE/name):raise ValueError('Changed launcher source: '+name)
    files={}
    for name in ('camera','godot','pose_service','openvr','scene_check','geometry','xr_runtime','recording_tool'):
        entry=config['files'][name];file=Path(entry['path'])
        if not file.is_absolute() or not file.is_file() or sha(file)!=entry['sha256']:
            raise ValueError('Unverified trial file: '+name)
        files[name]=file
    if config['files']['camera']['sha256']!=CAMERA_SHA:
        raise ValueError('Reviewed exact-v3 camera reader required')
    build=Path(config['build']);viewer=Path(config['viewer'])
    if not build.is_absolute() or not viewer.is_absolute():raise ValueError('Absolute staged paths required')
    meta=json.loads((build/'build.json').read_text())
    if sha(build/'build.json')!=config['build_sha256'] or sha(viewer/'manifest.json')!=config['viewer_manifest_sha256']:
        raise ValueError('Changed build or viewer manifest')
    if sha(build/'ft-hands-native-rig')!=meta['binary_sha256'] or sha(build/'source/manifest.json')!=meta['provenance']['bundle_manifest_sha256']:
        raise ValueError('Live tracker is not the verified build')
    source_manifest=verify_tree(build/'source');verify_tree(viewer)
    if config.get('whole_pose') and (meta['provenance'].get('native_whole_pose_verified') is not True or source_manifest.get('capture_head_support') is not True):
        raise ValueError('Native whole-pose replay and compiled head support required')
    verified_limits(build/'source/profiles',viewer)
    return dict(files=files,build=build,viewer=viewer,config=config)


def ring_ready(path,pid,started):
    try:
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
        try:data=os.pread(fd,1344,0);size=os.fstat(fd).st_size
        finally:os.close(fd)
        if len(data)!=1344:return False
        magic,version,header,count,_,total,owner,heartbeat=struct.unpack_from('<8sIIIIQqQ16x',data)
        now=time.monotonic_ns()
        if magic!=b'FHRING01' or version!=1 or header!=1344 or not 2<=count<=8 or total!=size or owner!=pid or not started<=heartbeat<=now or now-heartbeat>500_000_000:return False
        ready=0
        for i in range(count):
            offset=64+160*i
            width,height=struct.unpack_from('<II',data,offset+72)
            latest=struct.unpack_from('<Q',data,offset+104)[0]
            flags=struct.unpack_from('<I',data,offset+128)[0]
            ready+=latest>0 and flags==0 and (width,height)==(1056,1024)
        return ready>=2
    except (OSError,ValueError,struct.error):return False


def calibration_matches(expected,actual):
    if isinstance(expected,dict):
        return isinstance(actual,dict) and all(k in actual and calibration_matches(v,actual[k]) for k,v in expected.items())
    if isinstance(expected,list):
        if not isinstance(actual,list):return False
        # Named cameras/intrinsic models may be a subset of the runtime config.
        if all(isinstance(x,dict) for x in expected):
            return all(any(calibration_matches(item,other) for other in actual) for item in expected)
        return expected==actual
    return expected==actual


class Children:
    """Each child has a dedicated process group; cleanup never searches by name."""
    def __init__(self,root):self.root=root;self.items=[];self.forced=[]
    def start(self,name,command,env,affinity=None):
        log=(self.root/(name+'.log')).open('xb')
        try:child=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=(lambda:os.sched_setaffinity(0,set(affinity))) if affinity else None)
        except BaseException:log.close();raise
        self.items.append((name,child,log));return child
    def require_alive(self):
        for name,child,_ in self.items:
            if child.poll() is not None:raise RuntimeError(name+' exited early: '+str(child.returncode))
    def stop(self):
        # The subscriber's helper inherits its group, so even an exited parent
        # cannot leave its OpenVR helper alive after this session.
        for name,child,log in reversed(self.items):
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:child.wait(timeout=4)
            except subprocess.TimeoutExpired:
                self.forced.append(name)
                os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=2)
            # If a descendant ignores TERM after its parent exits, remove only
            # this session's remaining group before releasing the workflow lease.
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            log.close()


def wait_for(predicate,children,seconds):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        children.require_alive()
        if predicate():return
        time.sleep(.02)
    raise TimeoutError('Native session startup timed out')


def fresh_snapshot(path,started):
    try:
        value=decode(path.read_bytes());stamp=value['capture_mono_ns'];now=time.monotonic_ns()
        return value.get('version')==3 and started<=stamp<=now and now-stamp<250_000_000
    except (OSError,ValueError,KeyError,TypeError):return False


def fresh_pose(path,started):
    try:
        value=decode(path.read_bytes());stamp=value['capture_ns'];now=time.monotonic_ns()
        return value.get('version')==2 and value.get('context_valid') is True and started<=stamp<=now and now-stamp<250_000_000
    except (OSError,ValueError,KeyError,TypeError):return False


def clean_environment():
    return {k:v for k,v in os.environ.items() if not k.startswith(('FRAME_HANDS_','FRAME_UME_')) and k!='FRAME_JOB_DEVICE_ROOT'}


def commands(spec,output,seconds):
    files=spec['files'];build=spec['build'];viewer=spec['viewer'];env=clean_environment()
    tracker_env=env.copy()
    for flag in ('PAIR_CONTINUITY','OPEN_SIDE','IDENTITY_LATCH','ROTATION_SEARCH','FULL_RATE','FILTER_GUARD','OUTPUT_RECOVERY','JOINT_IDENTITY'):
        tracker_env['FRAME_HANDS_'+flag]='1'
    tracker_env.update(FRAME_HANDS_PALM_THRESHOLD='0.2',FRAME_HANDS_AUDIT=str(output/'recording/live-views.txt'),FRAME_HANDS_RIG_DIR=str(build/'source/profiles'),FRAME_HANDS_RIG_TRACE=str(output/'rig.txt'),FRAME_HANDS_RIG_SNAPSHOT=str(output/'rig.json'),FRAME_JOB_DEVICE_ROOT=str(build/'source/device'),FT_HANDS_SWAP_SIDES='0')
    if spec.get('config',{}).get('whole_pose'):
        tracker_env.update(FRAME_HANDS_RIG_WHOLE_POSE='1',FRAME_HANDS_RIG_OPENVR=str(files['openvr']))
    viewer_env=env.copy();viewer_env.setdefault('DISPLAY',':0')
    viewer_env.update(XR_RUNTIME_JSON=str(files['xr_runtime']),LD_LIBRARY_PATH=str(files['openvr'].parent)+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else ''),FRAME_UME_VIEW=str(output/'subscriber/pose.json'),FRAME_UME_RENDER_TRACE=str(output/'render.jsonl'),FRAME_UME_PRESENTATION='predict',FRAME_UME_SECONDS=str(seconds+3))
    return dict(camera=([str(files['camera']),'--sensor','og01a1b','--status','1'],env),
        tracker=([str(build/'ft-hands-native-rig'),'--no-publish','--models',str(build/'source/models/ncnn'),'--cams','mono','--cpus','5,6,7','--threads','3','--status','1','--seconds',str(seconds+35),'--record',str(output/'recording'),'--record-for',str(seconds+35)],tracker_env),
        subscriber=([sys.executable,str(BASE/'rig_pose_subscriber.py'),'--snapshot',str(output/'rig.json'),'--profiles',str(build/'source/profiles'),'--viewer',str(viewer),'--pose-service',str(files['pose_service']),'--openvr',str(files['openvr']),'--output',str(output/'subscriber'),'--seconds',str(seconds+15)],env),
        viewer=([str(files['godot']),'--path',str(viewer),'--xr-mode','on'],viewer_env))


def require_idle_scene(spec,output):
    result=subprocess.run([str(spec['files']['scene_check'])],capture_output=True,text=True,timeout=8)
    if result.returncode:raise RuntimeError('Cannot inspect XR scene owner')
    owner=decode(result.stdout.encode());atomic_json(output/('scene-owner-'+str(time.monotonic_ns())+'.json'),owner)
    if owner.get('available') is not True or owner.get('scene_pid')!=0:
        raise RuntimeError('Another XR scene is active; leave it unchanged')


def run_session(plan,output,seconds,ring_check,geometry_check,scene_check,recording_check,camera_affinity=None):
    """Policy-free lifecycle core; production caller holds all workflow locks."""
    children=Children(output);reason='failed';result={};started=time.monotonic_ns()
    try:
        scene_check()
        camera_start=time.monotonic_ns();camera=children.start('camera',*plan['camera'],affinity=camera_affinity)
        wait_for(lambda:ring_check(camera.pid,camera_start),children,30)
        geometry_check()
        (output/'recording').mkdir(mode=0o700)
        tracker_start=time.monotonic_ns();children.start('tracker',*plan['tracker'])
        wait_for(lambda:fresh_snapshot(output/'rig.json',tracker_start),children,10)
        children.start('subscriber',*plan['subscriber'])
        wait_for(lambda:fresh_pose(output/'subscriber/pose.json',tracker_start),children,5)
        scene_check()
        viewer=children.start('viewer',*plan['viewer'])
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            if viewer.poll() is not None:
                if viewer.returncode:raise RuntimeError('Viewer failed: '+str(viewer.returncode))
                reason='viewer_closed';break
            children.require_alive();time.sleep(.02)
        else:reason='duration_complete'
    except BaseException as error:
        result['error']=type(error).__name__+': '+str(error)
        raise
    finally:
        handlers={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
        try:
            for sig in handlers:signal.signal(sig,signal.SIG_IGN)
            children.stop()
        finally:
            for sig,handler in handlers.items():signal.signal(sig,handler)
        pose=output/'subscriber/pose.json'
        try:sequence=int(decode(pose.read_bytes()).get('sequence',0))+1
        except (OSError,ValueError,TypeError):sequence=1
        if pose.parent.exists():atomic_json(pose,dict(version=2,sequence=sequence,context_valid=False,hands=[],reason='session_stopped'))
        result.update(reason=reason,started_mono_ns=started,ended_mono_ns=time.monotonic_ns(),forced=children.forced,processes={name:dict(pid=child.pid,exit=child.returncode) for name,child,_ in children.items},input_injected=False)
        try:
            result['recording']=recording_check(output/'recording') if (output/'recording/sets.bin').exists() else dict(complete=False,reason='recording_not_started')
        except (OSError,ValueError) as error:result['recording']=dict(complete=False,reason=str(error))
        result['recording_ok']=result['recording'].get('structurally_complete') is True and result['recording'].get('all_audited_images_saved') is True
        atomic_json(output/'result.json',result)
    if not result['recording_ok']:raise RuntimeError('Session ended but the recording is incomplete; inspect result.json')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=int,default=90)
    parser.add_argument('--ready',action='store_true',help='Wearer freshly confirmed standalone mode, PC streaming disconnected, USB optional, and readiness for camera/VR trial')
    args=parser.parse_args()
    if not args.ready:parser.error('Fresh supervised standalone readiness is required')
    if platform.system()!='Linux' or platform.machine()!='aarch64':parser.error('Native Linux ARM64 only')
    if not 10<=args.seconds<=180:parser.error('seconds must be10..180')
    if not args.output.is_absolute() or args.output.exists():parser.error('New absolute output directory required')
    spec=configuration(args.config)
    with native_workflow():
        if shutil.disk_usage(args.output.parent).free<(args.seconds+40)*140_000_000+2_000_000_000:raise RuntimeError('Insufficient space for full camera recording')
        for name in ('xrservice.json','device_config.json'):
            expected=json.loads((spec['build']/'source/device/persist'/name).read_text())
            actual=json.loads((Path('/persist')/name).read_text())
            if not calibration_matches(expected,actual):raise ValueError('Live factory calibration differs from tested snapshot')
        args.output.mkdir(mode=0o700)
        atomic_json(args.output/'inputs.json',spec['config'])
        snapshot=args.output/'source';snapshot.mkdir()
        for name in SOURCES:
            shutil.copy2(BASE/name,snapshot/name)
        for name in ('camera','geometry','scene_check','pose_service','xr_runtime','recording_tool'):
            shutil.copyfile(spec['files'][name],snapshot/name)
        atomic_json(args.output/'runtime.json',dict(mode='standalone native diagnostic',machine=platform.machine(),kernel=platform.release(),python=sys.version,os_release=Path('/etc/os-release').read_text(),godot=subprocess.run([str(spec['files']['godot']),'--version'],capture_output=True,text=True,timeout=8,check=True).stdout.strip(),reader_cpus=[7],tracker_cpus=[5,6,7]))
        shutil.copytree(spec['build'],snapshot/'tracker-build')
        shutil.copytree(spec['viewer'],snapshot/'viewer',ignore=shutil.ignore_patterns('.godot'))
        plan=commands(spec,args.output,args.seconds)
        atomic_json(args.output/'commands.json',{name:dict(argv=argv,environment={k:v for k,v in env.items() if k.startswith(('FRAME_','FT_','XR_'))}) for name,(argv,env) in plan.items()})
        def geometry():
            report=args.output/'camera-geometry.json'
            checked=subprocess.run([sys.executable,str(spec['files']['geometry']),'--output',str(report)],timeout=15,capture_output=True,text=True)
            (args.output/'geometry.log').write_text(checked.stdout+checked.stderr)
            if checked.returncode or not decode(report.read_bytes()).get('verified'):raise RuntimeError('Camera geometry did not confirm expected ordering')
        module_spec=importlib.util.spec_from_file_location('session_recording_checks',spec['files']['recording_tool'])
        module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
        def interrupted(*_):raise KeyboardInterrupt
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,interrupted)
        def recording_check(directory):
            result=module.inspect_camera(directory)
            result['sets_sha256']=sha(directory/'sets.bin')
            result['views_sha256']=sha(directory/'live-views.txt')
            return result
        ring=Path(f'/run/user/{os.getuid()}/frametop-hands/cam-ring')
        result=run_session(plan,args.output,args.seconds,lambda pid,stamp:ring_ready(ring,pid,stamp),geometry,lambda:require_idle_scene(spec,args.output),recording_check,camera_affinity=(7,))
        print(json.dumps(result,indent=2))
if __name__=='__main__':main()
