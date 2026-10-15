"""Opt-in Frame-native service. No scene, video or per-frame trace."""
import argparse, hashlib, json, os, platform, signal, subprocess, sys, tempfile, threading, time, socket
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'native'))
from run_native_trial import configuration, commands, Children, wait_for, ring_ready, calibration_matches
from rig_pose_subscriber import Subscriber, PoseProcess, verified_limits, atomic_json, decode, MAX_BYTES
from workflow_guard import native_workflow, process_names
from wire import pack, request_mode
RUNTIME=Path('/run/user')/str(os.getuid())/'frame-hands-native'

def notify(value):
    target=os.environ.get('NOTIFY_SOCKET')
    if not target:return
    if target.startswith('@'):target='\0'+target[1:]
    with socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM) as sock:sock.sendto(value.encode(),target)

def require_camera_available():
    if any(n.startswith('frame-eyes-read') for _,n in process_names()):
        raise RuntimeError('Eye camera workflow is active')

def require_camera_permissions(reader):
    result=subprocess.run(['getcap',str(reader)],capture_output=True,text=True,check=True)
    fields=result.stdout.strip().split()
    granted=set()
    if len(fields)==2 and fields[1].endswith('=ep'):
        granted=set(fields[1][:-3].split(','))
    if not {'cap_sys_ptrace','cap_perfmon','cap_dac_read_search'} <= granted:
        raise RuntimeError('Camera reader permissions are missing. Run the Frame camera-access repair with local authentication, then retry.')

def require_loaded(binary):
    found=False
    for pid,name in process_names():
        if name!='vrserver':continue
        if str(binary.resolve()) in Path(f'/proc/{pid}/maps').read_text():found=True
    if not found:raise RuntimeError('Restart Frame to load its new hand driver, then start native hands')

def require_awake(spec,output):
    # Background query only: no cameras, scene or app switch.
    with (output/'readiness.log').open('wb') as log:
        probe=PoseProcess([str(BASE/'pose-service'),str(spec['files']['openvr']),'continuous'],log)
        try:
            head=probe.query(time.monotonic_ns()-10000000)
            if head.get('valid') is not True:raise RuntimeError('Put Frame on and wake head tracking, then start native hands')
        finally:probe.close()

def service_plan(spec,output):
    original=commands(spec,output,0)
    argv,env=original['tracker'];argv=list(argv);env=dict(env)
    for flag in ('--record','--record-for'):
        index=argv.index(flag);del argv[index:index+2]
    argv[argv.index('--seconds')+1]='0'
    argv[argv.index('--status')+1]='5'
    env.pop('FRAME_HANDS_AUDIT',None)
    env['FRAME_HANDS_RIG_TRACE']='/dev/null'
    argv[0]=str(BASE/'tracker-no-record')
    return dict(camera=original['camera'],tracker=(argv,env))

class Renderer:
    def __init__(self,spec,output):
        self.error=None;self.count=0;self.hands=0
        self.dest=RUNTIME/'input.bin';self.temp=RUNTIME/'input.tmp'
        self.log=(output/'processor.log').open('wb')
        env=dict(os.environ,FRAME_UME_VIEW=str(output/'pose.json'))
        self.process=subprocess.Popen([str(spec['files']['godot']),'--headless','--xr-mode','off','--path',str(BASE/'processor'),'--script','res://native_bridge.gd'],env=env,stdout=subprocess.PIPE,stderr=self.log,start_new_session=True)
        self.thread=threading.Thread(target=self.pump,daemon=True);self.thread.start()
    def pump(self):
        sequence=time.monotonic_ns()
        try:
            for line in self.process.stdout:
                if not line.startswith(b'FHMODEL '):self.log.write(line);self.log.flush();continue
                sequence+=1
                value=json.loads(line[8:]);data=pack(value,sequence,time.monotonic_ns()//1000000,time.time())
                if data is None:continue
                self.temp.write_bytes(data);os.replace(self.temp,self.dest)
                self.count+=1;self.hands+=len(value['hands'])
        except Exception as error:self.error=error
    def close(self):
        if self.process.poll() is None:self.process.terminate()
        try:self.process.wait(timeout=4)
        except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=2)
        self.thread.join(timeout=2);self.log.close()
        self.dest.unlink(missing_ok=True);self.temp.unlink(missing_ok=True)

def run(spec):
    RUNTIME.mkdir(mode=0o700,exist_ok=True)
    # Journald records errors; all mutable pose/camera data lives in runtime memory.
    with native_workflow(), tempfile.TemporaryDirectory(prefix='session-',dir=RUNTIME) as directory:
        require_camera_available()
        for name in ('device_config.json','xrservice.json'):
            expected=json.loads((spec['build']/'source/device/persist'/name).read_text())
            if not calibration_matches(expected,json.loads((Path('/persist')/name).read_text())):raise RuntimeError('Factory camera calibration changed')
        (RUNTIME/'input.bin').unlink(missing_ok=True)
        output=Path(directory);children=Children(output);renderer=None;helper=None
        def stop(*_):raise KeyboardInterrupt
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,stop)
        try:
            plan=service_plan(spec,output)
            require_camera_permissions(Path(plan['camera'][0][0]))
            require_awake(spec,output)
            started=time.monotonic_ns();camera=children.start('camera',*plan['camera'],affinity=(7,))
            ring=Path(f'/run/user/{os.getuid()}/frametop-hands/cam-ring')
            wait_for(lambda:ring_ready(ring,camera.pid,started),children,30)
            report=output/'geometry.json'
            geometry_deadline=time.monotonic()+15
            while True:
                children.require_alive()
                result=subprocess.run([sys.executable,str(spec['files']['geometry']),'--output',str(report)],capture_output=True,text=True,timeout=12)
                geometry=decode(report.read_bytes())
                if result.returncode==0 and geometry.get('verified'):break
                if result.returncode==3:raise RuntimeError('Camera order is reversed; tracking stopped')
                atomic_json(RUNTIME/'status.json',dict(state='starting',message='Look at a well-lit textured room with hands lowered',camera_reason=geometry.get('reason')))
                if time.monotonic()>=geometry_deadline:raise RuntimeError('Camera alignment check failed: '+str(geometry.get('reason'))+'. Look at a well-lit room with hands lowered, then retry.')
                time.sleep(.2)
            children.start('tracker',*plan['tracker'])
            limits=verified_limits(spec['build']/'source/profiles',spec['viewer'])
            subscriber=Subscriber(limits)
            with (output/'pose-helper.log').open('wb') as log:
                helper=PoseProcess([str(BASE/'pose-service'),str(spec['files']['openvr']),'continuous'],log)
                atomic_json(output/'pose.json',subscriber.invalid('starting',time.monotonic_ns(),time.time()))
                renderer=Renderer(spec,output);request_mode(RUNTIME,'hands')
                last_status=0;ready=False;notified=False
                while True:
                    children.require_alive()
                    if renderer.error:raise renderer.error
                    if renderer.process.poll() is not None:raise RuntimeError('Native pose processor stopped')
                    if helper.process.poll() is not None:raise RuntimeError('Native head-pose helper stopped')
                    try:
                        with (output/'rig.json').open('rb') as stream:data=stream.read(MAX_BYTES+1)
                    except FileNotFoundError:data=None
                    if data is not None:
                        packet,_=subscriber.consume(data,helper.query)
                        if packet is not None:atomic_json(output/'pose.json',packet)
                    expired=subscriber.expire(time.monotonic_ns(),time.time())
                    if expired is not None:atomic_json(output/'pose.json',expired)
                    if time.monotonic()-last_status>2:
                        require_camera_available();last_status=time.monotonic()
                        atomic_json(RUNTIME/'status.json',dict(state='tracking' if renderer.hands else 'waiting_for_hands',published=renderer.count,hand_outputs=renderer.hands,updated_mono_ns=time.monotonic_ns(),recording=False))
                    if renderer.count and not notified:notify('READY=1\nSTATUS=Native tracker running');notified=True
                    if renderer.hands and not ready:print('Native hands ready',flush=True);ready=True
                    time.sleep(.008)
        finally:
            failure=sys.exc_info()[1]
            if failure is not None and not isinstance(failure,KeyboardInterrupt):
                for name in ('camera','tracker','processor'):
                    log_path=output/(name+'.log')
                    if log_path.exists():
                        with log_path.open('rb') as stream:
                            stream.seek(0,2);stream.seek(max(0,stream.tell()-1800))
                            print(name+' startup diagnostic: '+stream.read().decode(errors='replace'),flush=True)
            for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,signal.SIG_IGN)
            if renderer:renderer.close()
            if helper:helper.close()
            children.stop();(RUNTIME/'input.bin').unlink(missing_ok=True)
            atomic_json(RUNTIME/'status.json',dict(state='stopped',updated_mono_ns=time.monotonic_ns(),recording=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args()
    if platform.machine()!='aarch64' or platform.system()!='Linux':raise RuntimeError('Native Frame ARM64 required')
    RUNTIME.mkdir(mode=0o700,exist_ok=True)
    try:
        # All staged code/assets and the accepted source configuration are pinned.
        manifest=json.loads((BASE/'manifest.json').read_text())
        for name,digest in manifest.items():
            if hashlib.sha256((BASE/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Changed package file: '+name)
        spec=configuration(BASE/'accepted-config.json')
        if a.check:print('Native service assets verified; no capture started');return
        deployment=json.loads((BASE/'deployment.json').read_text());require_loaded(Path(deployment['binary']))
        run(spec)
    except KeyboardInterrupt:pass
    except Exception as error:
        if RUNTIME.exists():atomic_json(RUNTIME/'status.json',dict(state='error',error=str(error),recording=False))
        raise
if __name__=='__main__':main()
