"""Bounded, isolated PC VR UmeTrack mesh trial. No SteamVR driver/input writes.

Requires explicit physical readiness (--ready), USB and connected PC VR. The
existing Frame source retains exclusive camera ownership and records raw frames.
"""
import argparse
from collections import deque
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import queue
import shutil
import struct
import subprocess
import sys
import threading
import time
import traceback
import cv2
import numpy as np
from frame_data import read_audit, load_calibration
from live_backend import Backend
from mesh_packet import pose_packet, rig_data, publish
from runtime_metadata import installed_revision

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
REMOTE='/home/steamos/devkit-game/FrameHands'


def read_exact(pipe,n):
    chunks=[]
    while n:
        chunk=pipe.read(n)
        if not chunk:raise EOFError('Exporter ended during response')
        chunks.append(chunk);n-=len(chunk)
    return b''.join(chunks)


def response(pipe):
    line=pipe.readline(24_000_000)
    if not line.endswith(b'\n'):raise EOFError('Incomplete exporter line')
    pipe=io.BytesIO(base64.b64decode(line.strip(),validate=True))
    header=read_exact(pipe,4)
    size,=struct.unpack('<I',header)
    if size>1_000_000:raise ValueError('Invalid exporter JSON size')
    raw=read_exact(pipe,size)
    meta=json.loads(raw)
    lengths=meta.get('lengths',[])
    if len(lengths)>8 or any(not 0<n<8_000_000 for n in lengths):raise ValueError('Invalid PNG lengths')
    blobs=[read_exact(pipe,n) for n in lengths]
    return meta,blobs,header+raw+b''.join(blobs)


def pose_matrix(values):
    xf=np.eye(4)
    xf[:3,:]=np.array(values).reshape(3,4)
    xf[:3,3]*=1000
    if not np.isfinite(xf).all() or not np.allclose(xf[:3,:3].T@xf[:3,:3],np.eye(3),atol=1e-3):
        raise ValueError('Invalid HMD transform')
    return xf


class Pose:
    def __init__(self,exe,dll):
        self.child=subprocess.Popen([str(exe),str(dll)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
        self.responses=queue.Queue()
        def reader():
            try:
                for line in self.child.stdout:self.responses.put(json.loads(line))
            except Exception as e:self.responses.put(e)
            finally:self.responses.put(EOFError('Pose helper exited'))
        threading.Thread(target=reader,daemon=True).start()
    def get(self,age):
        self.child.stdin.write(f'{max(0,min(age,4999)):.3f}\n'.encode());self.child.stdin.flush()
        value=self.responses.get(timeout=3)
        if isinstance(value,Exception):raise value
        return value
    def close(self):
        self.child.stdin.close()
        try:self.child.wait(timeout=4)
        except subprocess.TimeoutExpired:self.child.terminate();self.child.wait(timeout=4)


class NativeSink:
    def __init__(self, command, env, log):
        self.child=subprocess.Popen(command+['shell','-T','python3 '+REMOTE+'/experiments/piano-native-link-v1/native_piano_link.py'],env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log)
        self.responses=queue.Queue()
        def reader():
            try:
                for line in self.child.stdout:self.responses.put(json.loads(line))
            except Exception as e:self.responses.put(e)
            finally:self.responses.put(EOFError('Native Piano hand receiver stopped'))
        threading.Thread(target=reader,daemon=True).start()
    def send(self, packet):
        self.child.stdin.write((json.dumps(packet,allow_nan=False,separators=(',',':'))+'\n').encode());self.child.stdin.flush()
        result=self.responses.get(timeout=5)
        if isinstance(result,Exception):raise result
        return result
    def close(self):
        try:
            if self.child.poll() is None:self.child.stdin.write(b'quit\n');self.child.stdin.flush()
            self.child.stdin.close();self.child.wait(timeout=6)
        except (OSError,subprocess.TimeoutExpired):self.child.terminate();self.child.wait(timeout=3)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True,help='Original checkout containing tools; never modified')
    p.add_argument('--seconds',type=int,default=90)
    p.add_argument('--presentation',choices=('raw','smooth','predict'),default='predict')
    p.add_argument('--continuity',choices=('accepted','native','bridge'),default='native',help='Post-inference presentation; native adds bounded correction and anatomical joint limits')
    p.add_argument('--roi-source',choices=('buffered','flushed'),default='flushed')
    p.add_argument('--camera-reader',choices=('direct-v2','exact-v3'),default='direct-v2')
    p.add_argument('--identity',choices=('original','image-cuda'),default='original',help='Explicit experimental image-based identity; CUDA environment required')
    p.add_argument('--steamvr',type=Path,help='Prepared model-driver deployment; skeletons for PC VR games')
    p.add_argument('--native-piano',action='store_true',help='Feed PC estimates to the installed native Frame app; no PC scene')
    p.add_argument('--piano',type=Path,help='Verified Windows Piano export to consume the local hand feed instead of the mesh scene')
    p.add_argument('--stop-file',type=Path,help='Owned cancellation file created by the manual panel')
    p.add_argument('--ready',action='store_true',help='Wearer has confirmed PC VR visibility and readiness')
    p.add_argument('--probe-export',help='Existing recording ID; transport-only, no camera or XR scene')
    p.add_argument('--continuous',action='store_true',help='Run Piano hands until Stop or app exit; bounded live image spool, no full recording')
    a=p.parse_args()
    if a.continuous and (not (a.piano or a.native_piano or a.steamvr) or not a.stop_file or a.probe_export):p.error('Continuous mode requires Piano and an owned stop file')
    if a.steamvr and (a.piano or a.native_piano or not a.continuous or a.continuity!='bridge' or a.identity!='image-cuda' or a.camera_reader!='exact-v3'):p.error('SteamVR requires continuous image-cuda/exact-v3/bridge without Piano')
    if a.native_piano and (a.piano or not a.continuous or a.continuity!='bridge' or a.identity!='image-cuda' or a.camera_reader!='exact-v3'):p.error('Native Piano requires continuous image-cuda/exact-v3/bridge, without a PC Piano executable')
    if not 10<=a.seconds<=300:p.error('seconds must be 10..300')
    if a.piano and (not a.piano.is_file() or not a.piano.with_suffix('.pck').is_file()):p.error('Piano executable and matching pck required')
    if a.piano and a.continuity!='bridge':p.error('Piano integration requires the bridge presentation')
    if not a.ready and not a.probe_export:p.error('Physical readiness is required before a camera trial')
    if a.camera_reader=='exact-v3' and a.roi_source!='flushed':p.error('exact-v3 requires the isolated flushed source')
    if a.identity!='original' and a.probe_export:p.error('Transport probe cannot run identity inference')
    shared=a.workspace.resolve()
    adb=shared/'.local/android-tools/platform-tools/adb.exe'
    upstream=ROOT/'.local/hand-alternatives/UmeTrack'
    pose_exe=ROOT/'.local/hand-alternatives/pose-service.exe'
    dll=Path('C:/Program Files (x86)/Steam/steamapps/common/SteamVR/bin/win64/openvr_api.dll')
    godot=shared/'.local/deps/godot-4.7.2/Godot_v4.7.2-stable_win64.exe'
    env=dict(os.environ,ANDROID_USER_HOME=str(Path.home()/'.android'))
    command=[str(adb),'-s','frame']
    def remote(script,**kwargs):
        return subprocess.run(command+['shell',script],env=env,check=True,timeout=15,**kwargs)
    stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
    directory=ROOT/'.local/hand-alternatives/results'/('transport-'+stamp if a.probe_export else 'live-mesh-'+stamp)
    directory.mkdir(parents=True,exist_ok=False)
    snapshot=directory/'snapshot';snapshot.mkdir()
    for f in BASE.glob('*.py'):shutil.copy2(f,snapshot/f.name)
    shutil.copytree(BASE/'mesh-viewer',snapshot/'mesh-viewer',ignore=shutil.ignore_patterns('.godot','*.uid'))
    source=exporter=viewer=pose=sink=None
    pipeline=None
    logs=[]
    report=dict(version=1,session_id=stamp,mode='PC streamed isolated mesh diagnostic',driver_modified=False,
                source_seconds=a.seconds,roi_source=a.roi_source,camera_reader=a.camera_reader,presentation=a.presentation,continuity=a.continuity,identity=a.identity,errors=[])
    if a.steamvr:
        from steamvr_sink import validate_deployment
        report.update(mode='PC CUDA fixed-rig hands to SteamVR',continuous=True,full_recording=False,source_seconds=None,driver=validate_deployment(a.steamvr))
    if a.native_piano:
        report.update(mode='PC CUDA inference to native Frame Piano',continuous=True,full_recording=False,source_seconds=None,consumer={'kind':'installed native Frame Piano','app_changed':False})
    if a.piano:
        report['continuous']=a.continuous
        report['full_recording']=not a.continuous
        report['consumer']={'kind':'Frame Piano','files':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (a.piano,a.piano.with_suffix('.pck'))}}
    sys.path.insert(0,str(upstream))
    import torch
    report['model']=dict(upstream_commit=installed_revision(upstream),weights_sha256=hashlib.sha256((upstream/'pretrained_models/pretrained_weights.torch').read_bytes()).hexdigest(),mesh_sha256=hashlib.sha256((upstream/'dataset/generic_hand_model.json').read_bytes()).hexdigest(),torch=torch.__version__,numpy=np.__version__,opencv=cv2.__version__,python=sys.version,threads=4)
    def log(name):
        f=(directory/name).open('wb');logs.append(f);return f
    try:
        if a.identity=='image-cuda':
            from identity_pipeline import ImageIdentityPipeline
            pipeline=ImageIdentityPipeline(upstream)
            report['candidate']=dict(preparation_seconds=pipeline.preparation_seconds,device='cuda',cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),tf32=False)
            print('Image identity models warmed up; checking the connected headset.',flush=True)
        remote('test "$(systemctl is-active frame-eyes-reader-brightness.service)" != active')
        if a.stop_file and a.stop_file.exists():raise RuntimeError('Stopped before capture startup')
        if a.steamvr:
            from ready_panel import preflight
            preflight(shared,steamvr=a.steamvr)
        if a.native_piano:
            from ready_panel import preflight
            preflight(shared,native=True)
            installed=remote('cat /home/steamos/devkit-game/FramePiano/frame-piano-install.json',capture_output=True,text=True)
            report['native_piano_install']=json.loads(installed.stdout)
        if not a.probe_export and not a.native_piano:
            pose=Pose(pose_exe,dll)
            if not pose.get(0)['valid']:raise RuntimeError('No valid connected PC VR head pose; no camera started')
        stage=REMOTE+'/experiments/ume-mesh-v1'
        remote('mkdir -p '+stage+'/lib/common')
        for name in ('camera.py','affine.py'):
            local=upstream/'lib/common'/name
            if name=='affine.py':
                text=local.read_text()
                if text.count('from scipy.spatial.transform import Rotation')!=1:raise ValueError('Unexpected affine source')
                text=text.replace('from scipy.spatial.transform import Rotation\n','').replace('    z_local_rot = Rotation.', '    from scipy.spatial.transform import Rotation\n    z_local_rot = Rotation.')
                local=snapshot/'affine.py';local.write_text(text)
            subprocess.run(command+['push',str(local),stage+'/lib/common/'+name],env=env,check=True,timeout=15,stdout=subprocess.DEVNULL)
        subprocess.run(command+['push',str(upstream/'LICENSE'),stage+'/UME-LICENSE'],env=env,check=True,timeout=15,stdout=subprocess.DEVNULL)
        for name in ('export_live.py','frame_data.py','continuous_session.py','native_piano_link.py'):
            subprocess.run(command+['push',str(BASE/name),stage+'/'+name],env=env,check=True,timeout=15,stdout=subprocess.DEVNULL)
        session=a.probe_export or stamp
        if not a.probe_export:
            # Exclusive locks, geometry check, exact queue reader and full recording
            # are owned by the existing bounded source. No relay or driver mailbox.
            runner=REMOTE+'/experiments/ume-mesh-source-v1/launchers/pc-input/run_source.py' if a.roi_source=='flushed' else REMOTE+'/pc-input/run_source.py'
            if a.camera_reader=='exact-v3':
                runner=REMOTE+'/experiments/camera-exact-v3/launchers/pc-input/run_source.py'
            if a.roi_source=='flushed':
                remote('test -f '+runner)
                subprocess.run(command+['pull',REMOTE+'/experiments/ume-mesh-source-v1/trial-build.json',str(snapshot/'roi-source-build.json')],env=env,check=True,timeout=15,stdout=subprocess.DEVNULL)
            if a.continuous:
                live_launch=REMOTE+'/experiments/piano-live-source/'+stamp
                remote(f'python3 {stage}/continuous_session.py {REMOTE}/experiments/camera-exact-v3/launchers {live_launch}')
                runner=live_launch+'/pc-input/run_source.py'
            source=subprocess.Popen(command+['shell',f'python3 {runner} --seconds {a.seconds} --filter-guard --output-recovery --joint-identity --camera-direct --record-session --session-id {stamp}'+(' --continuous' if a.continuous else '')],
                env=env,stdin=subprocess.PIPE,stdout=log('source.log'),stderr=subprocess.STDOUT)
        exporter=subprocess.Popen(command+['shell','-T',f'PYTHONPATH={REMOTE}/deps/python python3 {stage}/export_live.py {REMOTE}/private/{session} --seconds {a.seconds+90}'+(' --repeat' if a.probe_export else '')+(' --continuous' if a.continuous else '')+(' --native-head' if a.native_piano else '')],
            env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log('exporter.log'))
        incoming=queue.Queue(maxsize=1)
        def receive():
            try:
                while exporter.poll() is None:incoming.put(response(exporter.stdout))
            except Exception as e:incoming.put(e)
        threading.Thread(target=receive,daemon=True).start()
        age_clock=time.perf_counter if a.steamvr else time.monotonic
        backend=None
        feed_ready=False;last_sink_send=0.0;last_source_frame=time.monotonic()
        start=time.monotonic()
        delivered_ages=deque(maxlen=3600);delivery_rejections=0
        seq=0;ages=deque(maxlen=3600 if a.continuous else None);latencies=deque(maxlen=3600 if a.continuous else None);infer=deque(maxlen=3600 if a.continuous else None);slots=[0,0];probe_times=[]
        def diagnostic(name, mode):
            return open(os.devnull, 'wb' if 'b' in mode else 'w') if a.continuous else (directory/name).open(mode)
        with diagnostic('transport.bin','xb') as wire,diagnostic('predictions.jsonl','x') as predictions,diagnostic('timing.jsonl','x') as timing,diagnostic('identity.jsonl','x') as identity:
            while a.continuous or time.monotonic()-start < a.seconds+75:
                if a.stop_file and a.stop_file.exists():
                    report['requested_stop']=True;break
                if source and source.poll() is not None:break
                if viewer and viewer.poll() is not None:
                    if a.continuous:report['requested_stop']=True
                    break
                if a.continuous:
                    source.stdin.write(b'H');source.stdin.flush()
                t0=age_clock()
                exporter.stdin.write(b'next\n');exporter.stdin.flush()
                item=incoming.get(timeout=8)
                t1=age_clock()
                if isinstance(item,Exception):raise item
                meta,blobs,raw=item
                # Every wire response, including waiting/clock samples, is retained.
                wire.write(raw)
                rtt=(t1-t0)*1000
                if meta.get('waiting'):
                    if a.steamvr and feed_ready and time.monotonic()-last_source_frame>1:
                        print('SteamVR model hands waiting for camera frames. Keep Frame awake.',flush=True);feed_ready=False
                    if a.native_piano and sink and time.monotonic()-last_source_frame>1 and time.monotonic()-last_sink_send>1:
                        sink.send({'kind':'heartbeat'});last_sink_send=time.monotonic()
                        if feed_ready:
                            print('Native hand feed waiting for camera frames. Keep Frame awake.',flush=True)
                            feed_ready=False
                    time.sleep(.02);continue
                last_source_frame=time.monotonic()
                if a.probe_export:
                    report.update(png_bytes=list(map(len,blobs)),probe_roundtrip_ms=rtt,decoded_shapes=[cv2.imdecode(np.frombuffer(b,np.uint8),0).shape for b in blobs])
                    probe_times.append(rtt)
                    if len(probe_times)<31:continue
                    report['probe_steady_ms']=np.percentile(probe_times[1:],[50,95,100]).tolist()
                    break
                if backend is None:
                    subprocess.run(command+['pull',f'{REMOTE}/private/{stamp}/snapshot/device/persist',str(directory/'calibration')],env=env,check=True,timeout=30,stdout=subprocess.DEVNULL)
                    cameras=load_calibration(directory/'calibration')
                    if pipeline:
                        pipeline.bind_calibration(cameras)
                        backend=pipeline.backend
                    else:backend=Backend(upstream,cameras)
                    (directory/'rig.json').write_text(json.dumps(rig_data(backend.hand)))
                    if a.native_piano:
                        sink=NativeSink(command,env,log('native-receiver.log'))
                        print('Native receiver connected. Waiting for an accepted hand pose.',flush=True)
                    elif a.steamvr:
                        from steamvr_sink import SteamSink
                        sink=SteamSink(godot,directory,a.steamvr,log('steam-processor.log'))
                        print('SteamVR hand processor started; waiting for tracked hands.',flush=True)
                    else:
                        view_env=dict(env,FRAME_UME_RIG=str(directory/'rig.json'),FRAME_UME_PRESENTATION=a.presentation,FRAME_UME_CONTINUITY=a.continuity,FRAME_UME_VIEW=str(directory/'view.json'),FRAME_UME_SECONDS=str(a.seconds+10),FRAME_UME_RENDER_TRACE=str(directory/'render.jsonl'))
                        if a.continuous:
                            view_env.pop('FRAME_UME_RENDER_TRACE',None)
                            view_env['FRAME_UME_SECONDS']='0'
                        viewer_command=[str(a.piano),'--xr-mode','on'] if a.piano else [str(godot),'--path',str(BASE/'mesh-viewer'),'--xr-mode','on']
                        viewer=subprocess.Popen(viewer_command,env=view_env,stdout=log('viewer.log'),stderr=subprocess.STDOUT)
                        print('Piano opened with optical hands. Map the piano before practice.' if a.piano else 'Mesh scene opened. Raise hands, turn palms, curl fingers, then cross and separate slowly.',flush=True)
                    # Initialization made this frame stale; request the newest one.
                    continue
                observation=next(iter(read_audit(io.StringIO(meta['audit'])).values()))
                # Conservative age: source reports time immediately before sending.
                # Full RTT is an upper bound on response travel time, not half RTT
                # (encoding is asymmetric). Never treat encode time as network delay.
                age=(meta['source_now_ns']-observation['mono_ns'])/1e6+rtt
                if age<0 or age>250:
                    if pipeline:
                        pipeline.reset('Stale transport frame')
                        identity.write(json.dumps(dict(stamp=meta['stamp'],**pipeline.last_debug))+'\n')
                    else:backend.previous={}
                    timing.write(json.dumps(dict(stamp=meta['stamp'],dropped=True,age_ms=age,roundtrip_ms=rtt))+'\n')
                    continue
                prepared={(v['camera'],v['id']):(cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_GRAYSCALE),v) for v,b in zip(meta['crops'],blobs)}
                if any(image is None or image.shape!=(96,96) for image,_ in prepared.values()):raise ValueError('Invalid decoded crop')
                pose_before=meta.get('head',{'valid':False}) if a.native_piano else pose.get(age+(age_clock()-t1)*1000)
                if a.native_piano and pose_before.get('capture_mono_ns')!=observation['mono_ns']:pose_before={'valid':False}
                capture=pose_matrix(pose_before['capture']) if pose_before['valid'] else None
                inferred=pipeline.predict(observation,capture,prepared) if pipeline else backend.predict(observation,{},capture,prepared)
                if pipeline:identity.write(json.dumps(dict(stamp=meta['stamp'],**pipeline.last_debug),allow_nan=False)+'\n')
                current=pose_before if a.native_piano else pose.get(0)
                seq+=1
                age+=(age_clock()-t1)*1000
                if not current['valid']:inferred['hands']=[]
                head=pose_matrix(current['current']) if current['valid'] else np.eye(4)
                value=pose_packet(inferred,head,seq,observation['mono_ns'],age,current['valid'] and pose_before['valid'])
                if a.native_piano:
                    accepted=sink.send(value);last_sink_send=time.monotonic()
                    if accepted['accepted'] and value['hands'] and not feed_ready:
                        print('Native hand feed ready. Open Devkit Game: FramePiano on Frame; PC VR streaming stays disconnected.',flush=True)
                        feed_ready=True
                    report['last_receiver_status']=accepted
                    if accepted['accepted']:delivered_ages.append(accepted['age_ms'])
                    else:delivery_rejections+=1
                elif a.steamvr:
                    sink.send(value)
                    if sink.valid_hands and not feed_ready:
                        print('SteamVR model hands ready. Open Maestro or another skeletal-input game.',flush=True);feed_ready=True
                else:publish(directory/'view.json',value)
                predictions.write(json.dumps(dict(sequence=seq,capture_raw_ns=meta['stamp'],capture_mono_ns=observation['mono_ns'],capture_pose=pose_before,current_pose=current,**inferred),allow_nan=False)+'\n')
                timing.write(json.dumps(dict(sequence=seq,stamp=meta['stamp'],age_upper_ms=age,roundtrip_ms=rtt,processing_ms=inferred['processing_ms']))+'\n')
                ages.append(age);latencies.append(rtt);infer.append(inferred['processing_ms'])
                for h in inferred['hands']:slots[h['side']]+=1
                if seq%150==0:print(f'{seq} frames; {slots} hand outputs; age upper median {np.median(ages):.0f} ms',flush=True)
        if a.steamvr and sink:report.update(driver_publications=sink.count,driver_hand_samples=sink.valid_hands,driver_clock_rejections=sink.rejected,driver_write_drops=sink.write_drops,driver_publication_max_gap_ms=sink.max_gap_ms)
        if a.native_piano:report.update(receiver_age_ms=np.percentile(delivered_ages,[50,95,100]).tolist() if delivered_ages else None,delivery_rejections=delivery_rejections)
        report.update(frames=seq,hand_outputs=slots,age_upper_ms=np.percentile(ages,[50,95,100]).tolist() if ages else None,
                      roundtrip_ms=np.percentile(latencies,[50,95,100]).tolist() if latencies else None,
                      inference_ms=np.percentile(infer,[50,95,100]).tolist() if infer else None)
    except Exception as exc:
        report['errors'].append(type(exc).__name__+': '+str(exc));traceback.print_exc()
    finally:
        if sink:
            try:sink.close()
            except Exception as e:report['errors'].append('Native receiver cleanup: '+str(e))
        for name,child in [('source',source),('exporter',exporter)]:
            if child:
                try:

                    if child.poll() is None:
                        child.stdin.write(b'quit\n' if name=='exporter' else b'Q');child.stdin.flush()
                    child.stdin.close();report[name+'_exit']=child.wait(timeout=20)
                except (OSError,subprocess.TimeoutExpired) as e:
                    report['errors'].append(name+' cleanup: '+str(e))
                    # Kill only the PC transport on failure. Frame source has its
                    # own EOF/timeout cleanup; never kill unrelated camera workers.
                    child.terminate()
        if viewer:
            report['viewer_exit_before_cleanup']=viewer.poll()
            if viewer.poll() not in (None,0):report['errors'].append('Viewer exited with '+str(viewer.returncode))
            if viewer.poll() is None:viewer.terminate();viewer.wait(timeout=5)
        for name in ('source','exporter'):
            if report.get(name+'_exit',0)!=0:report['errors'].append(name+' exited with '+str(report[name+'_exit']))
        if pose:pose.close()
        for f in logs:f.close()
        (directory/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    if source and not a.continuous:
        sys.path.insert(0,str(BASE.parent/'pc-input'))
        from session_recording import collect
        from recording import inspect_camera, digest
        try:
            collect(shared/'apps/frame-hands/pc-input',directory,stamp)
            report['camera']=inspect_camera(directory/'frame')
            report['frame_result']=json.loads((directory/'frame/result.json').read_text())
            if not report['camera']['all_audited_images_saved']:raise ValueError('Missing audited camera images')
            if (report['frame_result'].get('error') and not report.get('requested_stop')) or not report['frame_result'].get('owned_processes_stopped'):raise ValueError('Source did not finish cleanly')
            for name,entry in json.loads((directory/'frame/snapshot/files.json').read_text()).items():
                if digest(directory/'frame/snapshot'/name)!=entry['sha256']:raise ValueError('Source snapshot hash mismatch: '+name)
        except Exception as e:report['errors'].append('Recording collection: '+str(e))
        (directory/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    if source and a.continuous:
        try:
            subprocess.run(command+['pull',f'{REMOTE}/private/{stamp}/result.json',str(directory/'frame-result.json')],env=env,check=True,timeout=15,stdout=subprocess.DEVNULL)
            frame_result=json.loads((directory/'frame-result.json').read_text())
            report['frame_result']=frame_result
            if not frame_result.get('owned_processes_stopped'):raise RuntimeError('Source cleanup not confirmed')
            if frame_result.get('error'):raise RuntimeError(frame_result['error'])
            # Only transient live spool files belonging to this exact session. No saved trials are removed.
            remote(f'rm -f {REMOTE}/private/{stamp}/sets.bin {REMOTE}/private/{stamp}/live-views.txt')
        except Exception as e:report['errors'].append('Live session cleanup: '+str(e))
        (directory/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    hashes={str(f.relative_to(snapshot)):hashlib.sha256(f.read_bytes()).hexdigest() for f in snapshot.rglob('*') if f.is_file()}
    (directory/'snapshot-hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    print('Evidence: '+str(directory),flush=True)
    return int(bool(report['errors']))

if __name__=='__main__':raise SystemExit(main())
