"""Bounded Frame-to-PC landmark transport. USB adb is required for this first implementation."""
import argparse
import base64
import ctypes
import json
import math
import os
from pathlib import Path
import queue
import struct
import subprocess
import threading
import time
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from recording import RELAY, RELAY_SIZE

RAW_SIZE=2656
HEADER=struct.Struct('<8sIIQQQII')
HAND=struct.Struct('<II63f')


def publish_atomic(tmp, dest, packet):
    """A Windows reader can leave a replaced file delete-pending briefly."""
    tmp.write_bytes(packet)
    deadline=time.monotonic()+.05
    while True:
        try:
            os.replace(tmp,dest)
            return True
        except PermissionError:
            if time.monotonic()>=deadline:return False
            time.sleep(.002)


def convert(data, started_ms, received_ms, sequence, controls=False):
    if len(data)!=16+RAW_SIZE or data[:8]!=b'FHREAD1\0':
        raise ValueError('invalid export response')
    source_now=struct.unpack_from('<Q',data,8)[0]
    raw=data[16:]
    magic,version,size,seq,capture,publish,count,capsules=struct.unpack_from('<8sIIQQQII',raw)
    empty=HAND.pack(0,0,*([0.]*63))
    hands=[empty,empty]
    rtt=received_ms-started_ms
    valid=(magic==b'FHHANDS1' and version==1 and size==RAW_SIZE and seq>0 and seq%2==0 and
           count<=2 and capsules<=64 and capture<=publish<=source_now and source_now-capture<=100_000_000 and 0<=rtt<=100)
    age=(source_now-capture)/1e6 if valid else 0
    valid=valid and age+rtt<=100
    capture_ms=int((started_ms+received_ms)/2-age) if valid else received_ms
    seen=set();rejected_hands=0;slots=[[],[]]
    if valid:
        for i in range(count):
            ident,flags,confidence,_=struct.unpack_from('<IIff',raw,64+272*i)
            points=struct.unpack_from('<63f',raw,80+272*i);side=flags&1
            # Source "confidence" is min(1, track_frames/5), not probability.
            # A new track must not blink out its established opposite hand.
            if not ident or flags&~3 or not math.isfinite(confidence) or not 0<=confidence<=1 or any(not math.isfinite(v) or abs(v)>2.5 for v in points):
                rejected_hands+=1;continue
            slots[side].append((ident,HAND.pack(ident,1|(2 if flags&2 else 0),*points)))
        for side in range(2):
            if len(slots[side])==1:
                seen.add(side);hands[side]=slots[side][0][1]
            elif slots[side]:rejected_hands+=len(slots[side])
        if len(seen)==2 and slots[0][0][0]==slots[1][0][0]:
            hands=[empty,empty];seen.clear();rejected_hands+=2
    if not valid:hands=[empty,empty]
    return HEADER.pack(b'FHPCVR1\0',1,568,sequence,received_ms,capture_ms,int(controls),0)+b''.join(hands),dict(valid=valid,source_sequence=seq,source_capture=capture,rtt_ms=rtt,age_ms=age,hands=len(seen) if valid else 0,rejected_hands=rejected_hands)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parent
    parser.add_argument('--adb',type=Path,default=root.parents[2]/'.local/android-tools/platform-tools/adb.exe')
    parser.add_argument('--driver',type=Path,required=True,help='Exact registered fh_skeletal bundle')
    parser.add_argument('--seconds',type=int,default=90)
    parser.add_argument('--controls',action='store_true',help='Opt-in pinch/fist/button fallback; native skeleton mode is default')
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--stop-file',type=Path,help='Optional local stop request from run_session.py')
    parser.add_argument('--record',type=Path,help='Private raw transport and clock mapping trace')
    args=parser.parse_args()
    if os.name!='nt' or not 5<=args.seconds<=3600:parser.error('Windows PC and 5..3600 seconds required')
    dest=args.driver/'bin/win64/input.bin'
    if not (dest.parent/'driver_fh_skeletal.dll').is_file():parser.error('driver bundle not found')
    lock=dest.with_suffix('.lock')
    # A second relay must not overwrite a live session. A crashed relay's file can be removed manually.
    lock_fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(lock_fd,str(os.getpid()).encode());os.close(lock_fd)
    tick=ctypes.windll.kernel32.GetTickCount64;tick.restype=ctypes.c_ulonglong
    env=dict(os.environ,ANDROID_USER_HOME=str(Path.home()/'.android'))
    process=None;seq=tick()<<16;stats=[];error=None;tmp=dest.with_suffix('.tmp');write_drops=0
    recording=None;recorded=0
    def publish(packet):
        return publish_atomic(tmp,dest,packet)
    try:
        if args.record:recording=args.record.open('xb')
        process=subprocess.Popen([str(args.adb),'-s','frame','shell','-T','python3','/home/steamos/devkit-game/FrameHands/pc-input/export_frame.py','--seconds',str(args.seconds)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env)
        replies=queue.Queue(maxsize=2)
        def reader():
            while True:
                line=process.stdout.readline(8192)
                if not line:replies.put(None);return
                try:data=base64.b64decode(line.strip(),validate=True)
                except ValueError:replies.put(None);return
                replies.put(data)
        threading.Thread(target=reader,daemon=True).start()
        end=time.monotonic()+args.seconds;last_key=None;last_packet=None
        while time.monotonic()<end and not (args.stop_file and args.stop_file.exists()):
            started=tick();process.stdin.write(b'R');process.stdin.flush()
            data=replies.get(timeout=.25)
            if data is None:raise RuntimeError('Frame exporter exited')
            received=tick()
            seq+=1;packet,stat=convert(data,started,received,seq,args.controls)
            key=(stat['valid'],stat['source_sequence'],stat['source_capture'])
            reused=key==last_key and stat['valid']
            if reused:
                packet=last_packet # repeated polls never retime a captured image
            else:
                last_key=key;last_packet=packet
            published=publish(packet)
            if not published:write_drops+=1
            if recording:
                record=RELAY.pack(b'FHRELAY1',1,RELAY_SIZE,started,received,int(reused),int(published))+data+packet
                if recording.write(record)!=len(record):raise OSError('Partial transport trace write')
                recorded+=1
            stats.append(stat)
            time.sleep(.005)
    except KeyboardInterrupt:
        error='interrupted'
    except Exception as exc:
        error=str(exc)
    finally:
        seq+=1;now=tick()
        try:
            if not publish(HEADER.pack(b'FHPCVR1\0',1,568,seq,now,now,0,0)+bytes(520)):write_drops+=1
        except OSError as exc:
            error=error or str(exc) # Native 150 ms expiry still releases all inputs.
        if process:
            if process.poll() is None:
                try:process.stdin.close();process.wait(timeout=2)
                except (OSError,subprocess.TimeoutExpired):process.terminate();process.wait(timeout=3)
        lock.unlink(missing_ok=True);tmp.unlink(missing_ok=True)
        if recording:
            try:recording.flush();os.fsync(recording.fileno());recording.close()
            except OSError as exc:error=error or str(exc)
        def percentile(key,q):
            values=sorted(s[key] for s in stats if s['valid']);return values[int((len(values)-1)*q)] if values else None
        unique={(s['source_sequence'],s['source_capture']):s for s in stats if s['valid']}
        report=dict(error=error,write_drops=write_drops,packets=len(stats),valid=sum(s['valid'] for s in stats),both=sum(s['hands']==2 for s in stats),unique_observations=len(unique),unique_rejected_hands=sum(s['rejected_hands'] for s in unique.values()),unique_hand_count_histogram={str(n):sum(s['hands']==n for s in unique.values()) for n in range(3)},rtt_ms_p50=percentile('rtt_ms',.5),rtt_ms_p95=percentile('rtt_ms',.95),source_age_ms_p95=percentile('age_ms',.95),controls=args.controls,scope='transport counters; not gameplay acceptance')
        report.update(recording=str(args.record) if args.record else None,recorded_packets=recorded)
        args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
    if error and error!='interrupted':raise SystemExit(1)


if __name__=='__main__':main()
