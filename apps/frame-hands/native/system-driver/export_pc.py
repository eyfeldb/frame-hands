"""Read-only pose export over the restricted authenticated wireless pipe. No capture, listener or archive."""
import base64,json,os,struct,sys,time
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'native'))
from rig_pose_subscriber import PoseProcess
ROOT=Path('/run/user')/str(os.getuid())/'frame-hands-native'

def pose_reply(pose):
    try:
        data=(ROOT/'input.bin').read_bytes()
        if len(data)!=592:raise ValueError('Invalid native pose size')
        magic,version,size,seq,published,source=struct.unpack_from('<8sIIQQQ',data)
        now=time.monotonic_ns()//1000000
        if magic!=b'FHMODEL2' or version!=2 or size!=592 or not 0<=now-published<80 or not 0<=now-source<=500:
            return dict(state='waiting',reason='Native hand output is stale')
        head=pose.query(published*1000000)
        if not head.get('valid'):return dict(state='waiting',reason='Native head pose unavailable')
        mode=(ROOT/'mode.bin').read_bytes()
        if len(mode)!=24:raise ValueError('Invalid input mode')
        tag,epoch,selection,_=struct.unpack('<8sQII',mode)
        if tag!=b'FHMODE1\0' or selection not in (1,2):raise ValueError('Invalid input mode')
        return dict(state='native',version=1,native_now_ms=time.monotonic_ns()//1000000,
                    packet=base64.b64encode(data).decode(),head=head['capture'],mode=[epoch,selection])
    except FileNotFoundError:return dict(state='waiting',reason='Launch FrameHandsStart on the headset')

def read_mode():
    try:
        mode=(ROOT/'mode.bin').read_bytes()
        if len(mode)!=24:return None
        tag,epoch,selection,reserved=struct.unpack('<8sQII',mode)
        if tag==b'FHMODE1\0' and epoch>0 and selection in (1,2) and reserved==0:return [epoch,selection]
    except OSError:pass
    return None


def reply(pose):
    result=pose_reply(pose)
    mode=read_mode()
    if mode is not None:result['mode']=mode
    return result


def stream(pose):
    """Push a reply whenever the native output changes (and at least every 250 ms) until stdin closes or says quit.
    Removes the request round trip of the polled 'read' protocol: the PC sees each 30 Hz publication once, as soon
    as it exists, instead of whenever its next poll lands."""
    import threading
    stop=threading.Event()
    def watch_stdin():
        while True:
            line=sys.stdin.buffer.readline(32)
            if not line or line.strip()==b'quit':break
        stop.set()
    threading.Thread(target=watch_stdin,daemon=True).start()
    last_stat=None;last_sent=0.0
    while not stop.is_set():
        try:
            st=(ROOT/'input.bin').stat();stat=(st.st_mtime_ns,st.st_size)
        except OSError:stat=None
        now=time.monotonic()
        if stat!=last_stat or now-last_sent>=.25:
            last_stat=stat;last_sent=now
            ROOT.mkdir(mode=0o700,exist_ok=True)
            (ROOT/'pc-receiver.json').write_text(json.dumps(dict(updated_mono_ns=time.monotonic_ns(),transport='wifi-stream')))
            print(json.dumps(reply(pose),separators=(',',':')),flush=True)
        else:time.sleep(.002)

def main():
    with open('/dev/null','wb') as log:
        pose=PoseProcess([str(BASE/'pose-service'),'/opt/steamvr/bin/linuxarm64/libopenvr_api.so','continuous'],log)
        try:
            while True:
                line=sys.stdin.buffer.readline(32)
                if not line or line.strip()==b'quit':break
                if line.strip()==b'stream':stream(pose);break
                if line.strip()!=b'read':raise ValueError('Unknown request')
                ROOT.mkdir(mode=0o700,exist_ok=True)
                (ROOT/'pc-receiver.json').write_text(json.dumps(dict(updated_mono_ns=time.monotonic_ns(),transport='wifi')))
                print(json.dumps(reply(pose),separators=(',',':')),flush=True)
        finally:pose.close()
if __name__=='__main__':main()
