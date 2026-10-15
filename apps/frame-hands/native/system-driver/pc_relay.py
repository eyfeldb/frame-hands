"""Authenticated wireless transport of Frame-computed hands to the existing PC driver."""
import argparse,json,os,queue,struct,subprocess,sys,threading,time
from pathlib import Path
from pc_transport import Converter,unpack
BASE=Path(__file__).resolve().parent
HANDS=BASE.parents[1]
ROOT=HANDS.parents[1]
REMOTE='/home/steamos/devkit-game/FrameHandsNativeSystem-v1'
HIDDEN=getattr(subprocess,'CREATE_NO_WINDOW',0)
LAST_NATIVE_MODE=None
def tick():return time.perf_counter_ns()//1000000

def status(value):
    directory=ROOT/'.local/hand-alternatives/native-pc-relay';directory.mkdir(parents=True,exist_ok=True)
    path=directory/'wireless-status.json';temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(pid=os.getpid(),updated=time.time(),**value)))
    os.replace(temporary,path)


class Rpc:
    def __init__(self,args,env=None):
        self.child=subprocess.Popen(args,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,creationflags=HIDDEN)
        self.responses=queue.Queue(maxsize=1)   # latest reply wins: a slow consumer never builds a backlog
        def read():
            try:
                while True:
                    line=self.child.stdout.readline(10001)
                    if not line:raise EOFError('Pose connection closed')
                    if len(line)>10000 or not line.endswith(b'\n'):raise ValueError('Oversized pose reply')
                    self.offer((json.loads(line),tick()))
            except Exception as e:self.offer((e,tick()))
        threading.Thread(target=read,daemon=True).start()
    def offer(self,item):
        while True:
            try:self.responses.put_nowait(item);return
            except queue.Full:
                try:self.responses.get_nowait()
                except queue.Empty:pass
    def send(self,line):
        self.child.stdin.write((line+'\n').encode());self.child.stdin.flush()
    def ask(self,line):
        self.send(line)
        result,_=self.responses.get(timeout=3)
        if isinstance(result,Exception):raise result
        return result
    def latest(self,timeout):
        """Next streamed reply and its arrival time (ms), or None after timeout."""
        try:result,received=self.responses.get(timeout=timeout)
        except queue.Empty:return None
        if isinstance(result,Exception):raise result
        return result,received
    def close(self):
        try:self.child.stdin.close()
        except OSError:pass
        try:self.child.wait(timeout=2)
        except subprocess.TimeoutExpired:self.child.terminate();self.child.wait(timeout=3)

class OneWay:
    """One-way delay of streamed replies from (arrival - sender clock), relative to the fastest arrival in a
    sliding window: the two clocks share no origin, but the minimum offset over a few seconds is the baseline.
    The first replies are taken at face value; the estimate settles within a second."""
    def __init__(self,window=150):self.window=window;self.offsets=[]
    def delay(self,received,native_now):
        d=received-native_now;self.offsets.append(d)
        if len(self.offsets)>self.window:del self.offsets[0]
        return max(0,d-min(self.offsets))

def wireless_command(config):
    value=json.loads(config.read_text())
    for field in ('ssh','key','known_hosts'):
        if not Path(value[field]).is_file():raise RuntimeError('Wireless pairing is not installed: '+field)
    return [value['ssh'],'-T','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes',
            '-o','UserKnownHostsFile="'+Path(value['known_hosts']).as_posix()+'"','-o','ConnectTimeout=5','-o','ServerAliveInterval=5',
            '-o','ServerAliveCountMax=2','-i',value['key'],value['user']+'@'+value['host'],'frame-hands-pose-export']

def main():
    global LAST_NATIVE_MODE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--connection',type=Path,required=True)
    parser.add_argument('--stop-file',type=Path,required=True)
    parser.add_argument('--watch',action='store_true',help='Receiver only; never starts cameras or tracking')
    args=parser.parse_args()
    if os.name!='nt':raise RuntimeError('PC relay requires Windows SteamVR')
    sys.path.insert(0,str(HANDS/'exploratory'))
    sys.path.insert(0,str(HANDS/'pc-input'))
    from steamvr_sink import verify_live_driver
    from select_input_mode import request_mode
    from relay import publish_atomic
    deployment=ROOT/'.local/hand-alternatives/steamvr-model/deployment.json'
    dep=verify_live_driver(deployment)
    dll=Path('C:/Program Files (x86)/Steam/steamapps/common/SteamVR/bin/win64/openvr_api.dll')
    directory=ROOT/'.local/hand-alternatives/native-pc-relay';directory.mkdir(parents=True,exist_ok=True)
    import msvcrt
    lock=(directory/'relay.lock').open('a+b');lock.seek(0)
    if os.fstat(lock.fileno()).st_size==0:lock.write(b'0');lock.flush()
    lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:
        lock.close()
        raise RuntimeError('Native hands PC relay is already running')
    command=wireless_command(args.connection)
    dest=Path(dep['bundle'])/'bin/win64/input.bin';temp=dest.with_name('native-relay.tmp')
    owned=False;remote=None;pose=None;converter=Converter();count=hands=0;last_status=0;reason='starting';max_rtt=0;idle_since=None
    mode_receipt=directory/'applied-native-mode.json'
    if LAST_NATIVE_MODE is None:
        try:LAST_NATIVE_MODE=json.loads(mode_receipt.read_text())['epoch']
        except (OSError,ValueError,KeyError):pass
    sequence=time.time_ns()
    try:
        print('Wireless receiver ready. Launch FrameHandsStart on the headset; all inference stays there.',flush=True)
        remote=Rpc(command)
        pose=Rpc([str(ROOT/'.local/hand-alternatives/pose-service.exe'),str(dll)])
        # One 'read' proves the pipe, then the headset streams every publication (export_pc.py 'stream').
        remote.ask('read');remote.send('stream');oneway=OneWay();delays=[]
        def percentiles():
            if not delays:return {}
            d=sorted(delays);return dict(delay_p50_ms=d[len(d)//2],delay_p90_ms=d[int(len(d)*.9)],delay_max_ms=d[-1])
        while not args.stop_file.exists():
            item=remote.latest(timeout=.5)
            # 2026-10-12: the headset only reports its Hands/Controllers choice inside native replies. When the
            # native service is not running (the wearer chose Controllers from the menu, a physical takeover
            # stopped it, or it died), the PC driver used to stay in Hands and keep the hand roles, leaving the
            # physical controllers dead in the game. No native output for 2 s => Controllers on the PC.
            if item is not None and item[0].get('state')!='native':
                if idle_since is None:idle_since=time.monotonic()
                elif time.monotonic()-idle_since>2 and LAST_NATIVE_MODE!='idle-controllers':
                    request_mode('controllers',deployment);LAST_NATIVE_MODE='idle-controllers'
                    print('PC driver set to Controllers: no native hand output',flush=True)
            elif item is not None:idle_since=None
            if item is None:
                reason='no pose stream from the headset'
                if time.monotonic()-last_status>2:
                    value=dict(status=reason,packets=count,hand_outputs=hands,max_rtt_ms=max_rtt,inference='Frame',transport='wifi-stream',recording=False)
                    status(value);print(json.dumps(value),flush=True);last_status=time.monotonic()
                continue
            reply,received=item
            native_now=reply.get('native_now_ms')
            delay=oneway.delay(received,native_now) if type(native_now)==int else 0
            started=received-delay;max_rtt=max(max_rtt,2*delay);delays.append(delay)
            if len(delays)>3000:del delays[:1000]
            native_mode=reply.get('mode')
            if native_mode is not None:
                if not isinstance(native_mode,list) or len(native_mode)!=2:raise ValueError('Invalid native input selection')
                epoch,mode=native_mode
                if type(epoch)!=int or epoch<=0 or type(mode)!=int or mode not in (1,2):raise ValueError('Invalid native input selection')
                if epoch!=LAST_NATIVE_MODE:
                    request_mode('hands' if mode==1 else 'controllers',deployment)
                    temporary=mode_receipt.with_suffix('.tmp');temporary.write_text(json.dumps(dict(epoch=epoch,mode=mode)))
                    os.replace(temporary,mode_receipt);LAST_NATIVE_MODE=epoch
            decoded=unpack(reply,started,received)
            reason=reply.get('reason','waiting for fresh poses')
            if decoded:
                head=pose.ask(str(max(0,decoded['pose_age_ms']+tick()-received)))
                sequence+=1;data=converter.convert(decoded,head,started,received,sequence,tick())
                if data:
                    if publish_atomic(temp,dest,data):
                        owned=True;count+=1
                        current=sum(struct.unpack_from('<I',data,44+276*s)[0]!=0 for s in range(2));hands+=current
                        reason='Native hands reaching PC driver: '+str(current)+'/2'
                elif not head.get('valid'):reason='PC head pose is unavailable'
            elif owned and reply.get('state')!='native':
                dest.unlink(missing_ok=True);owned=False
            if time.monotonic()-last_status>2:
                value=dict(status=reason,packets=count,hand_outputs=hands,max_rtt_ms=max_rtt,inference='Frame',transport='wifi-stream',recording=False,**percentiles())
                status(value);print(json.dumps(value),flush=True);last_status=time.monotonic()
    finally:
        if owned:dest.unlink(missing_ok=True)
        temp.unlink(missing_ok=True)
        if remote:remote.close()
        if pose:pose.close()
        lock.close()
        status(dict(status='Receiver stopped',recording=False));print('PC wireless receiver stopped. Native tracking and games were not stopped.',flush=True)
if __name__=='__main__':
    if sys.stdout is None:sys.stdout=open(os.devnull,'w')
    if sys.stderr is None:sys.stderr=open(os.devnull,'w')
    if '--watch' not in sys.argv:main()
    else:
        # SteamVR companion lifecycle. Waiting for a native Start never opens cameras.
        stop=Path(sys.argv[sys.argv.index('--stop-file')+1])
        while not stop.exists():
            try:main()
            except Exception as error:
                if 'already running' in str(error):break
                status(dict(status='Waiting: '+str(error),recording=False));print('Wireless receiver waiting: '+str(error),flush=True)
            if stop.exists():break
            time.sleep(5)
            running=subprocess.run(['tasklist','/FI','IMAGENAME eq vrserver.exe','/NH'],capture_output=True,text=True,creationflags=HIDDEN)
            if 'vrserver.exe' not in running.stdout.lower():break
