"""Single headset entry; never selects a mode until the user presses a menu button."""
import fcntl,json,subprocess,struct,time
from pathlib import Path
BASE=Path(__file__).resolve().parent
TRACKER=Path('/home/steamos/devkit-game/FrameHandsNativeSystem-v1')
def main():
    runtime=Path('/run/user/1000/frame-hands-native');runtime.mkdir(mode=0o700,exist_ok=True)
    with (runtime/'menu.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        active=subprocess.run(['systemctl','--user','is-active','--quiet','frame-hands-native.service']).returncode==0
        mode='Controllers'
        try:
            if struct.unpack('<8sQII',(runtime/'mode.bin').read_bytes())[2]==1:mode='Hands'
        except (OSError,struct.error):pass
        state='Tracking is stopped.'
        if active:
            state='Tracking is running. Selected: '+mode+'.'
            try:
                packet=(runtime/'input.bin').read_bytes()
                render_ms=struct.unpack_from('<Q',packet,24)[0]
                if time.monotonic_ns()//1000000-render_ms>500:state='Tracker is started but waiting for fresh tracking. Wake Frame. Selected: '+mode+'.'
            except (OSError,struct.error):state='Tracker is starting or waiting for fresh tracking. Selected: '+mode+'.'
        text=state+'\n\nHands starts tracking or selects your hands.\nControllers returns to physical controllers.\nStop tracking releases the cameras.\n\nA physical controller button also returns to Controllers. No recording.'
        selected=subprocess.run([str(BASE/'menu'),text]).returncode
        if selected in (0,1,2):
            subprocess.run(['python3',str(TRACKER/'control.py'),('start','controllers','stop')[selected]],check=True)
        elif selected not in (3,6):
            subprocess.run([str(TRACKER/'notify'),'Frame Hands menu could not open. Input mode was not changed.'])
if __name__=='__main__':main()
