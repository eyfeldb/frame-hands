"""Headset Library controls with in-headset success/error notifications."""
import json,os,subprocess,sys
from pathlib import Path
from wire import request_mode
BASE=Path(__file__).resolve().parent
root=Path('/run/user')/str(os.getuid())/'frame-hands-native'

def message(text,wait=True):
    child=subprocess.Popen([str(BASE/'notify'),text[:255]],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    if wait:
        try:child.wait(timeout=9)
        except subprocess.TimeoutExpired:child.terminate()

def main():
    mode=sys.argv[1]
    if mode=='start':
        message('Starting Frame hands. Look at a well-lit room with your hands lowered.',False)
        result=subprocess.run(['systemctl','--user','start','frame-hands-native.service'],capture_output=True,text=True)
        if result.returncode:
            try:reason=json.loads((root/'status.json').read_text()).get('error','Native hands could not start.')
            except (OSError,ValueError):reason='Native hands could not start. Wake Frame and retry.'
            message('Hands NOT running. '+reason);return 1
        request_mode(root,'hands')
        message('Native hands are running for Frame apps. PC VR games receive hands through the wireless receiver on your PC.')
    elif mode=='stop':
        subprocess.run(['systemctl','--user','stop','frame-hands-native.service'],check=True);message('Frame hand tracking stopped.')
    elif mode=='controllers':
        root.mkdir(mode=0o700,exist_ok=True);request_mode(root,'controllers');message('Controllers selected. Launch FrameHandsStart to select hands again.')
    else:raise ValueError('Unknown action')
    return 0
if __name__=='__main__':raise SystemExit(main())
