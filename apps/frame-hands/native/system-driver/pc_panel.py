"""Manual wireless-receiver controls. Cameras are controlled on Frame only."""
import argparse,json,subprocess,sys,time
from pathlib import Path
import tkinter as tk
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[3]
HIDDEN=getattr(subprocess,'CREATE_NO_WINDOW',0)
def main():
    p=argparse.ArgumentParser();p.add_argument('--connection',type=Path,required=True);a=p.parse_args()
    stop=a.connection.parent/'receiver.stop'
    state=ROOT/'.local/hand-alternatives/native-pc-relay/wireless-status.json'
    window=tk.Tk();window.title('Frame Native Hands - Wireless Receiver');window.geometry('680x320')
    tk.Label(window,text='HAND TRACKING ON FRAME',font=('Segoe UI',20,'bold')).pack(pady=18)
    tk.Label(window,text='Wi-Fi carries finished hand poses to PC VR. No USB or PC inference.\nOn Frame, use FrameHandsStart / FrameHandsStop to control tracking.',font=('Segoe UI',12)).pack(pady=10)
    text=tk.StringVar(value='Checking receiver status...');tk.Label(window,textvariable=text,wraplength=630,font=('Segoe UI',11)).pack(pady=12)
    def start():
        stop.unlink(missing_ok=True)
        manifest=json.loads((a.connection.parent/'receiver.vrmanifest').read_text())['applications'][0]
        subprocess.Popen(subprocess.list2cmdline([manifest['binary_path_windows']])+' '+manifest['arguments'],cwd=manifest['working_directory'],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=HIDDEN)
        text.set('Connecting wireless receiver...')
    def end():stop.write_text('Stop receiver requested');text.set('Stopping receiver; native tracking is unchanged.')
    def poll():
        try:
            s=json.loads(state.read_text())
            text.set(s['status'] if time.time()-s['updated']<8 else 'Receiver is not connected.')
        except (OSError,ValueError):pass
        window.after(700,poll)
    tk.Button(window,text='Connect receiver',command=start,font=('Segoe UI',13)).pack(pady=5)
    tk.Button(window,text='Stop receiver',command=end,font=('Segoe UI',12)).pack(pady=5)
    tk.Label(window,text='SteamVR starts the receiver automatically. Closing this window leaves it running.').pack(pady=10)
    window.after(100,poll);window.mainloop()
if __name__=='__main__':main()
