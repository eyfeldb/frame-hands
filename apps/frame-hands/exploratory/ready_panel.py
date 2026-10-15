"""Manual PC VR test launcher. Opening the panel never starts XR or capture."""
import argparse
import json
import os
from pathlib import Path
import queue
import shlex
import subprocess
import sys
import threading
import time

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
HIDDEN = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0

# Readiness is checked only after an explicit click, then camera ownership is
# acquired independently by the existing bounded source immediately before use.
REMOTE_CHECK = r'''
import fcntl, os, pathlib, subprocess, sys
base = pathlib.Path('/home/steamos/devkit-game')
service = subprocess.run(['systemctl', 'is-active', 'frame-eyes-reader-brightness.service'], capture_output=True, text=True)
if service.stdout.strip() in ('active', 'activating', 'deactivating'):
    sys.exit('Eye tracking is using Frame. Finish that test, then click Start again.')
for relative in ('FrameEyes/workflow.lock', 'FrameHands/workflow.lock', 'FrameHands/trial.lock'):
    path = base / relative
    if path.exists():
        with path.open('r+') as handle:
            try: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: sys.exit('Another camera workflow is busy: ' + relative)
for path in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
    if path.parent.name == str(os.getpid()): continue
    try: args = path.read_bytes().decode(errors='replace').split('\0')
    except OSError: continue
    names = [pathlib.Path(arg).name.lower() for arg in args[:2] if arg]
    if any(name.startswith(('ft-hands', 'frame-camera-reader', 'frame-eyes-reader')) or name in ('run_source.py', 'run_native_trial.py') for name in names):
        sys.exit('A camera/tracking process is still active. Finish it before starting this test.')
print('USB is connected and the Frame camera workflow is available.')
'''


NATIVE_CHECK = r'''
import json, pathlib, subprocess, sys
for path in pathlib.Path('/proc').glob('[0-9]*/comm'):
    try:name=path.read_text().strip()
    except OSError:continue
    if name.startswith(('vrlink','run_vrlink')):sys.exit('Disconnect PC VR streaming and use Frame standalone. Keep USB connected.')
result=subprocess.run(['python3','/home/steamos/devkit-game/FrameHands/experiments/piano-native-link-v1/native_piano_link.py','--check'],capture_output=True,text=True,timeout=6)
if result.returncode:sys.exit('Native Frame head tracking is not ready. Put Frame on in its standalone environment, then press Start.')
print('Native Frame tracking is ready; no PC VR scene is needed.')
'''

def preflight(workspace, native=False, steamvr=None):
    if steamvr:
        from steamvr_sink import verify_live_driver
        verify_live_driver(steamvr)
    adb = workspace / '.local/android-tools/platform-tools/adb.exe'
    env = dict(os.environ, ANDROID_USER_HOME=str(Path.home() / '.android'))
    result = subprocess.run([str(adb), '-s', 'frame', 'shell', 'python3 -c ' + shlex.quote(REMOTE_CHECK)], env=env, capture_output=True, text=True, timeout=20, creationflags=HIDDEN)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip() or 'Frame is not available over USB.')
    if native:
        state=subprocess.run([str(adb),'-s','frame','shell','python3 -c '+shlex.quote(NATIVE_CHECK)],env=env,capture_output=True,text=True,timeout=12,creationflags=HIDDEN)
        if state.returncode:raise RuntimeError((state.stderr or state.stdout).strip())
        return result.stdout.strip()+' '+state.stdout.strip()
    probe = ROOT / '.local/hand-alternatives/readiness-probe.exe'
    dll = Path('C:/Program Files (x86)/Steam/steamapps/common/SteamVR/bin/win64/openvr_api.dll')
    state = subprocess.run([str(probe),str(dll)],capture_output=True,text=True,timeout=10,creationflags=HIDDEN)
    if state.returncode:raise RuntimeError('Could not query PC SteamVR. Open SteamVR, then click Start again.')
    require_pc_vr(json.loads(state.stdout))
    return result.stdout.strip() + ' PC SteamVR has a connected, tracked Frame headset.'


def require_pc_vr(state):
    if 'runtime_error' in state:
        raise RuntimeError('USB is connected, but PC SteamVR is not ready. Open SteamVR and connect Frame for VR streaming.')
    if not state.get('connected'):
        raise RuntimeError('USB is connected, but PC SteamVR reports the headset disconnected. In Frame, start PC VR streaming to this PC; a streamed desktop alone is not a VR connection.')
    if not state.get('frame_hmd'):
        raise RuntimeError('PC SteamVR is connected to a different headset. Select Frame for this PC VR session.')
    if not state.get('pose_valid'):
        raise RuntimeError('Frame is connected to PC SteamVR, but its head tracking is not ready. Put it on, wake it, and wait for the PC VR view before pressing Start.')


def trial_command(workspace, piano=None, stop_file=None, native=False, steamvr=None):
    command = [str(ROOT / '.local/hand-alternatives/venv-cuda/Scripts/python.exe'), '-u', str(BASE / 'run_mesh_trial.py'), '--workspace', str(workspace), '--seconds', '90', '--camera-reader', 'exact-v3', '--identity', 'image-cuda', '--presentation', 'predict', '--continuity', 'native', '--ready']
    if steamvr:
        command += ['--continuous','--steamvr',str(steamvr.resolve())]
        command[command.index('--continuity')+1]='bridge'
    elif native:
        command += ['--continuous','--native-piano']
        command[command.index('--continuity')+1]='bridge'
    elif piano:
        command += ['--continuous']
        command[command.index('--continuity')+1]='bridge'
        command += ['--piano',str(piano.resolve())]
    if stop_file:command += ['--stop-file',str(stop_file)]
    return command


class ManualGate:
    """No timer or automatic transition into a trial. One click owns one run."""
    def __init__(self, check, launch):
        self.check = check
        self.launch = launch
        self.busy = False
        self.guard = threading.Lock()

    def start_clicked(self):
        with self.guard:
            if self.busy: return None
            self.busy = True
        try:
            self.check()
            return self.launch()
        finally:
            with self.guard: self.busy = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--piano',type=Path,help='Open Piano with optical tracking until Stop or app exit')
    parser.add_argument('--native-piano',action='store_true')
    parser.add_argument('--steamvr',type=Path)
    parser.add_argument('--start-now',action='store_true',help='One explicit user-requested start; ordinary shortcuts remain manual')
    args = parser.parse_args()
    if sum(bool(x) for x in (args.native_piano,args.piano,args.steamvr))>1:parser.error('Choose native Piano, PC Piano, or SteamVR hands')
    workspace = args.workspace.resolve()
    output = ROOT / ('.local/hand-alternatives/steamvr-panel' if args.steamvr else '.local/hand-alternatives/piano-panel' if args.piano or args.native_piano or args.steamvr else '.local/hand-alternatives/ready-panel')
    output.mkdir(parents=True, exist_ok=True)
    import msvcrt
    lock = (output / 'panel.lock').open('a+b')
    lock.seek(0)
    if not lock.read(1): lock.write(b'0'); lock.flush()
    lock.seek(0)
    try: msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        print('The Frame Hands Start panel is already open.')
        return 1

    import tkinter as tk
    from tkinter import messagebox
    window = tk.Tk()
    window.title('Frame Hands - SteamVR' if args.steamvr else 'Frame Piano - Native App / PC Hands' if args.native_piano else 'Frame Piano - Optical Hands - Start' if args.piano else 'Frame Hands - PC Pose Correction - Start')
    window.geometry('740x520')
    window.minsize(660, 420)
    window.configure(bg='#111d2d')
    events = queue.Queue()
    state_path = output / 'state.json'
    current_log = [None]
    stop_path = [None]

    def record(state, **extra):
        state_path.write_text(json.dumps(dict(state=state, pid=os.getpid(), updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), auto_start=False, **extra), indent=2), encoding='utf-8')

    def label(text, size, color='#edf3ff'):
        node = tk.Label(window, text=text, font=('Segoe UI', size), bg='#111d2d', fg=color, wraplength=650, justify='center')
        node.pack(pady=12)
        return node

    label('STEAMVR / PC HAND TRACKING' if args.steamvr else 'NATIVE FRAME PIANO / PC HANDS' if args.native_piano else 'FRAME PIANO / OPTICAL HANDS' if args.piano else 'PC HANDS / POSE CORRECTION', 22)
    label('Waiting for you - no timeout', 16, '#77d7c7')
    def piano_build_text():
        pack=args.piano.with_suffix('.pck')
        try:stamp=time.strftime('%b %d, %Y %I:%M %p',time.localtime(pack.stat().st_mtime))
        except OSError:stamp='build missing'
        return f'Piano: {args.piano.parent.name} / {stamp}'
    build_label=label(piano_build_text(),10,'#c1cfdf') if args.piano else None
    label('Keep USB and PC VR streaming connected.\nStart tracking, then open Maestro or another compatible game.\nA physical controller button switches to controllers.' if args.steamvr else 'Keep USB connected. Disconnect PC VR streaming.\nPress Start tracking, then open Devkit Game: FramePiano\non the headset. The PC runs hand inference only.' if args.native_piano else 'Put Frame on, connect PC VR streaming, and keep USB connected.\nLeave Maestro and other tracking tests closed.\nThen press Start here, or from SteamVR\'s Desktop view.', 12)
    status = label(('No recording or time limit. Start when Frame is awake.' if args.steamvr else 'No recording and no time limit. Stop tracking leaves Piano open.' if args.native_piano else 'Tracking stays on until Stop hands or you close Piano.\nNo full video recording; USB and PC VR must stay connected.' if args.piano else 'Nothing is recording. Start confirms you are wearing Frame\nand can see the PC VR environment.'), 11, '#c1cfdf')

    def launch_trial():
        path = output / ('trial-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '.log')
        current_log[0] = path
        stop_path[0]=path.with_suffix('.stop')
        command = trial_command(workspace,args.piano,stop_path[0],native=args.native_piano,steamvr=args.steamvr)
        with path.open('xb') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=HIDDEN)
            events.put(('running', dict(child_pid=child.pid, log=str(path), command=command)))
            for line in iter(child.stdout.readline,b''):
                log.write(line);log.flush()
                if line.startswith((b'Native hand feed ready.',b'SteamVR model hands ready.')):
                    events.put(('tracking_ready',dict(child_pid=child.pid,log=str(path))))
                elif line.startswith((b'Native hand feed waiting',b'SteamVR model hands waiting')):
                    events.put(('tracking_waiting',dict(child_pid=child.pid,log=str(path))))
            code = child.wait()  # Worker only; UI stays responsive during capture/copy.
        events.put(('finished', dict(exit_code=code, log=str(path))))
        return code

    gate = ManualGate(lambda: preflight(workspace,native=args.native_piano,steamvr=args.steamvr), launch_trial)

    def clicked():
        if start['state'] == 'disabled': return
        start.config(state='disabled')
        if build_label:build_label.config(text=piano_build_text())
        status.config(text='Checking USB, standalone head tracking and camera availability...' if args.native_piano else 'Checking USB, PC streaming and camera availability...')
        record('checking')
        def work():
            try: gate.start_clicked()
            except Exception as exc: events.put(('failed', dict(error=str(exc))))
        threading.Thread(target=work, daemon=True).start()

    start = tk.Button(window, text='Start SteamVR hands' if args.steamvr else 'Start PC hand tracking' if args.native_piano else 'Start hands + Piano' if args.piano else 'Start 90-second test', command=clicked, font=('Segoe UI', 17, 'bold'), bg='#69d4bf', fg='#10202a', activebackground='#9de6d8', padx=24, pady=12)
    start.pack(pady=8)

    def stop_clicked():
        if stop_path[0] and start['state']=='disabled':
            stop_path[0].write_text('User requested stop\n',encoding='utf-8')
            status.config(text='Stopping optical hands...' if args.piano or args.native_piano or args.steamvr else 'Stopping tracking and saving the recording. Keep USB connected.')
    stop = tk.Button(window,text='Stop tracking' if args.steamvr or args.native_piano else 'Stop hands' if args.piano else 'Stop and save',command=stop_clicked,state='disabled',font=('Segoe UI',11))
    stop.pack(pady=3)

    def close():
        if start['state'] == 'disabled':
            messagebox.showinfo('Test still running', 'Use Stop first, then close this window.' if args.piano or args.native_piano or args.steamvr else 'The bounded test or recording copy is still running. Keep USB connected. This window will show when it finishes.')
            return
        record('closed')
        window.destroy()
    window.protocol('WM_DELETE_WINDOW', close)

    def poll():
        while not events.empty():
            kind, data = events.get_nowait()
            record(kind, **data)
            if kind == 'running':
                stop.config(state='normal')
                status.config(text='Preparing hand tracking for SteamVR games...' if args.steamvr else 'Starting the native hand feed. Open Piano on Frame when ready.' if args.native_piano else 'Preparing models, then opening Piano.' if args.piano else 'Preparing models, then opening the solid-hand scene.')
            elif kind == 'tracking_ready':
                status.config(text='SteamVR hands are live. Open Maestro; close the SteamVR dashboard.\nUse Input mode to reselect Hands after using a controller.' if args.steamvr else 'Hand feed ready. Open Devkit Game: FramePiano on the headset.\nStop tracking leaves Piano running.')
            elif kind == 'tracking_waiting':
                status.config(text='Waiting for camera frames. Keep Frame awake.\nThe connection stays open; hands resume when capture returns.')
            elif kind == 'finished':
                start.config(state='normal', text='Start SteamVR hands' if args.steamvr else 'Start PC hand tracking' if args.native_piano else 'Start hands + Piano' if args.piano else 'Start another 90-second test')
                stop.config(state='disabled')
                status.config(text=(('Optical hands stopped.' if args.piano or args.native_piano or args.steamvr else 'Test and recording copy finished.') if data['exit_code'] == 0 else 'Tracking stopped with an error. Open the log for details.') + '\nNo automatic restart.')
            elif kind == 'failed':
                stop.config(state='disabled')
                start.config(state='normal')
                status.config(text=data['error'])
        window.after(250, poll)

    if args.steamvr:
        def select_mode():
            subprocess.Popen([sys.executable,str(BASE.parent/'pc-input/select_input_mode.py'),'--ui','--deployment',str(args.steamvr)],creationflags=HIDDEN)
        tk.Button(window,text='Input mode: Hands / Controllers',command=select_mode,font=('Segoe UI',10)).pack(pady=3)

    def open_log():
        if current_log[0]: subprocess.Popen(['notepad.exe', str(current_log[0])])
    tk.Button(window, text='Open test log', command=open_log, font=('Segoe UI', 10)).pack(pady=5)
    window.after(250, poll)
    def shown():
        if start['state'] != 'disabled':
            record('waiting', window_title=window.title(), visible=bool(window.winfo_viewable()), start_button=start['text'], no_camera_or_xr_started=True)
        window.lift()
    window.after(300, shown)
    if args.start_now:window.after(500,clicked)
    window.mainloop()
    lock.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
