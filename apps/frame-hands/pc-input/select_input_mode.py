"""Explicit mode for the isolated mode-capable driver. Never starts capture."""
import argparse
import json
from pathlib import Path
import struct
import time
from relay import publish_atomic

def request_mode(mode, deployment_path):
    if mode not in ('auto','hands','controllers'):
        raise ValueError('Unknown input mode')
    deployment=json.loads(deployment_path.read_text())
    if not deployment.get('input_mode_supported'):
        raise ValueError('This deployment does not support explicit input selection; accepted driver unchanged')
    folder=Path(deployment['bundle'])/'bin/win64';dest=folder/'mode.bin'
    sequence=time.time_ns()
    if dest.exists():
        previous=dest.read_bytes()
        if len(previous)==24 and previous[:8]==b'FHMODE1\0':
            sequence=max(sequence,struct.unpack_from('<Q',previous,8)[0]+1)
    packet=struct.pack('<8sQII',b'FHMODE1\0',sequence,('auto','hands','controllers').index(mode),0)
    if not publish_atomic(folder/'mode.tmp',dest,packet):raise RuntimeError('Mode request could not be published')
    return dict(requested_mode=mode,sequence=sequence,bundle=str(folder.parent.parent),scope='Request only; driver log confirms application. Hands mode yields on a physical controller button, and must be selected again to return.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',nargs='?',choices=('auto','hands','controllers'))
    parser.add_argument('--ui',action='store_true',help='Open a desktop selector; opening it does not change mode')
    parser.add_argument('--deployment',type=Path,default=Path(__file__).with_name('deployment.json'))
    args=parser.parse_args()
    if not args.mode and not args.ui:parser.error('Select a mode or use --ui')
    if args.mode:
        try:print(json.dumps(request_mode(args.mode,args.deployment)),flush=True)
        except (OSError,ValueError,RuntimeError) as exc:parser.error(str(exc))
    if args.ui:
        import tkinter as tk
        from tkinter import ttk
        window=tk.Tk();window.title('Frame Hands - Input mode');window.geometry('480x240')
        frame=ttk.Frame(window,padding=18);frame.pack(fill='both',expand=True)
        ttk.Label(frame,text='Choose your input',font=('Segoe UI',16)).pack(anchor='w')
        ttk.Label(frame,text='Hands stays selected through tracking gaps.\nA physical controller button switches to controllers.\nSelect Hands again to return.',justify='left').pack(anchor='w',pady=10)
        status=tk.StringVar(value='No request sent by this window.')
        def choose(mode):
            try:
                request_mode(mode,args.deployment)
                status.set('Requested: '+mode.capitalize()+'.')
            except (OSError,ValueError,RuntimeError) as exc:status.set(str(exc))
        buttons=ttk.Frame(frame);buttons.pack(anchor='w')
        for mode in ('hands','controllers','auto'):
            ttk.Button(buttons,text=mode.capitalize(),command=lambda m=mode:choose(m)).pack(side='left',padx=(0,8))
        ttk.Label(frame,textvariable=status,wraplength=440).pack(anchor='w',pady=10)
        ttk.Label(frame,text='Selection request only; live driver state is not shown.').pack(anchor='w')
        window.mainloop()

if __name__=='__main__':main()
