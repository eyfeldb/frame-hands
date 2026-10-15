"""Consolidate only the three original hand-control shortcuts into Frame Hands."""
import json,shutil,subprocess,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
ROOT=Path.home()/'devkit-game'
def main():
    target=ROOT/'FrameHands';target.mkdir(exist_ok=True)
    launch=target/'launch.sh';launch.write_text('#!/bin/sh\nexec /usr/bin/python3 '+str(BASE/'menu.py')+'\n');launch.chmod(0o755)
    config=dict(gameid='FrameHands',directory=str(target),argv=['launch.sh'],env={},settings={'steam_play':'0','compat_tool':''},force_appid='')
    result=subprocess.run(['python3',str(Path.home()/'devkit-utils/steam-client-create-shortcut'),'--parms',json.dumps(config)],text=True,capture_output=True,check=True,timeout=45)
    reply=json.loads(result.stdout)
    if 'error' in reply:raise RuntimeError(reply['error'])
    backup=Path.home()/'.local/share/frame-hands/shortcut-backups';backup.mkdir(parents=True,exist_ok=True)
    # Move launcher directories outside Steam's devkit library; preserve all content.
    for name in ('FrameHandsStart','FrameHandsStop','FrameHandsControllers'):
        source=ROOT/name;dest=backup/name
        if source.exists():
            if dest.exists():raise RuntimeError('Existing shortcut backup needs inspection: '+name)
            if source.resolve().parent!=ROOT.resolve():raise RuntimeError('Unexpected shortcut path')
            shutil.move(str(source),str(dest))
    sys.path.insert(0,str(Path.home()/'devkit-utils'))
    import devkit_utils.resolve
    devkit_utils.resolve.resolve_shortcuts()
    print('Single FrameHands shortcut installed; old control shortcuts preserved outside Library.')
if __name__=='__main__':main()
