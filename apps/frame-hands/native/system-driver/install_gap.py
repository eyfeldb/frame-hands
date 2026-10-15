"""Install only the validated native gap processor while its service is stopped."""
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
BASE=Path(__file__).resolve().parent
TARGET=Path('/home/steamos/devkit-game/FrameHandsNativeSystem-v1')
FILES=['native_gap_base.gd','native_gap_continuity.gd','native_gap_motion.gd','native_game_motion.gd','gap-policy.json','native_bridge.gd']
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    if subprocess.run(['systemctl','--user','is-active','--quiet','frame-hands-native.service']).returncode==0:raise RuntimeError('Stop our native tracker before installing its processor')
    manifest_path=TARGET/'manifest.json';manifest=json.loads(manifest_path.read_text());source=BASE/'processor'
    new={('processor/'+n):sha(source/n) for n in FILES}
    for name in new:
        old=TARGET/name
        if old.exists() and manifest.get(name)!=sha(old):raise RuntimeError('Existing processor differs: '+name)
    backup=BASE/('rollback-'+str(time.time_ns()));backup.mkdir()
    previous={name:manifest.get(name) for name in new}
    (backup/'previous-files.json').write_text(json.dumps(previous,indent=2)+'\n')
    for name in new:
        old=TARGET/name
        if old.exists():shutil.copy2(old,backup/old.name)
    for name,digest in new.items():
        dest=TARGET/name;temporary=dest.with_suffix(dest.suffix+'.gap.tmp')
        shutil.copy2(source/Path(name).name,temporary);os.replace(temporary,dest)
        if sha(dest)!=digest:raise RuntimeError('Installed processor differs: '+name)
        manifest[name]=digest
    temporary=manifest_path.with_suffix('.gap.tmp');temporary.write_text(json.dumps(manifest,indent=2)+'\n');os.replace(temporary,manifest_path)
    (BASE/'installation.json').write_text(json.dumps(dict(installed=True,started=False,rollback=str(backup),files=new,expiry_ms=[170,250,350],inference_changed=False),indent=2)+'\n')
    print('Installed native gap handling; inference untouched. Select Hands in FrameHands to start.')
if __name__=='__main__':main()
