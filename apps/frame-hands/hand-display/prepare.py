"""Stage display/navigation driver resources. Never register, launch or change mode."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'pc-input'))
from prepare_navigation import prepare

def package(out,binary):
    prepare(out,binary)
    dest=out/'resources/rendermodels/fh_display_empty'
    dest.mkdir(parents=True)
    source=BASE.parent/'system-input/driver/fh_pointer/resources/rendermodels/ft_pointer_invisible'
    for p in source.iterdir():
        target=dest/p.name.replace('ft_pointer_invisible','fh_display_empty')
        if p.suffix=='.png':shutil.copy2(p,target)
        else:target.write_text(p.read_text().replace('ft_pointer_invisible','fh_display_empty'),encoding='utf-8')
    (out/'navigation-candidate.json').unlink()
    files={p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
    return files
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--binary',type=Path,required=True);a=p.parse_args()
    files=package(a.output,a.binary)
    (a.output.parent/'display-deployment.json').write_text(json.dumps(dict(bundle=str(a.output),files=files,input_mode_supported=True,skeleton_only=True,registered=False,live_validated=False),indent=2)+'\n')
