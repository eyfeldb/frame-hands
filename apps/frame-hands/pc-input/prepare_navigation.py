"""Package an opt-in navigation candidate; never register a driver or restart VR."""
import argparse,json,shutil,hashlib
from pathlib import Path
BASE=Path(__file__).resolve().parent

def prepare(out,binary):
    if out.exists():raise FileExistsError(out)
    shutil.copytree(BASE/'driver/fh_skeletal',out,ignore=shutil.ignore_patterns('bin'))
    folder=out/'resources/input'
    profile=json.loads((folder/'fh_skeletal_profile.json').read_text())
    profile['input_source']['/input/navigation']={'type':'button','click':True}
    (folder/'fh_skeletal_profile.json').write_text(json.dumps(profile,indent=2)+'\n')
    dashboard=json.loads((folder/'dashboard.json').read_text())
    dashboard['description']='Right-hand pointer; left index-thumb pinch clicks. Dashboard-only candidate.'
    dashboard['bindings']['/actions/lasermouse']={'poses':[{'path':'/user/hand/right/pose/raw','output':'/actions/lasermouse/in/Pointer'}], 'sources':[{'path':'/user/hand/right/input/navigation','mode':'button','inputs':{'click':{'output':'/actions/lasermouse/in/leftclick'}}}]}
    (folder/'dashboard.json').write_text(json.dumps(dashboard,indent=2)+'\n')
    arch='win64' if binary.suffix=='.dll' else 'linuxarm64'
    dest=out/'bin'/arch;dest.mkdir(parents=True);shutil.copy2(binary,dest/binary.name)
    receipt={'candidate':True,'registered':False,'started':False,'driver_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'requires':'navigation-context writes fresh dashboard.bin beside input.bin; live dashboard click acceptance pending','models':'No hand-model display change'}
    (out/'navigation-candidate.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--binary',type=Path,required=True);a=p.parse_args();prepare(a.output,a.binary)
