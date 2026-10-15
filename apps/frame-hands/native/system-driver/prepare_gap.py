"""Port the existing PC gap policy under distinct resource names; keep accepted native assets."""
import hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
SOURCE=BASE.parents[1]/'exploratory/mesh-viewer'
NAMES={'hand_motion.gd':'native_gap_base.gd','continuity_motion.gd':'native_gap_continuity.gd','gap_motion.gd':'native_gap_motion.gd','game_motion.gd':'native_game_motion.gd'}
def prepare(output):
    output.mkdir(parents=True,exist_ok=True);receipt={}
    for source,dest in NAMES.items():
        raw=(SOURCE/source).read_text(encoding='utf-8');text=raw.replace('class_name HandMotion\n','')
        for old,new in NAMES.items():text=text.replace('res://'+old,'res://'+new)
        (output/dest).write_text(text,encoding='utf-8',newline='\n')
        receipt[dest]={'source':source,'source_sha256':hashlib.sha256((SOURCE/source).read_bytes()).hexdigest(),'sha256':hashlib.sha256((output/dest).read_bytes()).hexdigest()}
    (output/'gap-policy.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    return receipt
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();prepare(a.output)
