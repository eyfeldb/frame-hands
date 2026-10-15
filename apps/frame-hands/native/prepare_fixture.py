"""Prepare a causal prefix of a saved accepted trial for native replay. No camera access."""
import argparse,json,shutil,struct,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'exploratory'))
from replay_mesh_trial import packets

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--frames',type=int,default=350);a=p.parse_args()
 if a.frames<1:p.error('positive frame count required')
 a.output.mkdir(parents=True,exist_ok=False);shutil.copytree(a.session/'calibration',a.output/'calibration');refs={}
 with (a.output/'reference.jsonl').open('w') as out:
  for line in (a.session/'predictions.jsonl').open():
   row=json.loads(line)
   if row['sequence']>a.frames:break
   if row['capture_raw_ns'] in refs:raise ValueError('Duplicate source timestamp')
   refs[row['capture_raw_ns']]=row;out.write(line)
 if not refs:raise ValueError('Empty reference')
 last=max(refs);seen=set()
 with (a.output/'transport.bin').open('wb') as out:
  for meta,blobs in packets(a.session/'transport.bin'):
   if not meta.get('waiting') and meta['stamp']>last:break
   if not meta.get('waiting'):seen.add(meta['stamp'])
   raw=json.dumps(meta).encode();out.write(struct.pack('<I',len(raw))+raw+b''.join(blobs))
 if not set(refs)<=seen:raise ValueError('Reference lacks matching crop packet')
 (a.output/'fixture.json').write_text(json.dumps({'session':a.session.name,'frames':len(refs),'scope':'Continuous prefix including startup; not selected hard-gesture coverage.'},indent=2)+'\n')
if __name__=='__main__':main()
