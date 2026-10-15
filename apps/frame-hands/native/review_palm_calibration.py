"""Exact saved-image review of every proposed palm-calibration observation."""
import argparse,json,sys
from pathlib import Path
import numpy as np,cv2
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE.parent/'exploratory'))
from frame_data import Recording,read_audit
from audit_native_lengths import sha

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('session','calibration','output'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);meta=json.loads((a.calibration/'summary.json').read_text());audit=read_audit(a.session/'frame/live-views.txt');recording=Recording(a.session/'frame/sets.bin');first=recording.index[0][0];sheets=[]
 for side,rows in meta['selection'].items():
  for page,start in enumerate(range(0,len(rows),4)):
   part=rows[start:start+4];canvas=np.zeros((len(part)*320,1000,3),np.uint8)
   for r,item in enumerate(part):
    seconds=(item['raw_ns']-first)/1e9;frames=[x for x in recording.frames(seconds-1e-6,seconds+1e-6) if x[1]==item['raw_ns']]
    if len(frames)!=1:raise ValueError('Missing exact calibration frame')
    for c,name in enumerate(('slam_left','slam_right')):
     view=next(v for v in audit[item['raw_ns']]['views'] if v['id']==item['source_id'] and v['camera']==name);image,capture=frames[0][2][name];uv=view['points'];lo=np.maximum(np.floor(uv.min(0)-25).astype(int),0);hi=np.minimum(np.ceil(uv.max(0)+25).astype(int),[image.shape[1],image.shape[0]]);panel=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
     for index in (0,5,9,13,17):cv2.circle(panel,tuple(np.rint(uv[index]).astype(int)),2,(0,255,255),1)
     crop=panel[lo[1]:hi[1],lo[0]:hi[0]];scale=min(480/crop.shape[1],275/crop.shape[0]);tile=cv2.resize(crop,None,fx=scale,fy=scale);x=c*500+(500-tile.shape[1])//2;y=r*320+37;canvas[y:y+tile.shape[0],x:x+tile.shape[1]]=tile
     cv2.putText(canvas,f'{item["seconds"]:.3f}s side{side} ID{item["source_id"]} {name}',(c*500+8,r*320+23),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
   path=a.output/f'side{side}-{page}.png'
   if not cv2.imwrite(str(path),canvas):raise ValueError('Review image write failed')
   sheets.append(dict(path=path.name,side=int(side),raw_ns=[r['raw_ns'] for r in part]))
 (a.output/'manifest.json').write_text(json.dumps(dict(sheets=sheets,calibration_summary_sha256=sha(a.calibration/'summary.json'),sets_sha256=sha(a.session/'frame/sets.bin'),source_sha256=sha(Path(__file__))),indent=2)+'\n')
if __name__=='__main__':main()
