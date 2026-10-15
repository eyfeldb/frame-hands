"""Review exact raw images for real monocular-flow cases; labels are not truth."""
import argparse,json,sys
from pathlib import Path
import cv2
import numpy as np
from probe_monocular_bridge import BASE,NAMES,RI
from frame_data import Recording,read_audit,load_calibration
from compare_replay import SEGMENTS


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('run','session','upstream'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();sys.path.insert(0,str(a.upstream))
 rows=[json.loads(x) for x in (a.run/'cases.jsonl').read_text().splitlines()]
 mono=[r for r in rows if r['kind']=='mono'];supported=[r for r in mono if r['fit']['accepted']]
 endpoints=[r for r in rows if r['kind']=='reacquired' and r['fit']['accepted']]
 rejected=[r for r in mono if not r['fit']['accepted']]
 selected=[('first-supported',supported[0]),('largest-step',max(supported,key=lambda r:r['step_mm'])),
           ('largest-observed-error',max(supported,key=lambda r:r['methods']['flow']['observed_error_px'])),
           ('rejected-largest-error',max(rejected,key=lambda r:r['methods']['hold']['observed_error_px'])),
           ('endpoint-best',min(endpoints,key=lambda r:r['methods']['flow']['withheld_error_px']-r['methods']['hold']['withheld_error_px'])),
           ('endpoint-worst',max(endpoints,key=lambda r:r['methods']['flow']['withheld_error_px']-r['methods']['hold']['withheld_error_px']))]
 cameras=load_calibration(a.session/'frame/snapshot/device/persist');audit=read_audit(a.session/'frame/live-views.txt');recording=Recording(a.session/'frame/sets.bin');first=recording.index[0][0]
 def images(stamp):
  t=(stamp-first)/1e9;frames=list(recording.frames(t-1e-7,t+1e-7));assert len(frames)==1 and frames[0][1]==stamp;return frames[0][2]
 review=[]
 for label,case in selected:
  current=images(case['raw_ns']);prior=images(case['prior_raw_ns'])
  canvas=np.full((760,1280,3),24,np.uint8)
  title=f"{label} | {case['seconds']:.3f}s | side {case['side']} | {case['age_ms']:.1f}ms | {case['step_mm']:.1f}mm"
  cv2.putText(canvas,title,(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(240,240,240),1,cv2.LINE_AA)
  for row,name in enumerate(NAMES):
   camera=cameras[name];image=current[name][0]
   projections={m:camera.eye_to_window(camera.world_to_eye(np.array(case['methods'][m]['head_mm']))) for m in ('hold','flow')}
   old=next(v for v in audit[case['prior_raw_ns']]['views'] if v['id']==case['source'] and v['camera']==name)
   views=[v for v in audit[case['raw_ns']]['views'] if v['id']==case['source'] and v['camera']==name]
   combined=np.vstack([projections['hold'][RI],projections['flow'][RI],old['points'],*[v['points'] for v in views]])
   center=(combined.min(0)+combined.max(0))*.5;size=int(min(min(image.shape),max(350,np.ptp(combined,axis=0).max()+100)))
   x=int(np.clip(center[0]-size*.5,0,image.shape[1]-size));y=int(np.clip(center[1]-size*.5,0,image.shape[0]-size))
   for col,mode in enumerate(('prior','image','hold','flow')):
    source=prior[name][0] if mode=='prior' else image
    picture=cv2.cvtColor(source[y:y+size,x:x+size],cv2.COLOR_GRAY2BGR)
    if mode in projections:
     uv=projections[mode]-[x,y];color=(240,220,40) if mode=='hold' else (70,230,60)
     for i,j,_,_ in SEGMENTS:cv2.line(picture,tuple(np.rint(uv[i]).astype(int)),tuple(np.rint(uv[j]).astype(int)),color,2,cv2.LINE_AA)
     for point in uv[RI]:cv2.circle(picture,tuple(np.rint(point).astype(int)),2,color,-1,cv2.LINE_AA)
    if mode=='prior':
     for point in old['points']-[x,y]:cv2.circle(picture,tuple(np.rint(point).astype(int)),2,(255,180,50),-1)
    canvas[70+row*345:390+row*345,col*320:(col+1)*320]=cv2.resize(picture,(320,320))
    role='SEED STEREO' if mode=='prior' else ('OBSERVED' if name==case['view'] else ('REACQUIRED' if views else 'NO DETECTION'))
    cv2.putText(canvas,f'{role} / {mode}',(col*320+5,60+row*345),cv2.FONT_HERSHEY_SIMPLEX,.47,(230,230,230),1,cv2.LINE_AA)
  cv2.imwrite(str(a.run/(label+'.png')),canvas)
  review.append(dict(label=label,case=case))
 (a.run/'review.json').write_text(json.dumps(review,indent=2)+'\n')
 print(json.dumps([dict(label=r['label'],seconds=r['case']['seconds']) for r in review]))


if __name__=='__main__':main()
