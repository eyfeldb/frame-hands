"""Render fixed largest benefit/regression cases on both recorded camera images."""
import argparse,json,sys
from pathlib import Path
import cv2
import numpy as np
from probe_monocular_bridge import BASE,NAMES,RI,MI
sys.path.insert(0,str(BASE.parent/'exploratory'))
from frame_data import Recording,read_audit,load_calibration
from compare_replay import SEGMENTS

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('run','session','upstream'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--candidate',default='plane',choices=('plane','flow'));p.add_argument('--secondary',default='rigid',choices=('rigid','plane'))
 a=p.parse_args();sys.path.insert(0,str(a.upstream));cases=[json.loads(x) for x in (a.run/'cases.jsonl').read_text().splitlines()]
 selected=[(a.candidate+'-worst',max(cases,key=lambda x:x['methods'][a.candidate]['hidden_mean_px']-x['methods']['hold']['hidden_mean_px'])),(a.candidate+'-best',min(cases,key=lambda x:x['methods'][a.candidate]['hidden_mean_px']-x['methods']['hold']['hidden_mean_px'])),(a.secondary+'-worst',max(cases,key=lambda x:x['methods'][a.secondary]['hidden_mean_px']-x['methods']['hold']['hidden_mean_px']))]
 cameras=load_calibration(a.session/'frame/snapshot/device/persist');audit=read_audit(a.session/'frame/live-views.txt');recording=Recording(a.session/'frame/sets.bin');first=recording.index[0][0];review=[]
 for name,case in selected:
  seconds=(case['raw_ns']-first)/1e9;frames=list(recording.frames(seconds-.00001,seconds+.00001));assert len(frames)==1 and frames[0][1]==case['raw_ns'];images=frames[0][2]
  canvas=np.full((760,1280,3),24,np.uint8)
  cv2.putText(canvas,f"{name} | {seconds:.3f}s | side {case['side']} | {case['age_ms']:.1f}ms history",(15,28),cv2.FONT_HERSHEY_SIMPLEX,.65,(240,240,240),1,cv2.LINE_AA)
  for row,camera_name in enumerate(NAMES):
   camera=cameras[camera_name];image=images[camera_name][0]
   view=next(v for v in audit[case['raw_ns']]['views'] if v['id']==case['source'] and v['camera']==camera_name)
   center=(view['points'].min(0)+view['points'].max(0))*.5;extent=max(240,float(np.ptp(view['points'],axis=0).max()+100));extent=min(extent,min(image.shape));x=int(np.clip(center[0]-extent*.5,0,image.shape[1]-extent));y=int(np.clip(center[1]-extent*.5,0,image.shape[0]-extent));size=int(extent)
   for col,mode in enumerate(('image','hold',a.candidate,a.secondary)):
    picture=cv2.cvtColor(image[y:y+size,x:x+size],cv2.COLOR_GRAY2BGR)
    if mode!='image':
     points=np.array(case['methods'][mode]['head_mm']);uv=camera.eye_to_window(camera.world_to_eye(points))-np.array([x,y]);color={'hold':(240,220,40),'plane':(30,150,255),'rigid':(230,70,220),'flow':(70,230,60)}[mode]
     for i,j,_,_ in SEGMENTS:cv2.line(picture,tuple(np.rint(uv[i]).astype(int)),tuple(np.rint(uv[j]).astype(int)),color,2,cv2.LINE_AA)
     for point in uv[RI]:cv2.circle(picture,tuple(np.rint(point).astype(int)),2,color,-1,cv2.LINE_AA)
    canvas[70+row*345:390+row*345,col*320:(col+1)*320]=cv2.resize(picture,(320,320))
    used='OBSERVED' if camera_name==case['view'] else 'WITHHELD'
    cv2.putText(canvas,f'{used} / {mode}',(col*320+5,60+row*345),cv2.FONT_HERSHEY_SIMPLEX,.5,(230,230,230),1,cv2.LINE_AA)
  cv2.imwrite(str(a.run/(name+'.png')),canvas)
  review.append(dict(name=name,seconds=seconds,case={k:v for k,v in case.items() if k!='methods'},metrics={key:{k:v for k,v in value.items() if k!='head_mm'} for key,value in case['methods'].items()}))
 (a.run/'review.json').write_text(json.dumps(review,indent=2)+'\n')
 print(json.dumps([dict(name=x['name'],seconds=x['seconds']) for x in review]))
if __name__=='__main__':main()
