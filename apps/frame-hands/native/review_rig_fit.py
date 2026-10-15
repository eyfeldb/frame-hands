"""Render exact saved-camera frames beside the experimental articulated fit."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import cv2
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE.parent/'exploratory'))
from frame_data import Recording,read_audit,load_calibration
from audit_native_lengths import PARENTS,sha
from compare_replay import SEGMENTS

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('session','upstream','run','output'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--poses-name',default='poses.jsonl');p.add_argument('--selection',type=Path)
 a=p.parse_args();sys.path.insert(0,str(a.upstream));a.output.mkdir(exist_ok=False)
 rows=[json.loads(x) for x in (a.run/a.poses_name).read_text().splitlines()];valid=[r for r in rows if r['tip_reprojection_change_px'] is not None]
 ordered=sorted(valid,key=lambda r:r['tip_reprojection_change_px']);selected=[ordered[int((len(ordered)-1)*q)] for q in (.25,.5,.95)]
 if a.selection:
  wanted=json.loads(a.selection.read_text())['cases'];lookup={(r['raw_ns'],r['source_id']):r for r in rows};selected=[lookup[(r['raw_ns'],r['source_id'])] for r in wanted]
  if len(selected)!=3:raise ValueError('Expected three fixed review cases')
 recording=Recording(a.session/'frame/sets.bin');first=recording.index[0][0];audit=read_audit(a.session/'frame/live-views.txt');cams=load_calibration(a.session/'frame/snapshot/device/persist')
 canvas=np.zeros((1200,1500,3),np.uint8);meta=[]
 for n,row in enumerate(selected):
  stamp=row['raw_ns'];seconds=(stamp-first)/1e9;frames=[x for x in recording.frames(seconds-1e-6,seconds+1e-6) if x[1]==stamp]
  if len(frames)!=1:raise ValueError('No exact camera frame')
  views=[v for v in audit[stamp]['views'] if v['id']==row['source_id']];view=next(v for v in views if v['camera']=='slam_left');image,capture=frames[0][2][view['camera']];cam=cams[view['camera']]
  raw=view['points'];fit=cam.eye_to_window(cam.world_to_eye(np.array(row['fit_rig_head_mm'])))
  low=np.minimum(raw.min(0),fit.min(0))-30;high=np.maximum(raw.max(0),fit.max(0))+30
  low=np.maximum(np.floor(low).astype(int),[0,0]);high=np.minimum(np.ceil(high).astype(int),[image.shape[1],image.shape[0]])
  def pixel(v):return tuple(np.rint(v).astype(int))
  for col,label in enumerate(('Camera only','Detector landmarks','Fixed rig fit')):
   panel=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
   if col==1:
    for j in range(1,21):cv2.line(panel,pixel(raw[PARENTS[j]]),pixel(raw[j]),(255,255,0),1,cv2.LINE_AA)
    for xy in raw:cv2.circle(panel,pixel(xy),2,(255,255,0),-1)
   if col==2:
    for x,y,_,_ in SEGMENTS:cv2.line(panel,pixel(fit[x]),pixel(fit[y]),(255,0,255),1,cv2.LINE_AA)
    for j in [6,8,11,14,17]:cv2.line(panel,pixel(fit[5]),pixel(fit[j]),(255,0,255),1,cv2.LINE_AA)
    for xy in fit[:20]:cv2.circle(panel,pixel(xy),2,(255,0,255),-1)
   crop=panel[low[1]:high[1],low[0]:high[0]];scale=min(480/crop.shape[1],340/crop.shape[0]);tile=cv2.resize(crop,None,fx=scale,fy=scale)
   left=col*500+(500-tile.shape[1])//2;top=n*400+55
   canvas[top:top+tile.shape[0],left:left+tile.shape[1]]=tile
   cv2.putText(canvas,f'{seconds:.3f}s ID{row["source_id"]} side{row["side"]}: {label}',(col*500+10,n*400+20),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1)
   cv2.putText(canvas,f'Tip reprojection change {row["tip_reprojection_change_px"]:.1f}px',(col*500+10,n*400+43),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
  meta.append(dict(raw_ns=stamp,camera_capture_ns=capture,source_id=row['source_id'],side=row['side'],seconds=seconds,tip_reprojection_change_px=row['tip_reprojection_change_px']))
 if not cv2.imwrite(str(a.output/'review.png'),canvas):raise ValueError('Image write failed')
 (a.output/'selection.json').write_text(json.dumps(dict(cases=meta,sets_sha256=sha(a.session/'frame/sets.bin'),poses_sha256=sha(a.run/a.poses_name),source_sha256=sha(Path(__file__))),indent=2)+'\n')
if __name__=='__main__':main()
