"""Review accepted fresh landmarks, fitted rig and the exact traced crop geometry.

Crop previews use OpenCV sampling, not byte-identical ncnn sampling; they diagnose
coverage/orientation only. Source images and learned points are exact saved records.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
import cv2
from audit_native_lengths import sha
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE.parent/'exploratory'))
from frame_data import read_audit,load_calibration,Recording
from compare_replay import SEGMENTS
from probe_overlap_acquisition import affine
MP_SEGMENTS=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('trace','reference','session','upstream','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();sys.path.insert(0,str(a.upstream));a.output.mkdir(exist_ok=False,parents=True)
 evidence=json.loads((a.trace/'result.json').read_text(encoding='utf-8'));assert evidence['outputs_match'] and sha(a.trace/'crops.txt')==evidence['trace_sha256']
 records=json.loads((a.reference/'records.json').read_text(encoding='utf-8'));selected=json.loads((a.reference/'review.json').read_text(encoding='utf-8'))['review'];views=read_audit(a.trace/'views.txt')
 crops={};raw=None
 for line in (a.trace/'crops.txt').read_text(encoding='utf-8').splitlines():
  row=line.split()
  if row[0]=='set':raw=int(row[1]);assert raw not in crops;crops[raw]=[]
  else:
   assert row[0]=='crop' and len(row)==51 and raw is not None
   values=np.array(row[3:],float);assert np.isfinite(values).all();crops[raw].append(dict(camera=row[1],source_before_assignment=int(row[2]),roi=values[:4],presence=values[4],right=values[5],points=values[6:].reshape(21,2)))
 rec=Recording(a.session/'frame/sets.bin');cams=load_calibration(a.session/'frame/snapshot/device/persist');result=[]
 for item in selected:
  r=records[item['index']];raw=r['raw_ns'];sec=item['seconds'];frames=list(rec.frames(sec-1e-7,sec+1e-7));assert len(frames)==1 and frames[0][1]==raw
  canvas=np.full((750,1280,3),24,np.uint8);cv2.putText(canvas,f"{item['label']} / {sec:.3f}s side{r['side']}",(8,23),cv2.FONT_HERSHEY_SIMPLEX,.6,(240,240,240),1,cv2.LINE_AA);metadata=[]
  for v,name in enumerate(('slam_left','slam_right')):
   matches=[x for x in views[raw]['views'] if x['id']==r['source'] and x['camera']==name];assert len(matches)==1;observed=matches[0];det=observed['points']
   matches=[x for x in crops[raw] if x['camera']==name and np.max(abs(x['points']-det))<1e-6];assert len(matches)==1
   native=matches[0];center=native['roi'][:2];size,rotation=native['roi'][2:];matrix=affine(center,size,rotation,224);inverse=cv2.invertAffineTransform(matrix);in_crop=np.column_stack([det,np.ones(21)])@inverse.T
   image=frames[0][2][name][0];cam=cams[name];rig=cam.eye_to_window(cam.world_to_eye(np.array(r['points'][1])));alluv=np.vstack([det,rig]);viewcenter=(alluv.min(0)+alluv.max(0))*.5;viewsize=int(min(min(image.shape),max(220,np.ptp(alluv,axis=0).max()+100)));x=int(np.clip(viewcenter[0]-viewsize*.5,0,image.shape[1]-viewsize));y=int(np.clip(viewcenter[1]-viewsize*.5,0,image.shape[0]-viewsize))
   patch=cv2.warpAffine(image,matrix,(224,224),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
   for col in range(4):
    pic=cv2.cvtColor(patch if col==3 else image[y:y+viewsize,x:x+viewsize],cv2.COLOR_GRAY2BGR)
    if col:
     pp=in_crop if col==3 else (det if col==1 else rig)-[x,y];edges=MP_SEGMENTS if col in (1,3) else [(s[0],s[1]) for s in SEGMENTS]
     for pi,qi in edges:cv2.line(pic,tuple(np.rint(pp[pi]).astype(int)),tuple(np.rint(pp[qi]).astype(int)),(230,170,60) if col in (1,3) else (60,230,70),1 if col==3 else 2,cv2.LINE_AA)
    canvas[65+v*340:365+v*340,col*320:col*320+300]=cv2.resize(pic,(300,300));label=('image',f"presence {observed['presence']:.3f}",'whole-pose fitted','traced crop / CV sampler')[col];cv2.putText(canvas,label,(col*320+3,53+v*340),cv2.FONT_HERSHEY_SIMPLEX,.44,(240,240,240),1,cv2.LINE_AA)
   cv2.imwrite(str(a.output/(item['label']+'-'+name+'-crop.png')),patch)
   metadata.append(dict(camera=name,roi=native['roi'].tolist(),source_before_assignment=native['source_before_assignment'],presence=observed['presence'],right=observed['right'],max_trace_point_difference=float(np.max(abs(native['points']-det))),predicted_points_outside_crop=int(np.any((in_crop<0)|(in_crop>223),axis=1).sum())))
  path=a.output/(item['label']+'.png');assert cv2.imwrite(str(path),canvas);result.append(dict(**item,crops=metadata,landmark_review_sha256=sha(path)))
 report=dict(scope=__doc__,review=result,trace_result_sha256=sha(a.trace/'result.json'),reviewer_sha256=sha(Path(__file__)),opencv=cv2.__version__,limit='Crop completeness is qualitative. Detector points are hypotheses, not truth. Presence is a hand-level score and cannot certify fingers.')
 (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
