"""Render paired original-detector fit outputs on exact saved stereo camera frames."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import cv2
from audit_native_lengths import sha
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE.parent/'exploratory'))
from frame_data import Recording,load_calibration,read_audit
from probe_overlap_acquisition import affine
from compare_replay import SEGMENTS

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('audit','session','upstream'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument("--crop-trace",type=Path)
 p.add_argument("--baseline-failures",action="store_true",help="Review worst control wrist/root steps and fit residual when the new mechanism is inactive")
 p.add_argument("--skip-reference-frame",action="store_true",help="Omit first-recording-specific38.067s diagnostic on other recordings")
 a=p.parse_args();sys.path.insert(0,str(a.upstream))
 crop_rows={};observations=None
 if a.crop_trace:
  evidence=json.loads((a.crop_trace/"result.json").read_text());assert evidence["outputs_match"] and sha(a.crop_trace/"crops.txt")==evidence["trace_sha256"]
  observations=read_audit(a.crop_trace/"views.txt");stamp=None
  for line in (a.crop_trace/"crops.txt").read_text().splitlines():
   x=line.split()
   if x[0]=="set":stamp=int(x[1]);crop_rows[stamp]=[]
   else:assert len(x)==51;crop_rows[stamp].append((x[1],np.array(x[3:7],float),np.array(x[9:],float).reshape(21,2)))
 summary=json.loads((a.audit/'summary.json').read_text(encoding='utf-8'));same_objective=summary['metrics'].get('objectives_comparable',True);same_correspondences=summary['metrics'].get('pixel_correspondences_comparable',True)
 records=json.loads((a.audit/'records.json').read_text(encoding='utf-8'));steps=json.loads((a.audit/'steps.json').read_text(encoding='utf-8'))
 rec=Recording(a.session/'frame/sets.bin');cams=load_calibration(a.session/'frame/snapshot/device/persist');origin=rec.index[0][0]
 improvement=max(steps,key=lambda r:r['wrist_degrees'][0]-r['wrist_degrees'][1]);regression=max(steps,key=lambda r:r['wrist_degrees'][1]-r['wrist_degrees'][0])
 selected=[('largest-jump-improvement',improvement['index']),('largest-jump-regression-before',regression['previous']),('largest-jump-regression',regression['index']),(('largest-cost-gain' if same_objective else 'largest-image-gain' if same_correspondences else 'largest-fit-residual-change'),max(records,key=lambda r:(r['cost'] if same_objective else r['rmse'])[0]-(r['cost'] if same_objective else r['rmse'])[1])['index']),('raised-finger-conflict',min((r for r in records if r['side']==0),key=lambda r:abs((r['raw_ns']-origin)/1e9-38.066665))['index'])]
 if a.skip_reference_frame:selected=selected[:-1]
 if a.baseline_failures:
  worst=max(steps,key=lambda r:r['wrist_degrees'][0]);root=max(steps,key=lambda r:r['root_mm'][0])
  selected=[('worst-wrist-before',worst['previous']),('worst-wrist-after',worst['index']),('worst-root-step',root['index']),('worst-fit-residual',max(records,key=lambda r:r['rmse'][0])['index'])]
 review=[]
 for label,index in selected:
  r=records[index];sec=(r['raw_ns']-origin)/1e9;frames=list(rec.frames(sec-1e-7,sec+1e-7));assert len(frames)==1 and frames[0][1]==r['raw_ns'];points=np.array(r['points'])
  canvas=np.full((750,960,3),24,np.uint8);cv2.putText(canvas,f"{label} / {sec:.3f}s side{r['side']} source{r['source']}",(8,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(240,240,240),1,cv2.LINE_AA)
  cv2.putText(canvas,f"{'image RMSE' if same_correspondences else 'fit RMSE (different labels)'} {r['rmse'][0]:.2f} -> {r['rmse'][1]:.2f}px",(8,47),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,240,240),1,cv2.LINE_AA)
  for v,name in enumerate(('slam_left','slam_right')):
   image=frames[0][2][name][0];cam=cams[name];uv=[cam.eye_to_window(cam.world_to_eye(p)) for p in points];alluv=np.vstack(uv);center=(alluv.min(0)+alluv.max(0))*.5;size=int(min(min(image.shape),max(300,np.ptp(alluv,axis=0).max()+100)));x=int(np.clip(center[0]-size*.5,0,image.shape[1]-size));y=int(np.clip(center[1]-size*.5,0,image.shape[0]-size))
   display=image[y:y+size,x:x+size];display_uv=[p-[x,y] for p in uv]
   if observations is not None:
    seen=[v for v in observations[r['raw_ns']]['views'] if v['camera']==name and v['id']==r['source']];assert len(seen)==1
    matches=[row for row in crop_rows[r['raw_ns']] if row[0]==name and np.max(abs(row[2]-seen[0]['points']))<1e-6];assert len(matches)==1
    roi=matches[0][1];matrix=affine(roi[:2],roi[2]*1.3,roi[3],300);inverse=cv2.invertAffineTransform(matrix)
    display=cv2.warpAffine(image,matrix,(300,300),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
    display_uv=[np.c_[p,np.ones(len(p))]@inverse.T for p in uv]
   for col in range(3):
    pic=cv2.cvtColor(display,cv2.COLOR_GRAY2BGR)
    if col:
     pp=display_uv[col-1]
     for pi,qi,_,_ in SEGMENTS:cv2.line(pic,tuple(np.rint(pp[pi]).astype(int)),tuple(np.rint(pp[qi]).astype(int)),((240,220,40),(70,230,60))[col-1],2,cv2.LINE_AA)
    canvas[90+v*330:390+v*330,col*320:col*320+300]=cv2.resize(pic,(300,300));cv2.putText(canvas,name+' / '+('image','control','candidate')[col],(col*320+3,77+v*330),cv2.FONT_HERSHEY_SIMPLEX,.45,(240,240,240),1,cv2.LINE_AA)
  path=a.audit/(label+'.png');assert not path.exists();assert cv2.imwrite(str(path),canvas)
  review.append(dict(label=label,index=index,raw_ns=r['raw_ns'],seconds=sec,side=r['side'],source=r['source'],rmse=r['rmse'],cost=r['cost'],image_sha256=sha(path)))
 (a.audit/'review.json').write_text(json.dumps(dict(review=review,renderer_sha256=sha(Path(__file__)),display="upright traced crop expanded1.3 for review only; OpenCV sampling" if a.crop_trace else "camera-aligned bounding crop"),indent=2)+'\n',encoding='utf-8');print(json.dumps(review,indent=2))
if __name__=='__main__':main()
