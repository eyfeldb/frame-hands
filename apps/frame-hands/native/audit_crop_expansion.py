"""Audit one fixed wider-crop probe; diagnostic only, no tracking policy selection."""
import argparse,json,sys
from pathlib import Path
import numpy as np,cv2
from audit_native_lengths import sha
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE.parent/'exploratory'))
from frame_data import read_audit,Recording,load_calibration,unproject
from probe_overlap_acquisition import affine
from review_native_landmarks import MP_SEGMENTS

def epi(cameras,points):
 names=('slam_left','slam_right');origins=[cameras[n].camera_to_world_xf[:3,3] for n in names];rays=[unproject(cameras[n],p)@cameras[n].camera_to_world_xf[:3,:3].T for n,p in zip(names,points)];base=origins[1]-origins[0]
 normals=[np.cross(base,r) for r in rays];norms=[np.linalg.norm(n,axis=1) for n in normals];assert min(n.min() for n in norms)>1e-6
 angles=[np.arcsin(np.clip(np.abs(np.sum(rays[1-i]*normals[i],axis=1))/norms[i],0,1)) for i in (0,1)]
 return np.rad2deg(np.sqrt((angles[0]**2+angles[1]**2)/2))

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('run','trace','session','upstream'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();sys.path.insert(0,str(a.upstream));requests=json.loads((a.run/'requests.json').read_text());z=np.loadtxt(a.run/'output-v2.txt');assert z.shape==(20,46) and np.isfinite(z).all();audit=read_audit(a.trace/'views.txt');rec=Recording(a.session/'frame/sets.bin');cams=load_calibration(a.session/'frame/snapshot/device/persist');reports=[];maxdiff=0
 for start in range(0,len(requests),2):
  pair=requests[start:start+2];raw=pair[0]['raw'];sec=(raw-rec.index[0][0])/1e9;frames=list(rec.frames(sec-1e-7,sec+1e-7));assert len(frames)==1 and frames[0][1]==raw;canvas=np.full((740,960,3),24,np.uint8);cv2.putText(canvas,f"{pair[0]['case']} / {sec:.3f}s / blue original, green wider",(8,23),cv2.FONT_HERSHEY_SIMPLEX,.52,(240,240,240),1,cv2.LINE_AA);points=[[],[]];presences=[[],[]]
  for k,r in enumerate(pair):
   entries=z[z[:,0]==start+k];assert entries.shape==(2,46) and np.array_equal(entries[:,1],[1.,1.3]);lm=entries[:,4:].reshape(2,21,2);v=next(v for v in audit[raw]['views'] if v['camera']==r['camera'] and v['id']==r['source']);difference=np.max(abs(lm[0]-v['points']));assert difference<1e-6;maxdiff=max(maxdiff,float(difference))
   image=frames[0][2][r['camera']][0];roi=r['roi'];matrix=affine(roi[:2],roi[2]*1.3,roi[3],224);inv=cv2.invertAffineTransform(matrix);patch=cv2.warpAffine(image,matrix,(224,224),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
   for col in range(3):
    pic=cv2.cvtColor(patch,cv2.COLOR_GRAY2BGR)
    if col:
     projected=np.c_[lm[col-1],np.ones(21)]@inv.T
     for pi,qi in MP_SEGMENTS:cv2.line(pic,tuple(np.rint(projected[pi]).astype(int)),tuple(np.rint(projected[qi]).astype(int)),(230,170,60) if col==1 else (60,230,70),1,cv2.LINE_AA)
     for finger,tip in enumerate((8,12,16,20)):cv2.putText(pic,('I','M','R','L')[finger],tuple(np.rint(projected[tip]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1,cv2.LINE_AA)
    canvas[65+k*340:365+k*340,col*320:col*320+300]=cv2.resize(pic,(300,300));label=('image',f"original {entries[0,2]:.3f}",f"wider {entries[1,2]:.3f}")[col];cv2.putText(canvas,label,(col*320+3,53+k*340),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,240,240),1,cv2.LINE_AA)
   for i in (0,1):points[i].append(lm[i]);presences[i].append(float(entries[i,2]))
  errors=[epi(cams,p) for p in points];swapped=[]
  for p in points:
   candidate=p[1].copy();candidate[5:9],candidate[9:13]=p[1][9:13],p[1][5:9];swapped.append(epi(cams,[p[0],candidate]))
  path=a.run/(pair[0]['case']+'.png');assert cv2.imwrite(str(path),canvas)
  reports.append(dict(case=pair[0]['case'],raw_ns=raw,seconds=sec,presence=presences,index_middle_epipolar_rms_degrees=[float(np.sqrt(np.mean(e[5:13]**2))) for e in errors],index_middle_swapped_rms_degrees=[float(np.sqrt(np.mean(e[5:13]**2))) for e in swapped],image_sha256=sha(path)))
 result=dict(scope=__doc__,cases=reports,exact_original_point_max_difference=maxdiff,output_sha256=sha(a.run/'output-v2.txt'),source_sha256=sha(Path(__file__)),limitations=['Ten preselected failure views only. No accuracy, coverage, speed or native-Frame claim.','Same-image enlarged ROI; presence scores and epipolar agreement cannot identify true anatomical labels.','Index/middle swap is diagnostic only, never applied to output. No history, gesture or presence policy changed.','Display crops use OpenCV sampling; inference uses the original C++ ncnn sampler and weights.'])
 (a.run/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
