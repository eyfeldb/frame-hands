"""Show both published sides together when auditing side transfers and recovery."""
import argparse,json,sys,subprocess
from pathlib import Path
import numpy as np,cv2
from scipy.spatial.transform import Rotation
BASE=Path(__file__).resolve().parents[1];sys.path[:0]=[str(BASE),str(BASE.parent/'exploratory')]
from frame_data import Recording,load_calibration
from compare_replay import SEGMENTS
from run_tracker_rig_check import sha

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('run','session','upstream','profiles','evaluator'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();sys.path.insert(0,str(a.upstream));out=a.run/'review-both';out.mkdir(exist_ok=False)
 data=json.loads((a.run/'first.json').read_text());rows=[json.loads(x) for x in (a.run/'first-result.json.frames.jsonl').read_text().splitlines()];events=data['events'];times=np.array([e['capture'] for e in events]);headroot=Path(data['provenance']['baseline']['head_evidence']['path'])
 heads={int(x[0]):(int(x[1]),int(x[2]),np.array(x[3:],float)) for x in [line.split() for line in (headroot/'heads.txt').read_text().splitlines()[1:]]}
 rec=Recording(a.session/'frame/sets.bin');cams=load_calibration(a.session/'frame/snapshot/device/persist');origin=rec.index[0][0];report=[]
 for target in (25.5888888888889,48.9222222222222):
  index=min(range(0,len(rows),2),key=lambda i:abs(rows[i]['now']-target));pair=rows[index:index+2];now=pair[0]['now'];event=events[int(np.argmin(abs(times-now)))];raw=event['raw_ns'];_,valid,h=heads[raw];assert valid;hr=Rotation.from_quat(h[:4][[1,2,3,0]]);points=[[],[]];meta=[[],[]]
  for variant in (0,1):
   for row in pair:
    v=row['values'][variant];side=row['side']
    if not v:continue
    q=hr.inv()*Rotation.from_quat(np.array(v['orientation'])[[1,2,3,0]]);t=hr.inv().apply(np.array(v['position'])*1000-h[4:]);pose=[*q.as_quat()[[3,0,1,2]],*t,*v['angles']];stem=f'{now:.3f}-{variant}-{side}';inp=out/(stem+'.txt');result=out/(stem+'-points.txt')
    inp.write_text(str(side)+' '+' '.join(format(x,'.17g') for x in pose)+'\n');subprocess.run([str(a.evaluator.resolve()),str((a.profiles/f'side{side}.fhrig').resolve()),str(inp.resolve()),str(result.resolve())],check=True,capture_output=True,text=True)
    pts=np.loadtxt(result)[1:].reshape(21,3);points[variant].append((side,v['id'],pts));meta[variant].append(dict(side=side,id=v['id']))
  seconds=(raw-origin)/1e9;frame=list(rec.frames(seconds-1e-7,seconds+1e-7))[0];assert frame[1]==raw;canvas=np.full((750,960,3),24,np.uint8);delta=(event['capture']-now)*1000
  cv2.putText(canvas,f'Both sides {now:.3f}s / blue=left orange=right / camera {delta:+.2f}ms',(5,25),cv2.FONT_HERSHEY_SIMPLEX,.52,(240,240,240),1,cv2.LINE_AA)
  for camidx,name in enumerate(('slam_left','slam_right')):
   im=frame[2][name][0];cam=cams[name];uv=[[(side,ident,cam.eye_to_window(cam.world_to_eye(pts))) for side,ident,pts in variant] for variant in points];alluv=np.vstack([v[2] for variant in uv for v in variant]);center=(alluv.min(0)+alluv.max(0))*.5;size=int(min(min(im.shape),max(480,np.ptp(alluv,axis=0).max()+220)));x=int(np.clip(center[0]-size*.5,0,im.shape[1]-size));y=int(np.clip(center[1]-size*.5,0,im.shape[0]-size))
   for col in range(3):
    pic=cv2.cvtColor(im[y:y+size,x:x+size],cv2.COLOR_GRAY2BGR)
    if col:
     for side,ident,pp in uv[col-1]:
      pp=pp-[x,y];color=(240,220,40) if side==0 else (40,150,250)
      for j,k,_,_ in SEGMENTS:cv2.line(pic,tuple(np.rint(pp[j]).astype(int)),tuple(np.rint(pp[k]).astype(int)),color,2,cv2.LINE_AA)
      cv2.putText(pic,f'{"L" if side==0 else "R"} source{ident}',tuple(np.rint(pp.mean(0)).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2,cv2.LINE_AA)
    canvas[90+330*camidx:390+330*camidx,col*320:col*320+300]=cv2.resize(pic,(300,300));cv2.putText(canvas,name+' / '+('image','control','candidate')[col],(col*320+3,77+330*camidx),cv2.FONT_HERSHEY_SIMPLEX,.43,(240,240,240),1,cv2.LINE_AA)
  path=out/(f'{now:.3f}.png');assert cv2.imwrite(str(path),canvas);report.append(dict(now=now,raw_ns=raw,camera_minus_display_ms=delta,visible=meta,image_sha256=sha(path)))
 (out/'result.json').write_text(json.dumps(dict(reviews=report,source_sha256=sha(Path(__file__)),scope='Exact saved head transform; nearest camera; assumed50ms delivery/90Hz. Both sides shown with their own profiles. No physical ground truth.'),indent=2)+'\n');print('Rendered two complete-side scenes.')
if __name__=='__main__':main()
