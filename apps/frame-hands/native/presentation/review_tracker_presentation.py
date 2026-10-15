"""Project displayed poses into nearby saved camera images; timing mismatch is explicit."""
import argparse,json,sys,subprocess
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
BASE=Path(__file__).resolve().parents[1];sys.path[:0]=[str(BASE),str(BASE.parent/'exploratory')]
from frame_data import Recording,load_calibration
from compare_replay import SEGMENTS
from run_tracker_rig_check import sha

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('run','first-session','second-session','upstream','profiles','evaluator'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();sys.path.insert(0,str(a.upstream));out=a.run/'review';out.mkdir(exist_ok=False);audit=json.loads((a.run/'audit.json').read_text());reports=[]
 for label,session in [('first',a.first_session),('second',a.second_session)]:
  if '143912Z' in str(session):raise ValueError('Accepted holdout excluded')
  data=json.loads((a.run/(label+'.json')).read_text());rows=[json.loads(x) for x in (a.run/(label+'-result.json.frames.jsonl')).read_text().splitlines()];heads_path=Path(data['provenance']['head_evidence']['path'])/'heads.txt'
  heads={int(x[0]):(int(x[1]),int(x[2]),np.array(x[3:],float)) for x in [line.split() for line in heads_path.read_text().splitlines()[1:]]}
  events=data['events'];times=np.array([e['capture'] for e in events]);rec=Recording(session/'frame/sets.bin');cameras=load_calibration(session/'frame/snapshot/device/persist');origin=rec.index[0][0]
  selected=[(key,audit['results'][label][key]['index']) for key in ('largest_baseline_step','max_offset','largest_candidate_step')]
  for kind,index in selected:
   row=rows[index];event=events[int(np.argmin(abs(times-row['now'])))];raw=event['raw_ns'];mono,valid,h=heads[raw];assert valid and all(row['values']);side=row['side']
   hr=Rotation.from_quat(h[:4][[1,2,3,0]]);local=[]
   for s in row['values']:
    q=hr.inv()*Rotation.from_quat(np.array(s['orientation'])[[1,2,3,0]]);t=hr.inv().apply(np.array(s['position'])*1000-h[4:]);local.append([*q.as_quat()[[3,0,1,2]],*t,*s['angles']])
   input_path=out/(label+'-'+kind+'-input.txt');points_path=out/(label+'-'+kind+'-points.txt')
   input_path.write_text(''.join(str(side)+' '+' '.join(format(v,'.17g') for v in pose)+'\n' for pose in local))
   subprocess.run([str(a.evaluator.resolve()),str((a.profiles/f'side{side}.fhrig').resolve()),str(input_path.resolve()),str(points_path.resolve())],check=True,capture_output=True,text=True)
   pts=np.loadtxt(points_path)[:,1:].reshape(2,21,3);seconds=(raw-origin)/1e9;frames=list(rec.frames(seconds-1e-7,seconds+1e-7));assert len(frames)==1 and frames[0][1]==raw
   canvas=np.full((750,960,3),24,np.uint8);delta=(event['capture']-row['now'])*1000
   cv2.putText(canvas,f'{label}/{kind} display{row["now"]:.3f}s side{side}',(6,24),cv2.FONT_HERSHEY_SIMPLEX,.56,(240,240,240),1,cv2.LINE_AA)
   cv2.putText(canvas,f'camera minus display {delta:+.2f}ms; assumed50ms delivery /90Hz',(6,48),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,240,240),1,cv2.LINE_AA)
   for v,name in enumerate(('slam_left','slam_right')):
    im=frames[0][2][name][0];cam=cameras[name];uv=[cam.eye_to_window(cam.world_to_eye(p)) for p in pts];alluv=np.vstack(uv);center=(alluv.min(0)+alluv.max(0))*.5;size=int(min(min(im.shape),max(360,np.ptp(alluv,axis=0).max()+200)));x=int(np.clip(center[0]-size*.5,0,im.shape[1]-size));y=int(np.clip(center[1]-size*.5,0,im.shape[0]-size))
    for col in range(3):
     pic=cv2.cvtColor(im[y:y+size,x:x+size],cv2.COLOR_GRAY2BGR)
     if col:
      pp=uv[col-1]-[x,y]
      for j,k,_,_ in SEGMENTS:cv2.line(pic,tuple(np.rint(pp[j]).astype(int)),tuple(np.rint(pp[k]).astype(int)),((240,220,40),(70,230,60))[col-1],2,cv2.LINE_AA)
     canvas[90+330*v:390+330*v,col*320:col*320+300]=cv2.resize(pic,(300,300));cv2.putText(canvas,name+' / '+('image','existing','reacquisition')[col],(col*320+3,77+v*330),cv2.FONT_HERSHEY_SIMPLEX,.43,(240,240,240),1,cv2.LINE_AA)
   file=out/(label+'-'+kind+'.png');assert cv2.imwrite(str(file),canvas)
   reports.append(dict(label=label,kind=kind,display_seconds=row['now'],camera_raw_ns=raw,camera_seconds=seconds,camera_minus_display_ms=delta,side=side,source_id=row['values'][0]['id'],image_sha256=sha(file),camera_alignment='Exact saved capture head; nearest saved camera time, mismatch shown',poses_world=row['values']))
 (out/'result.json').write_text(json.dumps(dict(reviews=reports,source_sha256=sha(Path(__file__)),limitations=['Static nearby-camera overlays, not optical ground truth or a rendered headset trial.','Counterfactual50ms delivery/90Hz display. Camera timing mismatch is shown explicitly.','Fixed rig projection validates shape/placement only; mesh self-intersection and occlusion are not simulated.']),indent=2)+'\n');print('Rendered six presentation comparison sheets.')
if __name__=='__main__':main()
