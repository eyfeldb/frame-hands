"""Audit and render the constrained Lite/Full saved-observation comparison."""
import argparse,json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np,cv2
from run_tracker_rig_check import sha,stats
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE.parent/'exploratory'))
from frame_data import load_calibration
from compare_replay import SEGMENTS
SELECTION=['raised-115972546855000-4','raised-115972680190000-4','label-conflict-115975446855000-4','raised-115972680190000-1','back-hand-115986246855000-8','back-hand-115986280188000-8','crossing-115981746856000-4','crossing-115982180189000-1']

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('run','models','session','upstream','profiles'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();sys.path.insert(0,str(a.upstream));meta=json.loads((a.run/'result.json').read_text());modelmeta=json.loads((a.models/'result.json').read_text());assert sha(a.models/'result.json')==meta['model_result_sha256']
 cams=load_calibration(a.session/'frame/snapshot/device/persist');values={};checks=0;summary=defaultdict(dict)
 limits=[np.loadtxt(a.profiles/f'side{s}.fhrig',skiprows=1,max_rows=20)[:,6:8] for s in (0,1)]
 for model in ('lite','full'):
  assert sha(a.run/(model+'-first.txt'))==meta['output_sha256'][model]
  rows=[line.split() for line in (a.run/(model+'-first.txt')).read_text().splitlines()]
  for fields,r in zip(rows,meta['records']):
   assert list(map(int,fields[:3]))==[r['raw_ns'],r['source'],r['side']];row=np.array(fields[3:],float);init=int(row[0]);chosen=int(row[1]);fits=row[11:].reshape(2,92)
   if chosen==-1:np.testing.assert_array_equal(fits[0],fits[1]);checks+=1
   if init:
    assert fits[1,0]<=fits[0,0]+1e-10
    for fit in fits:
     assert np.all(fit[9:29]>=limits[r['side']][:,0]-1e-8) and np.all(fit[9:29]<=limits[r['side']][:,1]+1e-8)
     assert abs(np.linalg.norm(fit[2:6])-1)<1e-6
   values[(r['raw_ns'],r['source'],model)]=dict(initialized=init,chosen=chosen,fits=fits,correspondence=row[4:11].tolist())
  for variant in (0,1):
   states={};windows=defaultdict(lambda:defaultdict(list))
   for r in meta['records']:
    item=values[(r['raw_ns'],r['source'],model)];metric=windows[r['window']];metric['initialized'].append(item['initialized']);metric['chosen'].append(item['chosen']>=0)
    key=(r['window'],r['source'],r['side']);prior=states.get(key);fit=item['fits'][variant]
    if item['initialized']:
     metric['residual_px'].append(fit[1])
     if prior and 0<r['raw_ns']-prior[0]<=100000000:
      metric['max_angle_step_rad'].append(float(np.max(abs(fit[9:29]-prior[1][9:29]))));metric['wrist_step_mm'].append(float(np.linalg.norm(fit[6:9]-prior[1][6:9])))
     states[key]=(r['raw_ns'],fit)
    else:states.pop(key,None)
   for window,m in windows.items():summary[window][model+('-correspondence' if variant else '-base')]=dict(hands=len(m['initialized']),initialized=sum(m['initialized']),correspondence_selected=sum(m['chosen']),**{k:stats(m[k]) if m[k] else None for k in ['residual_px','max_angle_step_rad','wrist_step_mm']},angle_steps_over_half_rad=sum(x>.5 for x in m['max_angle_step_rad']))
 outdir=a.run/'review';outdir.mkdir(exist_ok=False);reviewed=[]
 for case in SELECTION:
  rows=[r for r in modelmeta['records'] if r['case']==case];assert rows
  raw,source=rows[0]['raw_ns'],rows[0]['source'];canvas=np.full((750,1280,3),24,np.uint8);cv2.putText(canvas,case+' / image then fitted poses',(6,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(240,240,240),1,cv2.LINE_AA)
  for v,r in enumerate(rows):
   gray=np.fromfile(a.models/'inputs'/(str(r['index'])+'.gray'),np.uint8).reshape(224,224);inv=cv2.invertAffineTransform(np.array(r['matrix']));camera=cams[r['camera']]
   for col,(model,variant) in enumerate([(None,0),('lite',0),('full',0),('full',1)]):
    pic=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR);label='image';item=None
    if model:
     item=values[(raw,source,model)];label=model+(' corrected' if variant else ' fitted')
     if item['initialized']:
      fit=item['fits'][variant];xyz=fit[29:].reshape(21,3);uv=camera.eye_to_window(camera.world_to_eye(xyz));pts=np.c_[uv,np.ones(21)]@inv.T
      for i,j,_,_ in SEGMENTS:cv2.line(pic,tuple(np.rint(pts[i]).astype(int)),tuple(np.rint(pts[j]).astype(int)),(60,230,70) if model=='full' else (230,170,60),1,cv2.LINE_AA)
      for i in (6,8,11,14,17):cv2.line(pic,tuple(np.rint(pts[5]).astype(int)),tuple(np.rint(pts[i]).astype(int)),(150,150,150),1,cv2.LINE_AA)
      for tag,i in zip('IMRL',(1,2,3,4)):cv2.putText(pic,tag,tuple(np.rint(pts[i]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1,cv2.LINE_AA)
      label+=f" {fit[1]:.1f}px"
     else:label+=' unavailable'
    canvas[85+330*v:385+330*v,320*col:320*col+300]=cv2.resize(pic,(300,300));cv2.putText(canvas,r['camera']+' '+label,(320*col+3,72+330*v),cv2.FONT_HERSHEY_SIMPLEX,.40,(240,240,240),1,cv2.LINE_AA)
  path=outdir/(case+'.png');assert cv2.imwrite(str(path),canvas);reviewed.append(dict(case=case,image_sha256=sha(path),full_choice=values[(raw,source,'full')]['chosen'],full_correspondence=values[(raw,source,'full')]['correspondence']))
 report=dict(scope=__doc__,summary=summary,no_correction_exact_checks=checks,reviews=reviewed,result_sha256=sha(a.run/'result.json'),auditor_sha256=sha(Path(__file__)),limitations=meta['limitations']+['Residuals are against each model own measurements and optional relabeling, not independent accuracy.','Angle/wrist changes include real motion, anatomical side changes and different optimization minima; not headset jitter.','Joint limits passed, but this does not guarantee correct finger labels, skin collision avoidance or physical anatomy.'])
 (a.run/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(summary=summary,reviews=reviewed,no_correction_exact_checks=checks),indent=2))
if __name__=='__main__':main()
