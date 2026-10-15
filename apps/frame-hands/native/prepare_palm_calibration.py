"""Select early open, separate hands for shared palm calibration; later frames remain evaluation."""
import argparse,json,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE.parent/'exploratory'))
from frame_data import read_audit,load_calibration
from stereo_uncertainty import sensitivity
from audit_native_lengths import sha
from palm_calibration import PALM_MP,calibrate

def overlap(a,b):
 lo=np.maximum(a.min(0),b.min(0));hi=np.minimum(a.max(0),b.max(0));return bool(np.all(hi>lo))
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('session','run','upstream','profile','output'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--review',type=Path)
 a=p.parse_args();sys.path.insert(0,str(a.upstream));a.output.mkdir(parents=True,exist_ok=False)
 if '143912Z' in str(a.session):raise ValueError('Accepted holdout excluded')
 audit=read_audit(a.session/'frame/live-views.txt');cams=load_calibration(a.session/'frame/snapshot/device/persist');rows=[json.loads(x) for x in (a.run/'isotropic-poses.jsonl').read_text().splitlines()];first=min(audit);chosen={0:[],1:[]};last={0:-1,1:-1};rest=np.loadtxt(a.profile,skiprows=21)[:,:3]
 for r in rows:
  seconds=(r['raw_ns']-first)/1e9;side=r['side']
  if seconds>=35 or seconds-last[side]<.25 or len(chosen[side])>=24:continue
  allviews=audit[r['raw_ns']]['views'];views=[v for v in allviews if v['id']==r['source_id']]
  if len(views)!=2 or min(v['presence'] for v in views)<.95 or any((v['right']>.5)!=bool(side) or max(v['right'],1-v['right'])<.9 for v in views):continue
  if any(overlap(v['points'],other['points']) for v in views for other in allviews if other['id']!=r['source_id'] and other['camera']==v['camera']):continue
  x=np.array(r['raw_head_mm']);straight=[]
  for f in range(4):
   pts=x[np.arange(5+4*f,9+4*f)];straight.append(np.linalg.norm(pts[-1]-pts[0])/np.linalg.norm(np.diff(pts,axis=0),axis=1).sum())
  if min(straight)<.94:continue
  _,cov,info=sensitivity(cams,views)
  if np.percentile(info['reprojection_px'][:,PALM_MP],95)>4:continue
  uv=np.array([next(v for v in views if v['camera']==name)['points'][PALM_MP] for name in ('slam_left','slam_right')]);chosen[side].append(dict(raw_ns=r['raw_ns'],source_id=r['source_id'],seconds=seconds,side=side,pixels=uv.tolist(),stereo=x[PALM_MP].tolist()));last[side]=seconds
 review=None
 if a.review:
  review=json.loads(a.review.read_text())
  if review['session']!=a.session.name or review['audit_sha256']!=sha(a.session/'frame/live-views.txt'):raise ValueError('Mismatched visual review')
  excluded={(x['raw_ns'],x['source_id']) for x in review['exclude']}
  available={(x['raw_ns'],x['source_id']) for values in chosen.values() for x in values}
  if not excluded<=available:raise ValueError('Excluded frame was not selected')
  chosen={side:[r for r in items if (r['raw_ns'],r['source_id']) not in excluded] for side,items in chosen.items()}
 result={}
 for side,items in chosen.items():
  if len(items)<3:result[side]=dict(status='insufficient',frames=len(items));continue
  uv=np.array([r['pixels'] for r in items]);xyz=np.array([r['stereo'] for r in items]);np.savez_compressed(a.output/f'side{side}.npz',pixels=uv,stereo=xyz)
  result[side]=dict(frames=len(items),control=calibrate(rest,side,cams,uv,xyz,True),candidate=calibrate(rest,side,cams,uv,xyz))
 report=dict(scope=__doc__,session=a.session.name,selection_parameters=dict(before_seconds=35,min_spacing_seconds=.25,min_presence=.95,min_side_probability=.9,min_four_finger_straightness=.94,max_palm_stereo_p95_px=4,no_camera_bbox_overlap=True),selection=chosen,visual_review=review,results=result,provenance=dict(profile_sha256=sha(a.profile),poses_sha256=sha(a.run/'isotropic-poses.jsonl'),audit_sha256=sha(a.session/'frame/live-views.txt'),sources={n:sha(BASE/n) for n in ('palm_calibration.py','prepare_palm_calibration.py')}),limitations=['Selection heuristics are not proof of visibility; raw image review required.','Shape is fit to detector landmarks, whose relation to anatomical joints is imperfect.','35s boundary excludes later frames from shape fitting; no later-frame evaluation yet.','No user scan, accepted profile or runtime changes.'])
 (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({s:dict(frames=v['frames'],fits={m:{k:v[m][k] for k in ('shape','success','residual_px')} for m in ('control','candidate')}) for s,v in result.items() if 'candidate' in v},indent=2))
if __name__=='__main__':main()
