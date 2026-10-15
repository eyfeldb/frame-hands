"""Score the pre-loss rig against its own camera observations, not ground truth."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from probe_monocular_bridge import RI,MI
from frame_data import read_audit,load_calibration
from audit_native_lengths import stats,sha


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('session','native','run','upstream','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();sys.path.insert(0,str(a.upstream))
 if '143912Z' in str(a.session):raise ValueError('Latest holdout excluded')
 audit=read_audit(a.session/'frame/live-views.txt');cameras=load_calibration(a.session/'frame/snapshot/device/persist')
 records=[json.loads(x) for x in (a.native/'palm-profile-viewer-v2/poses.jsonl').read_text().splitlines()]
 fits=np.loadtxt(a.native/'rig-stage-v2/recorded-output.txt')
 if len(records)!=len(fits) or fits.shape[1]!=104:raise ValueError('Unmatched seed records')
 poses={(r['raw_ns'],r['source_id']):f[41:].reshape(21,3) for r,f in zip(records,fits)}
 rows=[json.loads(x) for x in (a.run/'cases.jsonl').read_text().splitlines()]
 for r in rows:
  points=poses[(r['prior_raw_ns'],r['source'])];errors={}
  for v in audit[r['prior_raw_ns']]['views']:
   if v['id']==r['source']:
    cam=cameras[v['camera']];errors[v['camera']]=float(np.linalg.norm(cam.eye_to_window(cam.world_to_eye(points[RI]))-v['points'][MI],axis=1).mean())
  r['seed_error_px']=errors
 result={}
 for kind in ('mono','reacquired'):
  for admitted in (False,True):
   sub=[r for r in rows if r['kind']==kind and (not admitted or r['fit']['accepted'])]
   key=kind+('_admitted' if admitted else '_all')
   result[key]=dict(count=len(sub),seed_observed_error_px=stats([r['seed_error_px'][r['view']] for r in sub]),current_hold_observed_error_px=stats([r['methods']['hold']['observed_error_px'] for r in sub]),current_flow_observed_error_px=stats([r['methods']['flow']['observed_error_px'] for r in sub]))
   if kind=='reacquired' and sub:
    delta=[r['methods']['flow']['withheld_error_px']-r['methods']['hold']['withheld_error_px'] for r in sub]
    result[key].update(withheld_delta_px=stats(delta),withheld_delta_min_px=float(min(delta)),withheld_delta_mean_px=float(np.mean(delta)))
 with a.output.open('x') as f:f.write(json.dumps(dict(results=result,source_sha256=sha(Path(__file__)),cases_sha256=sha(a.run/'cases.jsonl')),indent=2)+'\n')
 print(json.dumps(result,indent=2))


if __name__=='__main__':main()
