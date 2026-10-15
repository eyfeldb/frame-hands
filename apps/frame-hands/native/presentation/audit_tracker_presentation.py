"""Audit full-tracker presentation output independently, including camera-time selection."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_tracker_rig_check import sha,stats

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('run','control'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();results={}
 for label in ('first','second'):
  data=json.loads((a.run/(label+'.json')).read_text());rows=[json.loads(x) for x in (a.run/(label+'-result.json.frames.jsonl')).read_text().splitlines()];old=[json.loads(x) for x in (a.control/(label+'-result.json.frames.jsonl')).read_text().splitlines()]
  assert len(rows)==len(old) and all(r['values'][0]==o['values'][1] for r,o in zip(rows,old))
  previous={};steps=[];offset=[];hidden=[[],[]];active_gaps={};same_coverage=True
  for i,row in enumerate(rows):
   now,side=row['now'],row['side'];values=row['values'];key=side
   assert bool(values[0])==bool(values[1])
   for v,s in enumerate(values):
    k=(side,v)
    if not s and k not in active_gaps:active_gaps[k]=now
    if s and k in active_gaps:hidden[v].append(dict(side=side,start=active_gaps.pop(k),end=now))
   if values[0]:
    x,y=values;assert all(x[k]==y[k] for k in ('id','opacity','capture','age','held','prediction'))
    distance=float(np.linalg.norm(np.array(x['position'])-y['position'])*1000);offset.append(dict(index=i,now=now,side=side,mm=distance))
    assert all(np.isfinite(s['position']).all() and np.isfinite(s['angles']).all() and np.isfinite(s['orientation']).all() for s in values)
    if side in previous and all(previous[side]['values']):
     before=previous[side];record=dict(index=i,now=now,previous=before['index'],side=side,same_id=values[0]['id']==before['values'][0]['id'],mm=[],degrees=[])
     for v in (0,1):
      r,s=before['values'][v],values[v];record['mm'].append(float(np.linalg.norm(np.array(r['position'])-s['position'])*1000));q=Rotation.from_quat(np.array(r['orientation'])[[1,2,3,0]]);z=Rotation.from_quat(np.array(s['orientation'])[[1,2,3,0]]);record['degrees'].append(float((q.inv()*z).magnitude()*180/np.pi))
     steps.append(record)
   previous[side]=dict(index=i,values=values)
  for (side,v),start in active_gaps.items():hidden[v].append(dict(side=side,start=start,end=rows[-1]['now']+1/90))
  report=dict(control_samples_exact=len(rows),visibility_identity_opacity_capture_exact=True,step_mm=[stats([r['mm'][v] for r in steps]) for v in (0,1)],step_degrees=[stats([r['degrees'][v] for r in steps]) for v in (0,1)],steps_over45=[sum(r['degrees'][v]>45 for r in steps) for v in (0,1)],offset_mm=stats([r['mm'] for r in offset]),max_offset=max(offset,key=lambda r:r['mm']),largest_baseline_step=max(steps,key=lambda r:r['mm'][0]),largest_candidate_step=max(steps,key=lambda r:r['mm'][1]),hidden_runs=hidden,limits=['Assumed50ms delivery/90Hz sampling; no actual native rendering latency.','Step statistics combine real motion and estimation errors. Lower values alone do not establish better tracking.','Output pose differences are filter delay, not known error relative to the true hand.'])
  (a.run/(label+'-motion-steps.json')).write_text(json.dumps(steps)+'\n');results[label]=report
 report=dict(results=results,source_sha256=sha(Path(__file__)))
 (a.run/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:{n:v[n] for n in ('control_samples_exact','steps_over45','offset_mm','largest_baseline_step','largest_candidate_step')} for k,v in results.items()},indent=2))
if __name__=='__main__':main()
