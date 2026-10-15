"""Shadow replay the proposed scheduler against saved image-side evidence.

Association is recomputed causally. No inference speed/pose accuracy is claimed.
"""
import argparse,collections,io,json,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE.parent/'exploratory'))
from frame_data import load_calibration,read_audit
from temporal_association import TemporalAssociation
from replay_mesh_trial import packets
from run_mesh_trial import pose_matrix
from identity_schedule import IdentitySchedule
p=argparse.ArgumentParser(description=__doc__)
for k in ('session','evidence','output','upstream'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(a.upstream))
refs={}
for line in (a.session/'predictions.jsonl').open():
 r=json.loads(line);refs[r['capture_raw_ns']]={k:r[k] for k in ('capture_pose','current_pose')}
evs={r['stamp']:r for line in a.evidence.open() if (r:=json.loads(line))};cameras=load_calibration(a.session/'calibration')
schedule=IdentitySchedule();candidate=TemporalAssociation(cameras);control=TemporalAssociation(cameras);reasons=collections.Counter();differences=[];control_diffs=[];frames=0;fresh_views=all_views=0
with (a.output/'trace.jsonl').open('w') as out:
 for meta,blobs in packets(a.session/'transport.bin'):
  if meta.get('waiting'):continue
  ref=refs.get(meta['stamp']);ev=evs.get(meta['stamp'])
  if ref is None or ev is None or not ref['capture_pose']['valid']:
   schedule.clear();candidate.reset();control.reset();continue
  obs=next(iter(read_audit(io.StringIO(meta['audit'])).values()));world=pose_matrix(ref['capture_pose']['capture'])
  rows={(r['camera'],r['source_id']):r for r in ev.get('views',[])}
  fresh=dict(obs,views=[dict(v,right=rows[v['camera'],v['id']]['right']) for v in obs['views'] if (v['camera'],v['id']) in rows])
  selected,reason=schedule.decide(obs)
  if selected is None:selected=fresh;schedule.observe(obs,ev.get('views',[]));fresh_views+=len(rows)
  reasons[reason]+=1;all_views+=len(rows);frames+=1
  candidate.assign(selected,world);control.assign(fresh,world)
  def shape(debug):return [(r['side'],r['track'],sorted((v['camera'],v['source_id']) for v in r['views'])) for r in debug.get('assignments',[])]
  original=shape(ev.get('association',{}));baseline=shape(control.last_debug);actual=shape(candidate.last_debug)
  if baseline!=original:control_diffs.append(meta['stamp'])
  if actual!=baseline:differences.append({'stamp':meta['stamp'],'reason':reason,'candidate':actual,'baseline':baseline})
  out.write(json.dumps({'stamp':meta['stamp'],'reason':reason,'candidate':actual,'baseline':baseline})+'\n')
report={'scope':__doc__,'frames':frames,'reasons':dict(reasons),'image_evidence_views_original':all_views,'image_evidence_views_scheduled':fresh_views,'control_assignment_mismatch_count':len(control_diffs),'control_mismatch_stamps':control_diffs,'candidate_assignment_differences':differences}
(a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
