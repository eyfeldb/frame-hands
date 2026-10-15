"""Audit changed side/rig presentation without treating fewer flips as identity truth."""
import argparse,json,sys
from pathlib import Path
from collections import Counter
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_tracker_rig_check import sha,stats

def lines(p):return [json.loads(x) for x in p.read_text().splitlines()]
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('run','control'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();reports={}
 for label in ('first','second'):
  data=json.loads((a.run/(label+'.json')).read_text());rows=lines(a.run/(label+'-result.json.frames.jsonl'));old=lines(a.control/(label+'-result.json.frames.jsonl'));obs=lines(a.run/(label+'-result.json.observations.jsonl'))
  assert len(rows)==len(old) and all(r['now']==o['now'] and r['side']==o['side'] and r['values'][0]==o['values'][1] for r,o in zip(rows,old))
  accepted=[set(),set()];ambiguous=[0,0]
  for o in obs:
   variant=o['variant']
   if o.get('ambiguous_clear'):ambiguous[variant]+=1
   if o.get('accepted'):
    e=data['events'][o['event']];assert o['capture']==e['capture'] and o['stamp']==e['stamp']
    assert any(h['side']==o['side'] and h['source_id']==o['id'] for h in e['hands_by_variant'][variant]);accepted[variant].add((o['side'],o['id'],o['capture']))
  measures=[]
  for variant in (0,1):
   previous={};position=[];rotation=[];angle=[];visible=0;sourcechanges=0
   for row in rows:
    s=row['side'];v=row['values'][variant]
    if not v:previous.pop(s,None);continue
    visible+=1;assert (s,v['id'],v['capture']) in accepted[variant]
    age=row['now']-v['capture'];assert -1e-8<=age<.17 and abs(age-v['age'])<1e-8
    assert 0<=v['prediction']<=.025+1e-8 and abs(v['opacity']-(1-np.clip((age-.1)/.07,0,1)))<1e-7
    assert len(v['angles'])==20 and np.isfinite(v['angles']).all()
    prior=previous.get(s)
    if prior:
     if prior['id']==v['id']:
      position.append(float(np.linalg.norm(np.array(v['position'])-prior['position'])*1000))
      rotation.append(float(np.degrees(2*np.arccos(np.clip(abs(np.dot(v['orientation'],prior['orientation'])),0,1)))))
      angle.append(float(np.max(np.abs(np.array(v['angles'])-prior['angles']))))
     else:sourcechanges+=1
    previous[s]=v
   measures.append(dict(visible_samples=visible,ambiguous_clears=ambiguous[variant],same_source_position_step_mm=stats(position),same_source_rotation_step_degrees=stats(rotation),rotation_steps_over45=sum(x>45 for x in rotation),max_joint_angle_step_radians=max(angle,default=0),visible_source_changes=sourcechanges))
  extra=[i for i,r in enumerate(rows) if r['values'][1] and not r['values'][0]];lost=[i for i,r in enumerate(rows) if r['values'][0] and not r['values'][1]]
  changed=[i for i,r in enumerate(rows) if r['values'][0]!=r['values'][1]]
  coverage=dict(extra_source_samples=0,missing_source_samples=0,same_source_changed_side_samples=0)
  for i in range(0,len(rows),2):
   pair=rows[i:i+2];assert len(pair)==2 and pair[0]['now']==pair[1]['now'] and [r['side'] for r in pair]==[0,1]
   maps=[]
   for variant in (0,1):
    shown=[(r['values'][variant]['id'],r['side']) for r in pair if r['values'][variant]];assert len({x[0] for x in shown})==len(shown);maps.append(dict(shown))
   before,after=maps;coverage['extra_source_samples']+=len(after.keys()-before.keys());coverage['missing_source_samples']+=len(before.keys()-after.keys());coverage['same_source_changed_side_samples']+=sum(before[k]!=after[k] for k in before.keys()&after.keys())
  reports[label]=dict(source_visibility=coverage,control_samples_exact=len(rows),variants=measures,extra_visible_samples=len(extra),lost_visible_samples=len(lost),changed_samples=len(changed),first_extra_index=extra[0] if extra else None,first_lost_index=lost[0] if lost else None,first_changed_index=changed[0] if changed else None,frames_sha256=sha(a.run/(label+'-result.json.frames.jsonl')),observations_sha256=sha(a.run/(label+'-result.json.observations.jsonl')))
 for suffix in ('.frames.jsonl','.observations.jsonl'):assert sha(a.run/('second-result.json'+suffix))==sha(a.run/('second-repeat-result.json'+suffix))
 report=dict(results=reports,repeat_exact=True,source_sha256=sha(Path(__file__)),limitations=['Two saved clips; assumed50ms transport/90Hz ticks. No native performance or wearer result.','Motion counts exclude gaps and source changes. Lower motion is not physical accuracy.','Visible output has accepted capture provenance and original170ms expiry; no renewed clocks.'])
 (a.run/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(reports,indent=2))
if __name__=='__main__':main()
