"""Independent packet-quarantine coverage, freshness and presentation audit."""
import argparse,json,sys
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_tracker_rig_check import sha,stats

def read_lines(p):return [json.loads(x) for x in p.read_text().splitlines()]
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('run','control'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();results={}
 for label in ('first','second'):
  data=json.loads((a.run/(label+'.json')).read_text());rows=read_lines(a.run/(label+'-result.json.frames.jsonl'));old=read_lines(a.control/(label+'-result.json.frames.jsonl'));observations=read_lines(a.run/(label+'-result.json.observations.jsonl'))
  assert len(rows)==len(old) and all(r['values'][0]==o['values'][1] for r,o in zip(rows,old))
  by_event=defaultdict(list)
  for o in observations:
   if o['variant']==1:by_event[o['event']].append(o)
  expected_drops=0;expected_hands=0
  for i,e in enumerate(data['events']):
   current=by_event[i];seen=[o for o in current if 'accepted' in o];quarantine=[o for o in current if 'quarantined_sides' in o]
   if not e['valid']:assert not current;continue
   counts=Counter(h['side'] for h in e['hands']);allowed=[h for h in e['hands'] if counts[h['side']]==1];ambiguous=sorted(s for s,n in counts.items() if n>1)
   assert [(o['side'],o['id']) for o in seen]==[(h['side'],h['source_id']) for h in allowed]
   assert len(quarantine)==bool(ambiguous)
   if ambiguous:
    assert quarantine[0]['quarantined_sides']==ambiguous and quarantine[0]['discarded']==len(e['hands'])-len(allowed)
    expected_drops+=1;expected_hands+=quarantine[0]['discarded']
   for o in seen:
    if o['accepted']:assert o['capture']==e['capture'] and o['stamp']==e['stamp']
  latest={};event_index=0;extra=[];jointly_visible=0;groups=[];current_groups={}
  for index,row in enumerate(rows):
   while event_index<len(data['events']) and data['events'][event_index]['now']<=row['now']:
    e=data['events'][event_index]
    if not e['valid']:latest={}
    for o in by_event[event_index]:
     if 'accepted' not in o:continue
     other=1-o['side']
     if other in latest and latest[other]['id']==o['id']:latest.pop(other)
     if o['accepted']:latest[o['side']]=dict(id=o['id'],capture=e['capture'],event=event_index)
    event_index+=1
   x,y=row['values'];side=row['side']
   if x:
    assert y and x==y
    jointly_visible+=1
   if y:
    assert side in latest and y['id']==latest[side]['id'] and y['capture']==latest[side]['capture']
    age=row['now']-y['capture'];assert -1e-8<=age<.17 and abs(age-y['age'])<1e-8
    assert 0<=y['prediction']<=.025+1e-8 and abs(y['opacity']-(1-np.clip((age-.1)/.07,0,1)))<1e-7
   if y and not x:
    item=dict(index=index,now=row['now'],side=side,id=y['id'],age=y['age'],opacity=y['opacity'],evidence_event=latest[side]['event']);extra.append(item)
    if side not in current_groups:current_groups[side]=[]
    current_groups[side].append(item)
   elif side in current_groups:groups.append(current_groups.pop(side))
  groups+=list(current_groups.values())
  # Count complete control-hidden intervals, rather than implying extra frames fix a whole loss.
  fully_bridged=0;losses=0
  for side in (0,1):
   active=False;gap=[]
   for row in [r for r in rows if r['side']==side]:
    x,y=row['values']
    if x:
     if gap:
      losses+=1;fully_bridged+=all(bool(v) for v in gap);gap=[]
     active=True
    elif active:gap.append(y)
  episodes=[dict(start=g[0]['now'],end=g[-1]['now']+1/90,duration_seconds=len(g)/90,side=g[0]['side'],id=g[0]['id'],samples=len(g),first_index=g[0]['index'],max_opacity_index=max(g,key=lambda x:x['opacity'])['index']) for g in groups]
  result=dict(control_samples_exact=len(rows),jointly_visible_samples_exact=jointly_visible,quarantined_packets=expected_drops,quarantined_observations=expected_hands,extra_visible_samples=len(extra),extra_visible_hand_seconds=len(extra)/90,extra_age_seconds=stats([x['age'] for x in extra]),extra_episodes=episodes,completed_control_hidden_intervals=losses,fully_bridged_control_hidden_intervals=fully_bridged,all_extra_capture_clocks_supported=True,limitations=['Input age and visibility preserved; added frames are old estimated poses, not new detections.','Every existing visible output is exact. New visible frames require camera review for stale/wrong-source retention.','Hand-seconds add both sides; not wall-clock duration.'])
  (a.run/(label+'-extra.json')).write_text(json.dumps(extra,indent=2)+'\n');results[label]=result
 disabled=read_lines(a.run/'first-disabled-result.json.frames.jsonl');assert all(r['values'][0]==r['values'][1] for r in disabled)
 (a.run/'audit.json').write_text(json.dumps(dict(results=results,disabled_samples_exact=len(disabled),source_sha256=sha(Path(__file__))),indent=2)+'\n')
 print(json.dumps(results,indent=2))
if __name__=='__main__':main()
