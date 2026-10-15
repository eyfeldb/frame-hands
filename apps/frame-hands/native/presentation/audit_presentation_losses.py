"""Attribute display losses to the replay contract and recorded upstream availability."""
import argparse,json,sys
from pathlib import Path
from collections import Counter,defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit_host_tracker import read_trace
from run_tracker_rig_check import sha

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('presentation','tracker','output'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();report={}
 for label in ('first','second'):
  data=json.loads((a.presentation/(label+'.json')).read_text());events=data['events'];rows=[json.loads(x) for x in (a.presentation/(label+'-result.json.frames.jsonl')).read_text().splitlines()];obs=[json.loads(x) for x in (a.presentation/(label+'-result.json.observations.jsonl')).read_text().splitlines()]
  observations=defaultdict(list)
  for o in obs:observations[(o['event'],o['variant'])].append(o)
  rigs=[read_trace(a.tracker/label/n/'rig.txt') for n in ('0-baseline','1-candidate')];assert all(len(r)==len(events) for r in rigs)
  for r in rigs:assert all(e['raw_ns']==f['raw'] for e,f in zip(events,r))
  results=[]
  for variant in (0,1):
   previous={};losses=[]
   for row in rows:
    s=row['side'];v=row['values'][variant];old=previous.get(s)
    if old and old['value'] and not v:
     prior=old['value'];indices=[i for i,e in enumerate(events) if old['now']<e['now']<=row['now']];reason='unclassified'
     if any(not events[i]['valid'] for i in indices):reason='head_context_clear'
     elif any(o.get('ambiguous_clear') for i in indices for o in observations[(i,variant)]):reason='ambiguous_packet_clear'
     elif any(o.get('id')==prior['id'] and o.get('side')==1-s for i in indices for o in observations[(i,variant)]):reason='source_changed_side'
     elif row['now']-prior['capture']>=.17-1e-9:reason='capture_expiry'
     recent=[i for i,e in enumerate(events) if prior['capture']<=e['capture'] and e['now']<=row['now']];status=Counter()
     for i in recent:
      source=rigs[variant][i]['rows'].get(prior['id']);status[source['status'] if source else 'absent_source']+=1
     losses.append(dict(now=row['now'],side=s,source=prior['id'],reason=reason,age=row['now']-prior['capture'],upstream_since_last_capture=dict(status)))
    previous[s]=dict(now=row['now'],value=v)
   results.append(dict(counts=dict(Counter(v['reason'] for v in losses)),losses=losses))
  assert all('unclassified' not in v['counts'] for v in results), 'Inspect unclassified losses before using this report'
  report[label]=results
 a.output.write_text(json.dumps(dict(results=report,source_sha256=sha(Path(__file__)),limitations=['Counts are per-hand visible-to-hidden transitions, not whole-scene episodes. Cause labels describe the exact offline viewer contract, not physical tracking truth.','Upstream availability since last accepted capture can include rejected fits and is descriptive, not proof of their accuracy.','50ms delivery and90Hz display are assumptions.']),indent=2)+'\n')
 print(json.dumps({label:[v['counts'] for v in variants] for label,variants in report.items()},indent=2))
if __name__=='__main__':main()
