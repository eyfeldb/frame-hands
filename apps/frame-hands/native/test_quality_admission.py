"""Construct and verify full-replay quality-gate clock/identity controls."""
import argparse,copy,json,math
from pathlib import Path

def fixture(template):
 events=[]
 for frame in range(60):
  capture=frame/30.;h=copy.deepcopy(template);h['side']=0 if frame<36 else 1;h['source_id']=1 if frame<30 else 2
  h['wrist_world_m']=[[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,-.45],[0.,0.,0.,1.]]
  if 45<=frame<=47:
   angle=(frame-44)*math.pi/3;c,s=math.cos(angle),math.sin(angle);h['wrist_world_m']=[[c,-s,0.,(frame-44)*.03],[s,c,0.,0.],[0.,0.,1.,-.45],[0.,0.,0.,1.]]
  h['fit_rmse_px']=20. if frame in [7,8,9,30,36] or 16<=frame<=25 else 0.
  events.append(dict(now=capture+.05,capture=capture,stamp=capture,hands=[h],valid=frame!=42))
 return dict(name='quality_gate_boundary_controls',scope='Constructed pose stream, assumed50ms delivery. Tests gating clocks/identity, not anatomy accuracy.',events=events)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--prepare',action='store_true');a=p.parse_args()
 data=json.loads((a.run/'events.json').read_text());template=next(h for e in data['events'] for h in e['hands']);expected=fixture(template)
 if a.prepare:
  with (a.run/'controls.json').open('x',encoding='utf-8') as f:json.dump(expected,f)
  return
 assert json.loads((a.run/'controls.json').read_text())==expected
 frames=[json.loads(x) for x in (a.run/'controls-result.json.frames.jsonl').read_text().splitlines()]
 def at(frame,side=0):
  t=frame/30.+.05
  return next(x for x in frames if x['side']==side and x['now']>=t)['values']
 checks=[]
 def check(value,message):assert value,message;checks.append(message)
 for f in (7,8,9):
  b,c=at(f);check(bool(c) and abs(c['capture']-.2)<1e-7,'short rejection holds original capture '+str(f))
 b,c=at(10);check(bool(c) and abs(c['capture']-10/30.)<1e-7,'fresh recovery has no quality confirmation delay')
 b,c=at(22);check(bool(b) and not c,'extended bad observations expire rather than renew presence')
 b,c=at(26);check(bool(c) and abs(c['capture']-26/30.)<1e-7,'recovery after expiry')
 b,c=at(30);check(not c,'rejected new source clears old source')
 b,c=at(31);check(c['id']==2,'new valid source reacquires')
 check(not at(36,0)[1] and not at(36,1)[1],'rejected side reassignment clears both old and unavailable new side')
 check(at(37,1)[1]['id']==2,'side recovery')
 check(not at(42,1)[1],'invalid HMD clears presentation')
 for f in (45,46,47):
  b,c=at(f,1);check(b==c,'clean quick gesture unchanged '+str(f))
 good={(round(e['capture'],7),h['side'],h['source_id']) for e in expected['events'] if e['valid'] for h in e['hands'] if h['fit_rmse_px']<=9}
 for f in frames:
  c=f['values'][1]
  if c:assert (round(c['capture'],7),f['side'],c['id']) in good and c['age']<.170+1e-6
 check(True,'all visible outputs trace to admitted capture with bounded age')
 report=dict(controls=len(checks),checks=checks,failures=0,scope='Constructed replay through actual GDScript gate and presenter; no optical accuracy claim.')
 (a.run/'control-checks.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report))
if __name__=='__main__':main()
