"""Independent decision and neural-view audit for native provisional-side replay."""
import argparse,json,sys
from pathlib import Path
from collections import Counter
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'exploratory'))
from frame_data import read_audit
from audit_host_tracker import read_trace
from run_tracker_rig_check import sha

def read_sides(path):
 frames=[];current=None
 for line in path.read_text().splitlines():
  x=line.split()
  if x[0]=='frame':
   if current is not None:assert len(current['hands'])==current['count']
   current=dict(raw=int(x[1]),count=int(x[2]),hands={});frames.append(current)
  else:
   assert x[0]=='side' and len(x) in (16,18)
   ident=int(x[1]);assert ident not in current['hands'];n=list(map(int,x[3:]));current['hands'][ident]=dict(score=float(x[2]),side=n[0],latched=bool(n[1]),latch_side=n[2],candidate=n[3],streak=n[4],stereo_streak=n[5],enabled=bool(n[6]),active=bool(n[7]),provisional=n[8],last=n[9],vote=n[10],strong=n[11],nviews=n[12],stereo_first=bool(n[13]) if len(n)>13 else False,supported=bool(n[14]) if len(n)>14 else None)
 assert current and len(current['hands'])==current['count'];return frames

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args();reports={}
 for label in ('first','second'):
  root=a.run/label;result=json.loads((root/'result.json').read_text());assert result['baseline_exact'] and result['rig_outputs_repeat_exactly']
  views=[read_audit(root/n/'views.txt') for n in ('0-baseline','1-candidate')];assert views[0].keys()==views[1].keys()
  sides=[read_sides(root/n/'side.txt') for n in ('0-baseline','1-candidate')];assert len(sides[0])==len(sides[1])==1800
  view_rows=0
  for raw in views[0]:
   x,y=views[0][raw],views[1][raw];assert x['hands'].keys()==y['hands'].keys()
   for v,w in zip(x['views'],y['views']):
    assert all(v[k]==w[k] for k in ('camera','id','presence','right')) and np.array_equal(v['points'],w['points'])
   assert len(x['views'])==len(y['views']);view_rows+=len(x['views'])
  pose_rows=[(root/n/'poses.txt').read_text().splitlines() for n in ('0-baseline','1-candidate')];assert len(pose_rows[0])==len(pose_rows[1])
  for x,y in zip(*pose_rows):
   x=x.split();y=y.split();assert x[:2]+x[3:]==y[:2]+y[3:]
  previous={};checked=0;joint_overrides=0;side_changes=[0,0];duplicates=[0,0];last=[{},{}];changed=[];rigs=[read_trace(root/n/'rig.txt') for n in ('0-baseline','1-candidate')]
  for f,g,rf,rg in zip(*sides,*rigs):
   assert f['raw']==g['raw']==rf['raw']==rg['raw'];raw=f['raw'];assert f['hands'].keys()==g['hands'].keys()
   for variant,frame in enumerate((f,g)):
    duplicates[variant]+=any(n>1 for n in Counter(h['side'] for h in frame['hands'].values()).values())
    for ident,h in frame['hands'].items():
     if ident in last[variant] and raw-last[variant][ident][0]<=100000000:side_changes[variant]+=h['side']!=last[variant][ident][1]
     last[variant][ident]=(raw,h['side'])
   for ident,h in g['hands'].items():
    assert h['enabled'] and h['active'] and h['last']==raw
    observed=[v for v in views[1][raw]['views'] if v['id']==ident]
    votes=[1 if v['right']>.65 else 0 if v['right']<.35 else -1 for v in observed]
    qualified=len(votes)>=2 and votes[0]>=0 and all(v==votes[0] for v in votes) and all(v['presence']>=.8 for v in observed)
    assert h['vote']==(votes[0] if qualified else -1)
    old=previous.get(ident)
    # Joint conflict recovery occurs after provisional.observe. Its final score
    # overwrites the unlogged pre-recovery score; audit the published latch, not
    # an invented pre-recovery value.
    joint=h['latched'] and (h['strong']<0 or h['strong']!=h['latch_side'])
    if joint:joint_overrides+=1
    elif h['stereo_first']:
     supported=old is not None and raw-old['raw']<=100000000 and old['supported']
     if h['strong']>=0:expected=h['strong'];supported=True
     elif h['vote']>=0:expected=h['vote'];supported=True
     else:expected=old['side'] if supported else int(h['score']>.5)
     assert h['provisional']==expected and h['supported']==supported;checked+=1
    else:
     expected=h['strong'] if h['strong']>=0 else int(h['score']>.5) if old is None or raw-old['raw']>100000000 else old['side']
     if h['strong']<0 and old is not None and raw-old['raw']<=100000000:
      if expected==0 and h['score']>=.65 and h['vote']==1:expected=1
      elif expected==1 and h['score']<=.35 and h['vote']==0:expected=0
     assert h['provisional']==expected;checked+=1
    assert h['side']==(h['latch_side'] if h['latched'] else h['provisional'])
    previous[ident]=dict(raw=raw,side=h['provisional'],supported=h['supported'])
    assert rg['rows'][ident]['side']==h['side'] and rf['rows'][ident]['side']==f['hands'][ident]['side']
    if h['side']!=f['hands'][ident]['side']:changed.append(dict(raw_ns=raw,source_id=ident,before=f['hands'][ident]['side'],after=h['side'],score=h['score'],vote=h['vote'],latched=h['latched'],status=rg['rows'][ident]['status']))
  report=dict(input_result_sha256=sha(root/'result.json'),candidate_rig_sha256=sha(root/'1-candidate/rig.txt'),baseline_rig_sha256=sha(root/'0-baseline/rig.txt'),raw_detector_geometry_exact=True,frames=1800,neural_view_rows_exact=view_rows,pose_geometry_rows_exact=len(pose_rows[0]),component_decisions_checked=checked,joint_override_rows=joint_overrides,side_changes=side_changes,duplicate_side_frames=duplicates,changed_side_rows=len(changed),changed=changed,limitations=['Published side differs deliberately; raw per-view scores/points and source associations remain exact.','Fewer changes/duplicates is not proof of correct physical identity.','Post-observe joint-latch overrides are checked as published latch decisions; no pre-override score is invented.'])
  (root/'side-audit.json').write_text(json.dumps(report,indent=2)+'\n');reports[label]={k:v for k,v in report.items() if k!='changed'}
 (a.run/'audit.json').write_text(json.dumps(dict(results=reports,source_sha256=sha(Path(__file__))),indent=2)+'\n');print(json.dumps(reports,indent=2))
if __name__=='__main__':main()
