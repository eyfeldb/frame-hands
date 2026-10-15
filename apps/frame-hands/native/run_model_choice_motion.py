"""Replay frozen model-choice poses through unchanged native motion presentation.
Exact saved head joins; assumed50ms delivery/90Hz. No camera, XR, or native timing.
"""
import argparse,json,sys,subprocess,shutil,os
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.spatial.transform import Rotation
from run_tracker_rig_check import sha,verify_head_evidence,stats


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('run','rig','heads','godot','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();meta=json.loads((a.run/'result.json').read_text());assert meta['independent_history'] and meta['baseline_exact'];headmeta=json.loads((a.heads/'manifest.json').read_text());headproof=verify_head_evidence(a.heads,headmeta['recording_sha256'])
 assert headmeta['recording_sha256']=='34eb755f1f5285cdf0c94c6d45250ccb3739fe2409abbb958485ea13ef40d3d4'
 assert sha(a.rig/'result.json')==meta['rig_result_sha256'] and sha(a.run/'first-fitted.txt')==meta['outputs']['fitted']
 heads={}
 for line in (a.heads/'heads.txt').read_text().splitlines()[1:]:
  f=line.split();raw,mono,valid=map(int,f[:3]);assert raw not in heads;heads[raw]=(mono,valid,np.array(f[3:],float))
 rigmeta=json.loads((a.rig/'result.json').read_text());assert sha(a.rig/'full-first.txt')==rigmeta['output_sha256']['full'] and meta['records']==rigmeta['records']
 rows=[np.loadtxt(a.rig/'full-first.txt'),np.loadtxt(a.run/'first-fitted.txt')];byraw=defaultdict(list)
 for i,r in enumerate(meta['records']):byraw[r['raw_ns']].append(i)
 events=[];origin=next(iter(heads.values()))[0];roundtrip=0.
 for raw,(mono,valid,h) in list(heads.items())[:1800]:
  variants=[[],[]]
  if valid:
   hr=Rotation.from_quat(h[:4][[1,2,3,0]])
   for variant in (0,1):
    for i in byraw.get(raw,[]):
     r=meta['records'][i];row=rows[variant][i];assert list(row[:3].astype(np.int64))==[r['raw_ns'],r['source'],r['side']]
     if not row[3]:continue
     fit=row[14:106];world=np.eye(4);world[:3,:3]=(hr*Rotation.from_quat(fit[2:6][[1,2,3,0]])).as_matrix();world[:3,3]=(hr.apply(fit[6:9])+h[4:])*.001
     roundtrip=max(roundtrip,float(np.max(abs(hr.inv().apply(world[:3,3]*1000-h[4:])-fit[6:9]))))
     variants[variant].append(dict(side=r['side'],source_id=r['source'],wrist_world_m=world.tolist(),joint_angles=fit[9:29].tolist()))
  t=(mono-origin)/1e9;events.append(dict(now=t+.05,capture=t,stamp=t,raw_ns=raw,valid=bool(valid),hands_by_variant=variants))
 assert roundtrip<1e-9 and all(e['stamp']<f['stamp'] for e,f in zip(events,events[1:]));a.output.mkdir(parents=True,exist_ok=False)
 source=Path(__file__).parent;presentation=source/'presentation'
 for name in ('hand_motion.gd','replay_side_motion.gd'):shutil.copy2(presentation/name,a.output/name)
 shutil.copy2(source.parent/'exploratory/mesh-viewer/hand_motion.gd',a.output/'baseline_motion.gd')
 (a.output/'project.godot').write_text('config_version=5\n[application]\nconfig/name="Offline model-choice motion"\n[rendering]\nrenderer/rendering_method="gl_compatibility"\n')
 fixture=dict(name='model-choice-isolated-history',scope=__doc__,events=events,provenance=dict(head_evidence=headproof,fit_result_sha256=sha(a.run/'result.json')))
 (a.output/'input.json').write_text(json.dumps(fixture,separators=(',',':'))+'\n')
 control=json.loads(json.dumps(fixture))
 for e in control['events']:e['hands_by_variant'][1]=e['hands_by_variant'][0]
 (a.output/'control.json').write_text(json.dumps(control,separators=(',',':'))+'\n')
 command=[str(a.godot.resolve()),'--headless','--path',str(a.output.resolve()),'--xr-mode','off','--script','res://replay_side_motion.gd'];reports={}
 for name,input_name in [('first','input'),('repeat','input'),('control','control')]:
  env=os.environ.copy();env.update(FRAME_MOTION_INPUT=str((a.output/(input_name+'.json')).resolve()),FRAME_MOTION_RESULT=str((a.output/(name+'.json')).resolve()))
  run=subprocess.run(command,env=env,capture_output=True,text=True,timeout=60,creationflags=subprocess.CREATE_NO_WINDOW)
  (a.output/(name+'.log')).write_text(run.stdout+run.stderr)
  if run.returncode or 'ERROR:' in run.stderr:raise RuntimeError(run.stdout+run.stderr)
  reports[name]=json.loads((a.output/(name+'.json')).read_text())
 for suffix in ('.frames.jsonl','.observations.jsonl'):assert (a.output/('first.json'+suffix)).read_bytes()==(a.output/('repeat.json'+suffix)).read_bytes()
 frames=[json.loads(x) for x in (a.output/'first.json.frames.jsonl').read_text().splitlines()];control_frames=[json.loads(x) for x in (a.output/'control.json.frames.jsonl').read_text().splitlines()]
 assert len(frames)==len(control_frames)
 accepted=[set(),set()]
 for line in (a.output/'first.json.observations.jsonl').read_text().splitlines():
  ob=json.loads(line)
  if ob.get('accepted'):
   event=events[ob['event']];assert ob['capture']==event['capture'] and ob['stamp']==event['stamp']
   assert any(h['side']==ob['side'] and h['source_id']==ob['id'] for h in event['hands_by_variant'][ob['variant']])
   accepted[ob['variant']].add((ob['side'],ob['id'],ob['capture']))
 for r,c in zip(frames,control_frames):
  assert r['now']==c['now'] and r['side']==c['side'] and r['values'][0]==c['values'][0]==c['values'][1]
  for variant,v in enumerate(r['values']):
   if v:
    assert (r['side'],v['id'],v['capture']) in accepted[variant] and 0<=v['age']<.17 and 0<=v['prediction']<=.025+1e-8
    assert abs(r['now']-v['capture']-v['age'])<1e-8 and abs(v['opacity']-(1-np.clip((v['age']-.1)/.07,0,1)))<1e-7
 summary=[]
 for variant in (0,1):
  previous={};position=[];rotation=[];angle=[];visible=0
  for r in frames:
   v=r['values'][variant];side=r['side']
   if not v:previous.pop(side,None);continue
   visible+=1;old=previous.get(side)
   if old and old['id']==v['id']:
    position.append(float(np.linalg.norm(np.array(v['position'])-old['position'])*1000));rotation.append(float(np.degrees(2*np.arccos(np.clip(abs(np.dot(v['orientation'],old['orientation'])),0,1)))));angle.append(float(np.max(abs(np.array(v['angles'])-old['angles']))))
   previous[side]=v
  summary.append(dict(visible=visible,position_step_mm=stats(position),rotation_step_degrees=stats(rotation),angle_step_rad=stats(angle),rotations_over45=sum(x>45 for x in rotation)))
 version=subprocess.run([str(a.godot),'--headless','--version'],capture_output=True,text=True,check=True,creationflags=subprocess.CREATE_NO_WINDOW).stdout.strip()
 result=dict(scope=__doc__,godot=version,events=len(events),selected_input_sets=len(byraw),roundtrip_mm=roundtrip,repeat_exact=True,control_samples_exact=len(frames),summary=summary,replay=reports['first'],provenance=dict(head_evidence=headproof,run_sha256=sha(a.run/'result.json'),input_sha256=sha(a.output/'input.json'),frames_sha256=sha(a.output/'first.json.frames.jsonl'),sources={n:sha(a.output/n) for n in ['hand_motion.gd','baseline_motion.gd','replay_side_motion.gd']},runner_sha256=sha(Path(__file__))),limitations=['Only six selected windows contain hand observations; other recorded sets have empty hands. Not full recording tracking coverage.','Existing source/side IDs, duplicate-packet behavior and head-context losses preserved.','Model fitting and pose history differ from live full pipeline; motion script itself is unchanged.','Saved requested-time PC head estimates and assumed transport/ticks. Not Frame speed, latency or headset acceptance.'])
 (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(summary=summary,control_samples_exact=len(frames),repeat_exact=True),indent=2))
if __name__=='__main__':main()
