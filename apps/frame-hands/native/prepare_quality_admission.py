"""Prepare an isolated image-residual gate replay with unchanged native presentation."""
import argparse,json,sys,shutil
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'exploratory'))
from frame_data import read_audit,capture_poses
from audit_native_lengths import sha
from rig_pose_bridge import RigPoseBridge,profile_limits

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('native','session','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args()
 if '143912Z' in str(a.session):raise ValueError('Latest holdout excluded')
 a.output.mkdir(parents=True,exist_ok=False)
 audit=read_audit(a.session/'frame/live-views.txt');heads=capture_poses(a.session)
 fits=np.loadtxt(a.native/'whole-pose-history-v1/output.txt');records=[json.loads(x) for x in (a.native/'palm-profile-viewer-v2/poses.jsonl').read_text().splitlines()]
 assert fits.shape==(len(records),191) and np.isfinite(fits).all();by_raw={}
 for i,(row,fit) in enumerate(zip(records,fits)):
  assert [row['raw_ns'],row['source_id'],row['side']]==list(fit[:3])
  by_raw.setdefault(row['raw_ns'],[]).append(dict(source_id=row['source_id'],side=row['side'],status='fitted',wrist_head_mm=fit[101:108].tolist(),joint_angles=fit[108:128].tolist(),fit_rmse_px=float(fit[100]),fit_index=i))
 bridge=RigPoseBridge([profile_limits(a.native/f'palm-profile-v1/side{s}.fhrig',s) for s in (0,1)])
 first=min(e['mono_ns'] for e in audit.values());events=[];counts=dict(input_fits=len(records),routed_fits=0,over9=0,invalid_contexts=0,ambiguous_sides=0)
 for seq,(raw,entry) in enumerate(sorted(audit.items()),1):
  capture=entry['mono_ns'];matrix=heads.get(capture);h=dict(valid=False)
  if matrix is not None:
   matrix=matrix.copy();matrix[:3,3]*=.001;h=dict(valid=True,capture_mono_ns=capture,queried_mono_ns=capture+45_000_000,capture=matrix[:3].reshape(-1).tolist(),current=matrix[:3].reshape(-1).tolist())
  hands=by_raw.get(raw,[]);snapshot=dict(version=3,clock='CLOCK_MONOTONIC',space='capture_head_mm',sequence=seq,capture_raw_ns=raw,capture_mono_ns=capture,published_mono_ns=capture+20_000_000,hands=hands)
  packet=bridge.convert(snapshot,h,capture+50_000_000,(capture-first)/1e9+.05)
  for hand in packet['hands']:
   source=next(x for x in hands if x['source_id']==hand['source_id']);hand.update(fit_rmse_px=source['fit_rmse_px'],fit_index=source['fit_index']);counts['routed_fits']+=1;counts['over9']+=hand['fit_rmse_px']>9
  counts['invalid_contexts']+=not packet['context_valid'];counts['ambiguous_sides']+=sum(x['reason']=='ambiguous_side' for x in packet['unavailable_sides'])
  sec=(capture-first)/1e9;events.append(dict(now=sec+.05,capture=sec,stamp=sec,raw_ns=raw,hands=packet['hands'],valid=packet['context_valid']))
 result=dict(name='whole_pose_quality_gate_assumed_50ms_delivery',scope='Saved fitted poses with exact recorded HMD joins; assumed50ms delivery and90Hz display. Counterfactual, not measured live latency or accuracy.',events=events,counts=counts)
 (a.output/'events.json').write_text(json.dumps(result,separators=(',',':'))+'\n',encoding='utf-8')
 for original,name in [(BASE.parent/'exploratory/mesh-viewer/hand_motion.gd','baseline_motion.gd'),(BASE/'presentation/hand_motion.gd','hand_motion.gd'),(BASE/'presentation/replay_quality.gd','replay_quality.gd')]:shutil.copy2(original,a.output/name)
 (a.output/'project.godot').write_text('config_version=5\n[application]\nconfig/name="Offline pose admission check"\n[rendering]\nrenderer/rendering_method="gl_compatibility"\n',encoding='utf-8')
 manifest=dict(scope=__doc__,gate='Reject fit image RMSE >9pixels (3x assumed3px noise); no new presence/gesture confidence or extended prediction.',counts=counts,sha256={p.name:sha(p) for p in a.output.iterdir() if p.is_file()},inputs={str(p):sha(p) for p in [a.native/'whole-pose-history-v1/output.txt',a.native/'palm-profile-viewer-v2/poses.jsonl',a.session/'frame/live-views.txt',a.session/'relay-trace.bin',a.session/'sampler-trace.bin']},source_sha256=sha(Path(__file__)))
 (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8');print(json.dumps(counts))
if __name__=='__main__':main()
