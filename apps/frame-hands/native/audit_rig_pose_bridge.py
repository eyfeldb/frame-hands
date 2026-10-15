"""Compare the standard-library native pose bridge with frozen exact-HMD-join output."""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np
from rig_pose_bridge import RigPoseBridge,profile_limits
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE.parent/'exploratory'))
from frame_data import read_audit,capture_poses

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('session','native','output'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args()
 if '143912Z' in str(a.session):raise ValueError('Accepted holdout excluded')
 a.output.mkdir(parents=True,exist_ok=False)
 audit=read_audit(a.session/'frame/live-views.txt');heads=capture_poses(a.session)
 records=[json.loads(x) for x in (a.native/'palm-profile-viewer-v2/poses.jsonl').read_text().splitlines()];fits=np.loadtxt(a.native/'rig-stage-v2/recorded-output.txt')
 if len(records)!=len(fits) or fits.shape[1]!=104:raise ValueError('Incomplete input')
 grouped={}
 for r,fit in zip(records,fits):
  if fit[0]!=1:raise ValueError('Unexpected unavailable fit')
  grouped.setdefault(r['raw_ns'],[]).append(dict(source_id=r['source_id'],side=r['side'],status='fitted',wrist_head_mm=fit[14:21].tolist(),joint_angles=fit[21:41].tolist()))
 bridge=RigPoseBridge([profile_limits(a.native/f'palm-profile-v1/side{s}.fhrig',s) for s in (0,1)])
 reference=json.loads((a.native/'presentation-native-v1/native-rig-motion.json').read_text());first=min(x['mono_ns'] for x in audit.values())
 events=[];missing=0;hands=0;max_difference=0.;max_angles=0.;ambiguities=0
 if len(audit)!=len(reference['events']):raise ValueError('Reference length differs')
 with (a.output/'packets.jsonl').open('x') as out:
  for sequence,((raw,entry),expected) in enumerate(zip(sorted(audit.items()),reference['events']),1):
   capture=entry['mono_ns'];snapshot=dict(version=3,clock='CLOCK_MONOTONIC',space='capture_head_mm',sequence=sequence,capture_raw_ns=raw,capture_mono_ns=capture,published_mono_ns=capture+20000000,hands=grouped.get(raw,[]));valid=capture in heads
   head=dict(valid=False)
   if valid:
    m=heads[capture].copy();m[:3,3]*=.001
    head=dict(valid=True,capture_mono_ns=capture,queried_mono_ns=capture+45000000,capture=m[:3].reshape(-1).tolist(),current=m[:3].reshape(-1).tolist())
   packet=bridge.convert(snapshot,head,capture+50000000,1000+(capture-first)/1e9+.05)
   out.write(json.dumps(packet,separators=(',',':'))+'\n');missing+=not packet['context_valid'];hands+=len(packet['hands']);ambiguities+=sum(x['reason']=='ambiguous_side' for x in packet['unavailable_sides'])
   if packet['context_valid']!=expected['valid']:raise ValueError('Context changed')
   actual={h['source_id']:h for h in packet['hands']};control={h['source_id']:h for h in expected['hands']}
   if actual.keys()!=control.keys():raise ValueError('Changed source availability')
   for source,h in actual.items():
    old=control[source]
    if h['side']!=old['side']:raise ValueError('Changed side')
    max_difference=max(max_difference,float(np.max(np.abs(np.asarray(h['wrist_world_m'])-old['wrist_world_m']))))
    max_angles=max(max_angles,float(np.max(np.abs(np.asarray(h['joint_angles'])-old['joint_angles']))))
   t=(capture-first)/1e9;events.append(dict(now=t+.05,capture=t,stamp=t,hands=packet['hands'],valid=packet['context_valid']))
 if max_difference>1e-5 or max_angles>1e-4:raise ValueError('World pose differs from prior exact-join path')
 fixture=dict(name='native_bridge_frozen_poses_assumed_50ms_delivery',scope='Saved native rig stage outputs and exact recorded capture HMD join; simulated publication/query clocks. Current head equals capture head only in this offline fixture.',events=events)
 (a.output/'events.json').write_text(json.dumps(fixture,separators=(',',':'))+'\n')
 report=dict(scope=__doc__,frames=len(events),hands=hands,missing_head_context=missing,ambiguous_sides=ambiguities,max_wrist_matrix_element_difference=max_difference,max_joint_angle_difference_rad=max_angles,availability_and_sides_equal=True,provenance=dict(session=a.session.name,source_sha256=sha(Path(__file__)),bridge_sha256=sha(BASE/'rig_pose_bridge.py'),fitted_sha256=sha(a.native/'rig-stage-v2/recorded-output.txt'),reference_sha256=sha(a.native/'presentation-native-v1/native-rig-motion.json'),events_sha256=sha(a.output/'events.json')),limitations=['Standard-library bridge math executes on host; no pose_service, IPC latency, native renderer, camera or XR used.','Simulated20ms publish/45ms query/50ms delivery; current head equals capture head for fixture only. No physical timing claim.','Numerical comparison inherits small native-stage vs separated-fitter differences; no new accuracy claim.','Generic renderer mesh is not yet adapted to experimental palm geometry.'])
 (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
