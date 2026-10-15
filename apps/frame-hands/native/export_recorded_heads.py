"""Export exact sequence-joined saved HMD estimates for camera-only tracker replay."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from audit_native_lengths import sha
from frame_data import read_audit,capture_poses

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if '143912Z' in str(a.session):raise ValueError('Latest accepted holdout excluded')
 audit=read_audit(a.session/'frame/live-views.txt');heads=capture_poses(a.session);a.output.mkdir(parents=True,exist_ok=False)
 recording_hash=sha(a.session/'frame/sets.bin');valid=0
 with (a.output/'heads.txt').open('x') as out:
  out.write('FHHEAD 1 CLOCK_MONOTONIC raw_world_mm '+recording_hash+'\n')
  for raw,row in sorted(audit.items()):
   head=heads.get(row['mono_ns']);ok=head is not None
   transform=np.eye(4) if head is None else head
   q=Rotation.from_matrix(transform[:3,:3]).as_quat()[[3,0,1,2]]
   out.write(' '.join(map(str,[raw,row['mono_ns'],int(ok),*q,*transform[:3,3]]))+'\n');valid+=ok
 report=dict(version=1,scope=__doc__,recording_sha256=recording_hash,heads_sha256=sha(a.output/'heads.txt'),frames=len(audit),valid_frames=valid,clock='CLOCK_MONOTONIC',space='raw_world_mm',source_sha256={str(p):sha(p) for p in [Path(__file__),Path(__file__).parent.parent/'exploratory/frame_data.py',a.session/'frame/live-views.txt',a.session/'relay-trace.bin',a.session/'sampler-trace.bin']},limitations=['Saved SteamVR requested-time estimates, not optical ground truth or Frame-local runtime validation.','Lookup requires exact camera raw timestamp; no interpolation or latest-pose substitution. Missing joins remain explicitly invalid.','Single recording coordinate context; reference-reset events were not recorded.','No query/return clock is invented; this file is admissible only in explicit replay mode.'])
 (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(frames=len(audit),valid_frames=valid,heads_sha256=report['heads_sha256'],manifest_sha256=sha(a.output/'manifest.json'))))
if __name__=='__main__':main()
