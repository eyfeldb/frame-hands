"""Convert frozen native rig fits to tracking-world poses using exact recorded HMD joins."""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE.parent/'exploratory'))
from frame_data import read_audit,capture_poses

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('session','recording','fitted','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if '143912Z' in str(a.session):raise ValueError('Accepted holdout excluded')
    if a.output.exists():raise FileExistsError(a.output)
    audit=read_audit(a.session/'frame/live-views.txt');head=capture_poses(a.session)
    records=[json.loads(x) for x in (a.recording/'poses.jsonl').read_text().splitlines()];fits=np.loadtxt(a.fitted)
    if len(fits)!=len(records) or fits.shape[1]<99 or not np.isfinite(fits).all():raise ValueError('Invalid fits')
    by_stamp={}
    for row,fit in zip(records,fits):
        if row['side']!=fit[8]:raise ValueError('Changed side')
        wrist=np.eye(4);wrist[:3,:3]=Rotation.from_quat(fit[[10,11,12,9]]).as_matrix();wrist[:3,3]=fit[13:16]
        by_stamp.setdefault(row['raw_ns'],[]).append((row,fit,wrist))
    first=min(x['mono_ns'] for x in audit.values());events=[];missing=0;hands=0
    for raw,entry in sorted(audit.items()):
        stamp=entry['mono_ns'];seconds=(stamp-first)/1e9;valid=stamp in head;current=[]
        if not valid:missing+=1
        for row,fit,wrist in by_stamp.get(raw,[]):
            if not valid:continue
            world=head[stamp]@wrist;world[:3,3]*=.001
            current.append(dict(side=row['side'],source_id=row['source_id'],wrist_world_m=world.tolist(),joint_angles=fit[16:36].tolist()));hands+=1
        events.append(dict(now=seconds+.05,capture=seconds,stamp=seconds,hands=current,valid=valid))
    result=dict(name='native_rig_saved_poses_assumed_50ms_delivery',scope='Frozen native fit poses, exact capture-time HMD joins, assumed50ms delivery and90Hz sampling; not measured live latency or physical truth.',events=events,missing_head_joins=missing,hands=hands,provenance=dict(session=a.session.name,fitted_sha256=sha(a.fitted),record_metadata_sha256=sha(a.recording/'poses.jsonl'),inputs={n:sha(a.session/n) for n in ('frame/live-views.txt','relay-trace.bin','sampler-trace.bin')},source_sha256=sha(Path(__file__))))
    a.output.write_text(json.dumps(result,separators=(',',':'))+'\n');print(json.dumps(dict(events=len(events),hands=hands,missing_head_joins=missing)))
if __name__=='__main__':main()
