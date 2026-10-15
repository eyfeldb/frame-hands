"""Replay recorded source poses through subscriber ordering and head joins, offline."""
import argparse, json, sys
from pathlib import Path
import numpy as np
from rig_pose_subscriber import Subscriber, verified_limits
from audit_rig_pose_bridge import sha
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'exploratory'))
from frame_data import read_audit, capture_poses

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('session','native','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if '143912Z' in str(args.session):raise ValueError('Accepted holdout excluded')
    args.output.mkdir(parents=True,exist_ok=False)
    audit=read_audit(args.session/'frame/live-views.txt');heads=capture_poses(args.session)
    native=args.native
    records=[json.loads(x) for x in (native/'palm-profile-viewer-v2/poses.jsonl').read_text().splitlines()]
    fits=np.loadtxt(native/'rig-stage-v2/recorded-output.txt')
    if len(records)!=len(fits) or fits.shape[1]!=104:raise ValueError('Incomplete fitted poses')
    grouped={}
    for record,fit in zip(records,fits):
        if fit[0]!=1:raise ValueError('Unexpected unfitted pose')
        grouped.setdefault(record['raw_ns'],[]).append(dict(source_id=record['source_id'],side=record['side'],status='fitted',wrist_head_mm=fit[14:21].tolist(),joint_angles=fit[21:41].tolist()))
    state=Subscriber(verified_limits(native/'tracker-rig-v3/profiles',native/'profile-viewer-v2'))
    control=[json.loads(x) for x in (native/'rig-pose-bridge-v1/packets.jsonl').read_text().splitlines()]
    if len(control)!=len(audit):raise ValueError('Mismatched recording')
    first=min(x['mono_ns'] for x in audit.values());queries=0;hands=0;invalid=0
    with (args.output/'trace.jsonl').open('x') as output:
        for sequence,((raw,entry),expected) in enumerate(zip(sorted(audit.items()),control),1):
            capture=entry['mono_ns']
            source=dict(version=3,clock='CLOCK_MONOTONIC',space='capture_head_mm',sequence=sequence,capture_raw_ns=raw,capture_mono_ns=capture,published_mono_ns=capture+20_000_000,hands=grouped.get(raw,[]))
            head=dict(valid=False)
            if capture in heads:
                matrix=heads[capture].copy();matrix[:3,3]*=.001
                head=dict(valid=True,capture_mono_ns=capture,queried_mono_ns=capture+45_000_000,capture=matrix[:3].reshape(-1).tolist(),current=matrix[:3].reshape(-1).tolist())
            def query(stamp):
                nonlocal queries
                if stamp!=capture:raise ValueError('Wrong query clock')
                queries+=1
                return head
            data=json.dumps(source).encode()
            packet,response=state.consume(data,query,lambda:capture+50_000_000,lambda:1000+(capture-first)/1e9+.05)
            if packet!=expected:raise ValueError('Subscriber changed reference packet at '+str(sequence))
            duplicate,_=state.consume(data,query,lambda:capture+55_000_000,lambda:1000+(capture-first)/1e9+.055)
            if duplicate is not None:raise ValueError('Duplicate refreshed output')
            output.write(json.dumps(dict(snapshot=source,head=response,packet=packet))+'\n')
            hands+=len(packet['hands']);invalid+=not packet['context_valid']
    report=dict(scope=__doc__,frames=len(control),hands=hands,invalid_contexts=invalid,queries=queries,duplicates_suppressed=len(control),all_packets_exactly_equal=True,
        source_sha256={name:sha(BASE/name) for name in ('audit_rig_subscriber.py','rig_pose_subscriber.py','rig_pose_bridge.py')},
        inputs_sha256={name:sha(native/name) for name in ('rig-stage-v2/recorded-output.txt','palm-profile-viewer-v2/poses.jsonl','rig-pose-bridge-v1/packets.jsonl','profile-viewer-v2/profile-rigs.json')},
        trace_sha256=sha(args.output/'trace.jsonl'),limitations=['Saved fitted poses, not image inference.','Simulated 20/45/50 ms publication/query/delivery; current head equals capture head.','Separate pipe tests use a host helper fixture; no actual OpenVR/device timing or wearable quality claim.'])
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
