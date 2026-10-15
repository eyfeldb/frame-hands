"""Paired causal angle initialization on frozen original-detector observations; no live inference."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    return dict(p50=float(np.percentile(values,50)),p95=float(np.percentile(values,95)),maximum=float(np.max(values)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('executable','profiles','recording','session','cameras','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--mode',choices=['causal','guarded'],default='causal')
    args=parser.parse_args()
    if '143912Z' in str(args.session):raise ValueError('Accepted holdout excluded')
    if json.loads((args.recording/'summary.json').read_text())['session']!=args.session.name:raise ValueError('Session mismatch')
    first=min(int(line.split()[1]) for line in (args.session/'frame/live-views.txt').read_text().splitlines() if line.startswith('set '))
    args.output.mkdir(parents=True,exist_ok=False)
    records=[json.loads(line) for line in (args.recording/'poses.jsonl').read_text().splitlines()]
    payload=np.zeros((len(records),199));baseline=np.zeros((len(records),99))
    sides=np.array([row['side'] for row in records])
    for side in (0,1):
        ids=np.flatnonzero(sides==side)
        payload[ids]=np.loadtxt(args.recording/f'side{side}-pixel-input.txt')
        baseline[ids]=np.loadtxt(args.recording/f'side{side}-pixel-fit.txt')
    timestamps=np.array([row['raw_ns'] for row in records],dtype=np.int64)
    if np.any(np.diff(timestamps)<0):raise ValueError('Nonchronological source')
    packed=np.column_stack([timestamps,[row['source_id'] for row in records],payload])
    def run(name,data,mode=None,expected_code=0):
        mode=mode or args.mode
        inp=args.output/(name+'-input.txt');out=args.output/(name+'-output.txt')
        np.savetxt(inp,data,fmt='%.17g')
        call=subprocess.run([str(args.executable.resolve()),str((args.profiles/'side0.fhrig').resolve()),
                             str((args.profiles/'side1.fhrig').resolve()),str(args.cameras.resolve()),str(inp.resolve()),str(out.resolve()),mode],capture_output=True,text=True,timeout=90)
        if call.returncode!=expected_code:raise ValueError(name+': '+call.stderr)
        if expected_code:return
        result=np.loadtxt(out,ndmin=2)
        if result.shape!=(len(data),104) or not np.isfinite(result).all() or np.any(result[:,5]>result[:,4]+1e-9):raise ValueError('Invalid output '+name)
        return result
    fresh=run('fresh',packed,'fresh');causal=run('causal',packed)
    np.testing.assert_array_equal(fresh[:,8:99],baseline[:,8:99])
    np.testing.assert_array_equal(causal[:,:2],fresh[:,:2]);np.testing.assert_array_equal(causal[:,8],fresh[:,8])
    # Replaying a prefix must be exactly invariant to all later observations.
    prefix=run('prefix',packed[:128]);columns=np.r_[0:7,8:104]
    np.testing.assert_array_equal(prefix[:,columns],causal[:128,columns])
    # Force source change, long gap and a missing observation, keeping geometry fixed.
    ids=np.flatnonzero(sides==sides[0])[:3];two=packed[ids[:2]].copy();two[:,0]=[1000000000,1033333333];two[:,1]=5
    changed=two.copy();changed[1,1]=6
    gap=two.copy();gap[1,0]=1200000000
    controls=[]
    for name,data in [('source_reset',changed),('gap_reset',gap)]:
        actual=run(name,data);reference=run(name+'-fresh',data,'fresh')
        if actual[1,99]!=0 or actual[1,100]!=0:raise ValueError('History survived '+name)
        np.testing.assert_array_equal(actual[1,8:99],reference[1,8:99]);controls.append(name)
    missing=packed[ids].copy();missing[:,0]=[1000000000,1033333333,1066666666];missing[:,1]=5;missing[1,33::4]=0
    result=run('missing_reset',missing)
    if result[1,0]!=0 or result[2,99]!=0:raise ValueError('Missing-frame state retained')
    controls.append('missing_reset')
    duplicate=two.copy();duplicate[1,0]=duplicate[0,0];run('duplicate_timestamp',duplicate,expected_code=7)
    negative=two.copy();negative[1,1]=-1;run('negative_source',negative,expected_code=7)
    limits=[np.loadtxt(args.profiles/f'side{s}.fhrig',skiprows=1,max_rows=20)[:,6:8] for s in (0,1)]
    for side in (0,1):
        angles=causal[sides==side,16:36]
        if not ((angles>=limits[side][:,0]-1e-8)&(angles<=limits[side][:,1]+1e-8)).all():raise ValueError('Joint bounds')
    delta=causal[:,5]-fresh[:,5]
    if args.mode=='guarded' and np.any(delta>1e-8):raise ValueError('Completed-fit guard regressed objective')
    displacement=np.max(np.linalg.norm((causal[:,36:99]-fresh[:,36:99]).reshape(-1,21,3),axis=2),axis=1)
    groups={}
    for name,mask in [('all',np.ones(len(records),bool)),('later',timestamps>=first+35000000000)]:
        groups[name]=dict(hands=int(mask.sum()),eligible=int(causal[mask,99].sum()),used=int(causal[mask,100].sum()),
                         objective_improved_over_1=int((delta[mask]<-1).sum()),objective_worse_over_1=int((delta[mask]>1).sum()),
                         objective_delta=stats(delta[mask]),max_landmark_displacement_mm=stats(displacement[mask]),
                         fresh_pixel_rmse=stats(fresh[mask,6]),causal_pixel_rmse=stats(causal[mask,6]))
    report=dict(scope=__doc__,mode=args.mode,session=args.session.name,session_first_raw_ns=first,groups=groups,extra_pixel_fits=int((causal[:,103]-1).sum()),reset_controls=controls,rejected=['duplicate_timestamp','negative_source'],prefix_exact=True,
                fresh_frozen_control_exact=True,fresh_host_us=stats(fresh[:,7]),causal_host_us=stats(causal[:,7]),
                largest_worsening_index=int(np.argmax(delta)),largest_improvement_index=int(np.argmin(delta)),
                provenance=dict(executable_sha256=sha(args.executable),input_sha256=sha(args.output/'causal-input.txt'),
                                profile_sha256={str(s):sha(args.profiles/f'side{s}.fhrig') for s in (0,1)},
                                sources={name:sha(Path(__file__).parent/name) for name in ('causal_rig_fit.h','evaluate_causal_rig.cpp','audit_causal_rig.py','fit_rig_pixels.h')}),
                limitations=['Frozen detector identities and current 3D initializer; not identity recovery, output hold or motion prediction.',
                             'Only previous local angles reused; wrist is current-frame. Prior remains anchored to fresh pose in both branches.',
                             'Lower image objective is not independent pose accuracy. No live or native performance claim.',
                             '100ms gap bound chosen before replay; same recording as palm profile experiment. Latest holdout excluded.'])
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
