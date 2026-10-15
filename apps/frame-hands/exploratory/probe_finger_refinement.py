"""Offline fixed-wrist finger refinement on declared raw-frame cases."""
import argparse,hashlib,json,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
from frame_data import Recording,load_calibration
from live_backend import Backend
from run_mesh_trial import pose_matrix
from probe_roi_presence import PresenceModel,verify_hand
from finger_refinement import fit_fingers,apply_angles,PARAMETERS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['protocol','upstream','models','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.upstream));a.output.mkdir(parents=True,exist_ok=False)
    protocol=json.loads(a.protocol.read_text());shutil.copy2(a.protocol,a.output/'protocol.json')
    snapshot=a.output/'snapshot';snapshot.mkdir()
    for file in Path(__file__).parent.glob('*.py'):shutil.copy2(file,snapshot/file.name)
    verifier=PresenceModel(a.models);contexts={};selected={};groups={}
    for case in protocol['cases']:groups.setdefault(case['baseline'],[]).append(case)
    for reference,cases in groups.items():
        for case in cases:
            if case['session'] not in contexts:
                session=Path(case['session']);rec=Recording(session/'frame/sets.bin');cal=session/'calibration'
                backend=Backend(a.upstream,load_calibration(cal if cal.exists() else session/'frame/snapshot/device/persist'))
                contexts[case['session']]=(rec,backend)
        best={c['name']:float('inf') for c in cases}
        with Path(reference).open() as stream:
            for line in stream:
                row=json.loads(line)
                for case in cases:
                    rec,_=contexts[case['session']];seconds=(row['capture_raw_ns']-rec.index[0][0])/1e9
                    distance=abs(seconds-case['seconds'])
                    if distance<best[case['name']]:best[case['name']]=distance;selected[case['name']]=row
    reports=[]
    for case in protocol['cases']:
        rec,backend=contexts[case['session']];row=selected[case['name']];stamp=row['capture_raw_ns']
        seconds=(stamp-rec.index[0][0])/1e9
        if abs(seconds-case['seconds'])>.2 or not row['capture_pose']['valid']:raise ValueError('No nearby valid exact capture')
        hand=next(h for h in row['hands'] if h['side']==case['side'])
        _,actual,images=next(rec.frames(seconds-.000001,seconds+.000001))
        if actual!=stamp:raise ValueError('Wrong raw capture')
        world=pose_matrix(row['capture_pose']['capture'])
        before,_=verify_hand(verifier,backend.hand,hand,backend.cameras,images,world)
        cameras={name:cam.copy(camera_to_world_xf=world@cam.camera_to_world_xf) for name,cam in backend.cameras.items()}
        start=time.perf_counter();angles,debug=fit_fingers(backend.hand,hand,cameras,before);elapsed=(time.perf_counter()-start)*1000
        candidate=apply_angles(backend,hand,angles) if angles is not None else hand
        if angles is not None:
            assert candidate['wrist_world_mm']==hand['wrist_world_mm'] and candidate['source_id']==hand['source_id'] and candidate['side']==hand['side']
            assert candidate['joint_angles'][20:]==hand['joint_angles'][20:]
        after,_=verify_hand(verifier,backend.hand,candidate,backend.cameras,images,world) if angles is not None else (before,None)
        out=dict(row,hands=[candidate if h['side']==case['side'] else h for h in row['hands']],finger_refinement_hypothesis=True)
        folder=a.output/case['name'];folder.mkdir();(folder/'baseline.jsonl').write_text(json.dumps(row,allow_nan=False)+'\n');(folder/'candidate.jsonl').write_text(json.dumps(out,allow_nan=False)+'\n')
        report=dict(case=case,seconds=seconds,stamp=stamp,changed=angles is not None,fit_ms=elapsed,fit=debug,before_verifier=before,after_verifier=after)
        (folder/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');reports.append(report)
        print(json.dumps(dict(case=case['name'],seconds=seconds,changed=angles is not None,fit_ms=elapsed,fit=debug)),flush=True)
    summary=dict(scope=__doc__+' Numerical fit improvement is against the fitted observations, not physical accuracy. Hypotheses only, no temporal or live validation.',parameters=PARAMETERS,torch=torch.__version__,ncnn=verifier.version,protocol_sha256=hashlib.sha256(a.protocol.read_bytes()).hexdigest(),source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in snapshot.glob('*.py')},reports=reports)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
