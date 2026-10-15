"""Compare filtered finger curl against the same replay's unfiltered estimate.

This estimates software-filter phase shift, not physical tracking latency.
Input is ft-handreplay --poses with FRAME_HANDS_POSE_RAW_APPEND=1.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def curl(points):
    wrist=points[:,0:1,:]
    tips=np.linalg.norm(points[:,[8,12,16,20],:]-wrist,axis=2)
    knuckles=np.linalg.norm(points[:,[5,9,13,17],:]-wrist,axis=2)
    return np.mean(tips/np.maximum(knuckles,0.001),axis=1)


def analyze(path):
    tracks={}
    for line in path.read_text().splitlines():
        fields=line.split()
        if len(fields)!=194: raise ValueError('Expected model, filtered and raw 21-joint sets')
        values=np.array(list(map(float,fields[5:])),dtype=float).reshape(3,21,3)
        if not np.isfinite(values).all(): continue
        tracks.setdefault(int(fields[1]),[]).append((float(fields[0]),values[1],values[2]))
    results=[]
    for identity,rows in tracks.items():
        if len(rows)<100: continue
        t=np.array([row[0] for row in rows])
        if np.max(np.diff(t))>0.1: continue
        filtered=curl(np.stack([row[1] for row in rows]))
        raw=curl(np.stack([row[2] for row in rows]))
        mask=t>=t[0]+0.25
        errors=[]
        for lag_ms in range(0,202,2):
            shifted=np.interp(t[mask]-lag_ms/1000,t,raw)
            errors.append(float(np.sqrt(np.mean((filtered[mask]-shifted)**2))))
        best=int(np.argmin(errors))
        results.append(dict(track_id=identity,samples=len(rows),raw_curl_range=float(np.ptp(raw)),
            best_fit_filter_lag_ms=best*2,zero_lag_curl_rmse=errors[0],best_lag_curl_rmse=errors[best],
            optimum_at_search_boundary=best in (0,len(errors)-1)))
    return dict(kind='Filter lag relative to raw estimated curl; not ground truth or physical latency',
                source=str(path.name),tracks=results,physical_accuracy_verified=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('poses',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.poses)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
