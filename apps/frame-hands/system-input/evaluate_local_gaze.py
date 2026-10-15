"""Offline development comparison only: combines formerly withheld targets as training.

Results cannot activate a profile. Any selected model needs NEW validation targets.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import calibrate_gaze as cal


def fit_local(points, width, ridge):
    affine=cal.fit(points,3)
    matrix=[[math.exp(-sum((p['raw'][k]-q['raw'][k])**2 for k in (0,1))/(2*width*width))+(ridge if i==j else 0)
             for j,q in enumerate(points)] for i,p in enumerate(points)]
    residual=[[p['target'][k]-cal.correct(p['raw'],affine)[k] for p in points] for k in (0,1)]
    weights=[cal.solve(matrix,r) for r in residual]
    return dict(affine=affine,centers=[p['raw'] for p in points],weights=weights,width=width,ridge=ridge)


def predict(raw, model):
    result=cal.correct(raw,model['affine'])
    kernel=[math.exp(-sum((raw[k]-center[k])**2 for k in (0,1))/(2*model['width']**2)) for center in model['centers']]
    return [result[k]+sum(a*b for a,b in zip(kernel,model['weights'][k])) for k in (0,1)]


def compare(points):
    reports=[]
    for terms in (1,3,6):
        errors=[cal.error(cal.correct(p['raw'],cal.fit(points[:i]+points[i+1:],terms)),p['target']) for i,p in enumerate(points)]
        reports.append(dict(model=f'polynomial_{terms}',mean_deg=statistics.mean(errors),median_deg=statistics.median(errors),worst_deg=max(errors)))
    for width in (4,8,12):
        for ridge in (.1,1):
            errors=[cal.error(predict(p['raw'],fit_local(points[:i]+points[i+1:],width,ridge)),p['target']) for i,p in enumerate(points)]
            reports.append(dict(model='affine_plus_local',width_deg=width,ridge=ridge,mean_deg=statistics.mean(errors),median_deg=statistics.median(errors),worst_deg=max(errors)))
    return dict(development_only=True,independent_validation=False,targets=len(points),candidates=reports)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('traces',nargs='+',type=Path);parser.add_argument('--output',type=Path);args=parser.parse_args()
    points=[];sessions=set()
    for path in args.traces:
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        if not rows[-1].get('completed'):raise ValueError('Incomplete trace')
        sessions.add(rows[0]['session'])
        points.extend(cal.summarize(r) for r in rows if 'point' in r)
    if len(sessions)!=1:raise ValueError('Cannot combine different runtime sessions')
    result=compare(points);text=json.dumps(result,indent=2)+'\n'
    if args.output:args.output.write_text(text)
    print(text,end='')
