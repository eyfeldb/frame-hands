"""Collect guided public gaze targets, fit on nine points, validate on five others.

No third-party Python dependencies, clicks, cameras, Steam settings or services.
"""
import argparse
import datetime
import json
import math
from pathlib import Path
import signal
import statistics
import subprocess
import time


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered)-1, int((len(ordered)-1)*q))]


def features(a):
    x, y = (v/20 for v in a)
    return [1, x, y, x*x, x*y, y*y]


def direction(a):
    y, p = map(math.radians, a)
    return [-math.sin(y)*math.cos(p), math.sin(p), -math.cos(y)*math.cos(p)]


def error(a, b):
    return math.degrees(math.acos(max(-1, min(1, sum(x*y for x,y in zip(direction(a), direction(b)))))))


def solve(matrix, vector):
    n = len(vector)
    a = [list(row)+[v] for row,v in zip(matrix,vector)]
    for col in range(n):
        pivot = max(range(col,n), key=lambda i:abs(a[i][col]))
        if abs(a[pivot][col]) < 1e-9:
            raise ValueError('Degenerate gaze coverage')
        a[col],a[pivot] = a[pivot],a[col]
        scale = a[col][col]
        a[col] = [v/scale for v in a[col]]
        for i in range(n):
            if i != col:
                scale = a[i][col]
                a[i] = [v-scale*w for v,w in zip(a[i],a[col])]
    return [a[i][-1] for i in range(n)]


def fit(points, terms, ridge=.002):
    x = [features(p['raw'])[:terms] for p in points]
    matrix = [[sum(row[i]*row[j] for row in x) for j in range(terms)] for i in range(terms)]
    # Weak ridge only on higher-order terms; offset/affine remain identifiable.
    for i in range(3,terms):
        matrix[i][i] += ridge
    return [solve(matrix, [sum(row[i]*(p['target'][axis]-p['raw'][axis]) for row,p in zip(x,points)) for i in range(terms)])+[0]*(6-terms) for axis in (0,1)]


def correct(raw, coefficient):
    f = features(raw)
    return [raw[k]+sum(a*b for a,b in zip(f,coefficient[k])) for k in (0,1)]


def summarize(row):
    samples = row['samples']
    if len(samples)<45 or len(samples)<.7*row['total_polls']:
        raise ValueError(f"Point {row['point']}: insufficient valid eye polls")
    if any(len(s)!=4 or any(not math.isfinite(v) or abs(v)>70 for v in s) for s in samples):
        raise ValueError('Invalid gaze sample')
    center = [statistics.median(s[k] for s in samples) for k in range(4)]
    spread = percentile([error(s[:2],center[:2]) for s in samples], .9)
    if spread>1.2:
        raise ValueError(f"Point {row['point']}: fixation spread {spread:.2f} degrees; repeat while looking steadily")
    return dict(point=row['point'], raw=center[:2],target=center[2:],spread_deg=spread,samples=samples)


def plausible(coefficient, bounds):
    # Reject excessive corrections or a folded/magnified angular map across the fit area.
    for i in range(9):
        for j in range(9):
            a=[bounds[0][0]+i*(bounds[0][1]-bounds[0][0])/8,bounds[1][0]+j*(bounds[1][1]-bounds[1][0])/8]
            b=correct(a,coefficient)
            if any(abs(b[k]-a[k])>12 for k in (0,1)):
                return False
            dx=correct([a[0]+.01,a[1]],coefficient)
            dy=correct([a[0],a[1]+.01],coefficient)
            ux,uy=(dx[0]-b[0])*100,(dx[1]-b[1])*100
            vx,vy=(dy[0]-b[0])*100,(dy[1]-b[1])*100
            if not (.15<ux*vy-uy*vx<4 and .3<ux<2.5 and .3<vy<2.5):
                return False
    return True


def analyze(rows, fixed=None):
    meta=rows[0]
    if not meta.get('session') or not rows[-1].get('completed'):
        raise ValueError(rows[-1].get('reason','Incomplete collection or unknown eye-tracker session'))
    measurements=[r for r in rows if 'point' in r]
    if sorted(r['point'] for r in measurements)!=list(range(14)):
        raise ValueError('Require exactly nine calibration and five validation targets')
    if any(r['validation']!=(r['point']>=9) for r in measurements):
        raise ValueError('Incorrect train/validation partition')
    training=[summarize(r) for r in measurements if not r['validation']]
    validation=[summarize(r) for r in measurements if r['validation']]
    bounds=[[min(p['raw'][k] for p in training),max(p['raw'][k] for p in training)] for k in (0,1)]
    if bounds[0][1]-bounds[0][0]<18 or bounds[1][1]-bounds[1][0]<10:
        raise ValueError('Insufficient gaze coverage; eye signal may be stuck or targets were not followed')
    # Select complexity ONLY with leave-one-target-out training predictions.
    candidates=[]
    for terms in (1,3,6):
        try:
            coefficient=fit(training,terms)
            cv=[error(correct(p['raw'],fit(training[:i]+training[i+1:],terms)),p['target']) for i,p in enumerate(training)]
            if plausible(coefficient,bounds):
                candidates.append(dict(terms=terms,coefficient=coefficient,cv_mean_deg=statistics.mean(cv)))
        except ValueError:
            continue
    if not candidates:
        raise ValueError('No stable correction model')
    chosen=candidates[0]
    for candidate in candidates[1:]:
        if candidate['cv_mean_deg']<chosen['cv_mean_deg']*.85 and chosen['cv_mean_deg']-candidate['cv_mean_deg']>.1:
            chosen=candidate
    if fixed is not None:
        # A development-data choice must face a NEW holdout collection. The fixed
        # coefficients are computed before that collection, never refitted to it.
        if not plausible(fixed['coefficient'],bounds):
            raise ValueError('Refinement map exceeds correction or Jacobian limits')
        chosen=fixed
    before=[];after=[];points=[]
    for p in validation:
        if any(not bounds[k][0]-2<=p['raw'][k]<=bounds[k][1]+2 for k in (0,1)):
            raise ValueError('Validation gaze lies outside calibration coverage')
        raw=[error(s[:2],s[2:]) for s in p['samples']]
        fixed=[error(correct(s[:2],chosen['coefficient']),s[2:]) for s in p['samples']]
        before.extend(raw);after.extend(fixed)
        points.append(dict(point=p['point'],raw_median_deg=statistics.median(raw),corrected_median_deg=statistics.median(fixed),spread_deg=p['spread_deg']))
    old,new=statistics.median(before),statistics.median(after)
    worst=max(p['corrected_median_deg'] for p in points)
    accepted=new<=1.25 and worst<=2 and percentile(after,.9)<=2.5 and old-new>=.15 and new<=old*.9
    return dict(accepted=accepted,session=meta['session'],created=meta['created'],terms=chosen['terms'],coefficient=chosen['coefficient'],bounds=bounds,
                raw_median_deg=old,corrected_median_deg=new,corrected_p90_deg=percentile(after,.9),worst_target_median_deg=worst,
                candidates=[{k:v for k,v in c.items() if k!='coefficient'} for c in candidates],validation=points,
                selection='fixed_regularized_refinement_new_holdout' if fixed else 'training_only_cross_validation',
                note='Engineering acceptance thresholds; not pixel accuracy or a physical sensor-rate measurement. Validate again after adjusting the headset.')


def write_profile(report, path, preview=False):
    if not preview and not report['accepted']:
        raise ValueError('A failed validation cannot activate a profile')
    # Preview files have a distinct magic and cannot be loaded by the input bridge.
    lines=['FHGAZE_PREVIEW1' if preview else 'FHGAZE1', report['session'],str(report['created']),str(report['terms'])]
    lines += [' '.join(format(v,'.12g') for v in pair) for pair in report['bounds']]
    lines += [' '.join(format(v,'.12g') for v in pair) for pair in report['coefficient']]
    temporary=path.with_suffix('.tmp')
    temporary.write_text('\n'.join(lines)+'\n')
    temporary.replace(path)


def run_owned(command, timeout):
    import os
    child=subprocess.Popen(command,start_new_session=True)
    try:
        return child.wait(timeout=timeout)
    finally:
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collect',action='store_true')
    parser.add_argument('--trace',type=Path)
    parser.add_argument('--refine',type=Path,help='With --collect, reuse the fitting points from this completed trace and test a regularized candidate on five NEW targets; headset fit must be unchanged')
    parser.add_argument('--timeout',type=int,default=110)
    args=parser.parse_args()
    if not 1<=args.timeout<=120:parser.error('timeout must be 1..120 seconds')
    if args.collect==bool(args.trace):parser.error('Choose --collect or --trace FILE')
    if args.refine and not args.collect:parser.error('--refine requires --collect')
    base=Path(__file__).resolve().parent
    lock=None;fixed=None;source_rows=None
    if args.collect:
        import fcntl
        def interrupted(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM,interrupted)
        lock=(base.parent/'trial.lock').open('a')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:parser.error('Stop the existing hand trial before calibration')
        # Do not calibrate while another bridge is making gaze-dependent clicks.
        for proc in Path('/proc').glob('[0-9]*/comm'):
            try:name=proc.read_text().strip()
            except OSError:continue
            if name=='fh-input':parser.error('A pointer bridge is running; leave it unchanged and stop that trial first')
        folder=base.parent/'evidence'/('gaze-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ'))
        folder.mkdir(parents=True)
        trace=folder/'targets.jsonl'
        command=[str(base/'build/gaze-calibrator'),'--manifest',str(base/'gaze_actions.json'),'--output',str(trace),'--timeout',str(args.timeout)]
        if args.refine:
            source_rows=[json.loads(line) for line in args.refine.read_text().splitlines()]
            age=time.time()-source_rows[0]['created']
            if not source_rows[-1].get('completed') or not 0<=age<=900:parser.error('Refinement requires a completed source capture less than 15 minutes old; redo calibration')
            fit_rows=[r for r in source_rows if 'point' in r and not r['validation']]
            if sorted(r['point'] for r in fit_rows)!=list(range(9)):parser.error('Refinement requires the nine original fitting targets')
            training=[summarize(r) for r in fit_rows]
            bounds=[[min(p['raw'][k] for p in training),max(p['raw'][k] for p in training)] for k in (0,1)]
            fixed=dict(terms=6,coefficient=fit(training,6,.5),ridge=.5)
            if not plausible(fixed['coefficient'],bounds):parser.error('Refinement is not a plausible angular map')
            candidate=dict(fixed,accepted=False,session=source_rows[0]['session'],created=int(time.time()),bounds=bounds)
            write_profile(candidate,folder/'candidate.preview',preview=True)
            (folder/'candidate-before-validation.json').write_text(json.dumps(candidate,indent=2)+'\n')
            command+=['--verify-only','--feedback-profile',str(folder/'candidate.preview')]
        print(f'Calibration evidence: {folder}',flush=True)
        run_owned(command,args.timeout+10)
    else:trace=args.trace;folder=trace.parent
    try:
        rows=[json.loads(line) for line in trace.read_text().splitlines()]
        if source_rows is not None:
            if rows[0]['session']!=source_rows[0]['session']:raise ValueError('Eye-tracker session changed; redo calibration')
            combined=[rows[0]]+[r for r in source_rows if 'point' in r and not r['validation']]+rows[1:]
            (folder/'combined-targets.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in combined))
            rows=combined
        report=analyze(rows,fixed=fixed)
    except (ValueError,OSError,KeyError,IndexError) as exc:
        report=dict(accepted=False,reason=str(exc))
    (folder/'calibration-report.json').write_text(json.dumps(report,indent=2)+'\n')
    if report['accepted']:
        write_profile(report,folder/'gaze.profile')
    print(json.dumps(report,indent=2),flush=True)
    if args.collect and args.timeout>=30:
        run_owned([str(base/'build/gaze-calibrator'),'--manifest',str(base/'gaze_actions.json'),'--result','pass' if report['accepted'] else 'fail'],10)
    if lock:lock.close()
    return 0 if report['accepted'] else 3


if __name__=='__main__':
    try:raise SystemExit(main())
    except KeyboardInterrupt:raise SystemExit(130)
