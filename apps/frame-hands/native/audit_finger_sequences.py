"""Independent arithmetic/lifecycle audit for the saved finger-sequence diagnostic."""
import argparse
import copy
import itertools
import json
from pathlib import Path
import numpy as np
from audit_native_lengths import sha


def verify(rows):
    verified = 0
    persistent = 0
    resets = 0
    for _, window_rows in itertools.groupby(rows, key=lambda r: r['window']):
        last = {}
        last_stamp = None
        for stamp, frame_rows in itertools.groupby(window_rows, key=lambda r: r['raw_ns']):
            if last_stamp is not None and stamp <= last_stamp:
                raise ValueError('Nonmonotonic sequence')
            current = {}
            for row in frame_rows:
                if not row['usable']:
                    continue
                key = (row['source'], row['side'])
                if key in current:
                    raise ValueError('Duplicate source/side')
                computed = []
                for k, model in enumerate(('native', 'mercury')):
                    ratios = []
                    for view in row['views']:
                        points = np.asarray(view['native'] if model == 'native' else view['mercury']['points_raw'])
                        scale = float(np.sqrt(np.sum((points[9] - points[0])**2)))
                        ratios.append(None if scale < 1 else [float(np.sqrt(np.sum((points[tip] - points[tip-3])**2)) / scale) for tip in (8, 12, 16, 20)])
                    decisions = [0 if ratio is None or abs(ratio[0]-ratio[1]) <= .25 else (1 if ratio[0] > ratio[1] else -1) for ratio in ratios]
                    agreement = decisions[0] if len(set(decisions)) == 1 else 0
                    previous = last.get(key) if last_stamp is not None and 0 < stamp-last_stamp <= 100000000 else None
                    length = 0 if not agreement else 1
                    step = None
                    if previous is not None:
                        old = previous[k]
                        if agreement and old['agreement'] == agreement:
                            length = old['length'] + 1
                        if all(v is not None for v in ratios + old['ratios']):
                            step = float(np.max(abs(np.array(ratios)-old['ratios'])))
                    elif k == 0:
                        resets += 1
                    expected_persistence = agreement if length >= 3 else 0
                    metric = row['metrics'][k]
                    if metric['name'] != model or metric['labels'] != decisions or metric['agreed'] != agreement or metric['streak'] != length or metric['persistent'] != expected_persistence:
                        raise ValueError('Changed causal label/persistence')
                    if metric['max_reach_step'] is None:
                        if step is not None:
                            raise ValueError('Missing step')
                    elif step is None or not np.isclose(metric['max_reach_step'], step, rtol=0, atol=1e-12):
                        raise ValueError('Incorrect step')
                    for a,b in zip(metric['reach'],ratios):
                        if (a is None) != (b is None) or a is not None and not np.allclose(a,b,rtol=0,atol=1e-12):
                            raise ValueError('Incorrect reach')
                    computed.append(dict(agreement=agreement,length=length,ratios=ratios))
                    persistent += bool(expected_persistence)
                    verified += 1
                current[key] = computed
            last, last_stamp = current, stamp
    return dict(model_rows_verified=verified,persistent_model_rows=persistent,history_start_or_reset=resets)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--repeat',type=Path,required=True)
    args=parser.parse_args()
    report=json.loads((args.run/'result.json').read_text(encoding='utf-8'))
    if sha(args.run/'rows.json') != report['rows_sha256']:
        raise ValueError('Changed rows')
    rows=json.loads((args.run/'rows.json').read_text(encoding='utf-8'))
    if rows != json.loads((args.repeat/'rows.json').read_text(encoding='utf-8')):
        raise ValueError('Repeat output changed')
    result=verify(rows)
    # Negative controls must fail even though all model point arrays stay unchanged.
    bad=copy.deepcopy(rows)
    next(r for r in bad if r['usable'])['metrics'][0]['persistent']=1
    try:
        verify(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('Forged persistence was accepted')
    bad=copy.deepcopy(rows)
    row=next(r for r in bad if r['usable'] and r['metrics'][0]['streak']>1)
    row['source']+=10000
    try:
        verify(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('Continuity across a new source was accepted')
    result.update(repeated_rows_exact=True,negative_controls=2,source_sha256=sha(Path(__file__)),input_result_sha256=sha(args.run/'result.json'))
    path=args.run/'arithmetic-audit.json'
    with path.open('x',encoding='utf-8') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
