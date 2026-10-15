"""Audit fixed-crop model sequences without treating consistency as pose accuracy."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from run_tracker_rig_check import sha, stats
from probe_finger_sequences import extension, semantic_label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'reference'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = json.loads((args.run / 'result.json').read_text())
    reference = json.loads((args.reference / 'result.json').read_text())
    requests = json.loads((args.run / 'requests.json').read_text())
    assert result['native_geometry_reproduced'] and result['repeat_exact']
    assert result['models'] == reference['models']
    assert sha(args.run / 'requests.json') == result['provenance']['requests_sha256']
    sparse = {(r['raw_ns'], r['source'], r['camera']): r for r in reference['records']}
    cases = defaultdict(list)
    states = {}
    window_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    repeated = 0
    per_case = {r['case']: r for r in result['cases']}
    for row in result['records']:
        cases[row['case']].append(row)
        old = sparse.get((row['raw_ns'], row['source'], row['camera']))
        if old:
            assert row['native_gray_sha256'] == old['native_gray_sha256']
            assert row['models'] == old['models']
            repeated += 1
        for model in ('lite', 'full'):
            values = window_metrics[row['window']][model]
            prediction = row['models'][model]
            values['presence'].append(prediction['presence'])
            reach = extension(prediction['points_raw'])
            key = (row['window'], row['source'], row['camera'], model)
            previous = states.get(key)
            if previous and 0 < row['raw_ns'] - previous[0] <= 100000000:
                values['side_flips'].append(int((prediction['right'] >= .5) != (previous[1] >= .5)))
                if reach is not None and previous[2] is not None:
                    values['reach_step'].append(float(np.max(np.abs(reach - previous[2]))))
            states[key] = (row['raw_ns'], prediction['right'], reach)
    ranked = []
    source_metrics = defaultdict(lambda: defaultdict(list))
    for case, rows in cases.items():
        if len(rows) != 2:
            continue
        assert [r['camera'] for r in rows] == ['slam_left', 'slam_right']
        item = per_case[case]
        for model in ('lite', 'full'):
            values = window_metrics[rows[0]['window']][model]
            p = [r['models'][model] for r in rows]
            values['paired_presence_pass'].append(int(all(v['presence'] >= .8 for v in p)))
            values['side_disagreement'].append(int((p[0]['right'] >= .5) != (p[1]['right'] >= .5)))
            labels = [semantic_label(extension(v['points_raw'])) for v in p]
            values['opposed_finger_labels'].append(int(labels[0] * labels[1] == -1))
            values['index_middle_epi'].append(item['index_middle_epipolar_rms_degrees'][model])
            source_metrics[(rows[0]['window'], rows[0]['source'])][model].append(item['index_middle_epipolar_rms_degrees'][model])
        ranked.append(dict(case=case, window=rows[0]['window'], seconds=rows[0]['seconds'], source=rows[0]['source'],
            full_minus_lite_epi=item['index_middle_epipolar_rms_degrees']['full'] - item['index_middle_epipolar_rms_degrees']['lite']))
    summary = {}
    for window, models in window_metrics.items():
        summary[window] = {}
        for model, values in models.items():
            summary[window][model] = dict(views=len(values['presence']),
                low_presence=sum(p < .8 for p in values['presence']),
                paired_views=len(values['paired_presence_pass']),
                paired_presence_pass=sum(values['paired_presence_pass']),
                side_disagreements=sum(values['side_disagreement']),
                side_flips=sum(values['side_flips']),
                side_transitions=len(values['side_flips']),
                opposed_finger_labels=sum(values['opposed_finger_labels']),
                reach_step=stats(values['reach_step']),
                index_middle_epi=stats(values['index_middle_epi']))
    ordered = sorted(ranked, key=lambda r: r['full_minus_lite_epi'])
    report = dict(scope=__doc__, result_sha256=sha(args.run / 'result.json'), auditor_sha256=sha(Path(__file__)),
        sets=len(requests['sets']), views=len(result['records']), hand_hypotheses=len(cases),
        single_view_hypotheses=sum(len(v) == 1 for v in cases.values()),
        sparse_exact_views=repeated, summary=summary,
        source_epipolar=[dict(window=k[0], source=k[1], pairs=len(v['lite']),
                             models={m: stats(x) for m, x in v.items()})
                         for k, v in source_metrics.items()],
        largest_improvements=ordered[:8], largest_regressions=list(reversed(ordered[-8:])),
        limitations=['Baseline detector selection and source IDs are fixed; no detection, association, coverage or recovery claim.',
            'Projected reach steps include real motion, projection and crop changes. They are not measured jitter.',
            'Epipolar agreement and repeated labels do not establish physical anatomy or correct identity.',
            'All windows from one previously studied recording. No independent recording or wearer acceptance.',
            'No smoothing, guessed gestures, presence thresholds or production tracking policies changed.'])
    (args.run / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
