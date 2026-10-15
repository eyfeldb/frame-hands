"""Score fresh correctly identified hand availability, including detector misses.

Sparse manual visible-hand labels; not pose accuracy or rendered availability.
"""
import argparse,json
from collections import Counter
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',type=Path,required=True);p.add_argument('--labels',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();annotation=json.loads(a.labels.read_text());wanted={r['raw_ns'] for r in annotation['visibility']}
    observed={(r['raw_ns'],r['camera'],r['source_id']):r['physical_side'] for r in annotation['labels']}
    predictions={}
    with a.predictions.open() as stream:
        for line in stream:
            r=json.loads(line)
            if r['capture_raw_ns'] in wanted:predictions[r['capture_raw_ns']]=r
    opportunities={}
    for v in annotation['visibility']:
        if v['visible']:opportunities.setdefault((v['raw_ns'],v['physical_side']),[]).append(v)
    rows=[]
    for (stamp,side),views in opportunities.items():
        record=predictions.get(stamp)
        if record is None:raise ValueError('Candidate omitted annotated capture entirely')
        hands=[h for h in record['hands'] if h['side']==side]
        if len(hands)>1:raise ValueError('Duplicate side output')
        detector_seen=any(v['detector_ids'] for v in views)
        sources=[];verdict='no_side_output'
        if hands:
            sources=[observed.get((stamp,v['camera'],v['source_id']),'unlabeled') for v in hands[0]['source_views']]
            if not sources or 'unlabeled' in sources:verdict='unverifiable'
            elif set(sources)=={side}:verdict='correct_supported_output'
            elif side in sources:verdict='mixed_physical_hands'
            else:verdict='wrong_physical_hand'
        rows.append(dict(stamp=stamp,side=side,detector_seen=detector_seen,capture_pose_valid=record['capture_pose']['valid'],source_physical_sides=sources,verdict=verdict))
    summary=dict(scope=__doc__,opportunities=len(rows),outcomes=dict(Counter(r['verdict'] for r in rows)),no_detected_view=sum(not r['detector_seen'] for r in rows),missing_capture_pose=sum(not r['capture_pose_valid'] for r in rows))
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'rows.json').write_text(json.dumps(rows,indent=2)+'\n');(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
