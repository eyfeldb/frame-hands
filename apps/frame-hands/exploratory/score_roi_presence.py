"""Diagnostic availability of a frozen ROI-evidence rule; not physical accuracy.

Accept a hypothesis only when the SAME camera has presence >=0.5 and median
20-joint disagreement <=0.1 crop widths. These are engineering probe thresholds,
not calibrated likelihoods. No accepted model, renderer or live code uses them.
"""
import argparse,json,math
from pathlib import Path


def supported_views(rows):
    return [r for r in rows if math.isfinite(r['presence']) and math.isfinite(r['normalized_disagreement']) and .5<=r['presence']<=1 and 0<=r['normalized_disagreement']<=.1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('hypotheses','presence','baseline','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();hypotheses=[json.loads(s) for s in a.hypotheses.open()]
    stamps={r['capture_raw_ns'] for r in hypotheses};baseline={}
    for line in a.baseline.open():
        row=json.loads(line)
        if row['capture_raw_ns'] in stamps:baseline[row['capture_raw_ns']]={h['side'] for h in row['hands']}
    if baseline.keys()!=stamps:raise ValueError('Baseline lacks exact captures')
    evidence={}
    for row in json.loads(a.presence.read_text()):evidence.setdefault((row['stamp'],row['side']),[]).append(row)
    counts={str(side):dict(frames=len(hypotheses),hypotheses=0,baseline_outputs=0,any_presence=0,supported=0,added_over_baseline=0,baseline_without_support=0,no_projectable_view=0) for side in (0,1)}
    accepted=[]
    for row in hypotheses:
        stamp=row['capture_raw_ns'];hands={h['side'] for h in row['hands']}
        for side in (0,1):
            c=counts[str(side)];views=evidence.get((stamp,side),[])
            support=side in hands and bool(supported_views(views));before=side in baseline[stamp]
            c['hypotheses']+=side in hands;c['baseline_outputs']+=before
            c['any_presence']+=side in hands and any(.5<=r['presence']<=1 for r in views)
            c['supported']+=support;c['added_over_baseline']+=support and not before;c['baseline_without_support']+=before and not support
            c['no_projectable_view']+=side in hands and not views
            accepted.append(dict(seconds=row['seconds'],stamp=stamp,side=side,supported=support,baseline_output=before))
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'decisions.json').write_text(json.dumps(accepted)+'\n')
    report=dict(scope=__doc__,counts=counts,args={k:str(v) for k,v in vars(a).items()})
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
