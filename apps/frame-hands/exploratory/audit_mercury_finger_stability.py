"""Independently reconstruct native curl weights and summarize their observed use."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np


def stats(values):
    if not values:return None
    return dict(count=len(values),minimum=float(min(values)),median=float(np.median(values)),p95=float(np.percentile(values,95)),maximum=float(max(values)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--result',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();groups={};records=[];events=[];counts=Counter()
    path=a.result/'frames.jsonl'
    with path.open() as stream:
        for line in stream:
            row=json.loads(line)
            for fit in row['fits']:
                phase='prefix' if row['seconds']<18 else 'evaluation';key=phase+'_'+str(fit['side'])+'_'+fit['branch'];group=groups.setdefault(key,dict(counts=Counter(),zero=[],partial=[],full=[]))
                if fit['before_diagnostic'] is None:
                    counts['reset_fits_without_stability']+=1;group['counts']['reset_fits']+=1;continue
                counts['nonreset_fits']+=1;after=np.asarray(fit['after_angles']);diag=np.asarray(fit['before_diagnostic'],np.float32)
                for index,d in enumerate(diag):
                    total=np.float32(0);weighted=np.float32(0)
                    for value,variance in ((d[6],d[7]),(d[8],d[9])):
                        inverse=np.float32(1/variance);weighted=np.float32(weighted+np.float32(value*inverse));total=np.float32(total+inverse)
                    mean=np.float32(weighted/total);prior=np.float32(np.float32(d[0]+d[1])+d[2]);difference=np.abs(np.float32(mean-prior))
                    multiplier=np.clip(np.float32(float(np.float32(1-difference))+.2),0,1)
                    np.testing.assert_allclose(d[3],prior,atol=1e-6,rtol=0);np.testing.assert_allclose(d[4],mean,atol=1e-6,rtol=0);np.testing.assert_allclose(d[5],multiplier,atol=1e-6,rtol=0)
                    category='zero' if d[5]==0 else 'full' if d[5]==1 else 'partial'
                    record=dict(seconds=row['seconds'],side=fit['side'],branch=fit['branch'],finger=index,category=category,weight=float(d[5]),curl_disagreement_rad=float(difference),mean_view_variance=float((d[7]+d[9])/2),between_view_curl_difference_rad=float(abs(d[6]-d[8])),max_angle_step_rad=float(abs(after[index]-d[:3]).max()),curl_sum_step_rad=float(abs(after[index].sum()-prior)),crop_switch=fit['crop_switch'])
                    group['counts'][category]+=1;group[category].append(record);records.append(record);counts['audited_finger_weights']+=1
                    if 17.099<row['seconds']<17.14 and fit['side']==1 and fit['branch'].startswith('fallback'):events.append(record)
    def summarize(rows):
        return {key:stats([r[key] for r in rows]) for key in ('weight','curl_disagreement_rad','mean_view_variance','between_view_curl_difference_rad','max_angle_step_rad','curl_sum_step_rad')}
    for g in groups.values():
        for category in ('zero','partial','full'):g[category]=summarize(g[category])
    overall={category:summarize([r for r in records if r['category']==category]) for category in ('zero','partial','full')}
    variances=[r['mean_view_variance'] for r in records];q=np.percentile(variances,[25,50,75]);quantiles=[]
    for low,high in zip([-np.inf,*q],[*q,np.inf]):
        subset=[r for r in records if low<r['mean_view_variance']<=high]
        quantiles.append(dict(low=None if not np.isfinite(low) else low,high=None if not np.isfinite(high) else high,count=len(subset),zero_weights=sum(r['category']=='zero' for r in subset),max_angle_step_rad=stats([r['max_angle_step_rad'] for r in subset])))
    high_zero=sorted([r for r in records if r['category']=='zero' and r['mean_view_variance']>q[2]],key=lambda r:r['max_angle_step_rad'],reverse=True)[:12]
    out=dict(counts=counts,groups=groups,overall=overall,variance_quartiles=quantiles,high_reported_variance_zero_weight_examples=high_zero,near_static_right_switch_events=events,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),frames_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),limitations='Variance quartiles are descriptive, not proposed admission thresholds or calibrated uncertainty. The four branches share recordings/observations; counts are correlated contexts, not independent trials. Zero weights can appropriately release a genuinely moving finger. No causal improvement or physical accuracy follows from correlation. Reset fits have no temporal stability and are excluded from multiplier conclusions.')
    a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(counts=counts,overall=overall,variance_quartiles=quantiles,examples=high_zero[:3],near_static_right_switch_events=events),indent=2))


if __name__=='__main__':main()
