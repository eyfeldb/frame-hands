"""Pair candidate poses on the latest recorded viewer schedule, with explicit age limits.

First viewer-consumption ticks are observed; exact publication times are not. Capture age is exact when an accepted original hand
was traced, otherwise recorded publisher age plus median observed post-publish
delay. Candidate timing is counterfactual; extra inference latency is not modeled.
"""
import argparse,json,bisect
from pathlib import Path
import numpy as np
from mesh_packet import pose_packet


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--candidate-delay-ms',type=float,default=0.,help='Constant added candidate delivery delay sensitivity; does not model throughput or changed sampling')
    p.add_argument('--arrival-phase',choices=('earliest','midpoint','latest'),default='latest',help='Assumed original publication phase between previous and first consuming viewer ticks')
    a=p.parse_args()
    if not np.isfinite(a.candidate_delay_ms) or a.candidate_delay_ms<0:p.error('Delay must be finite and nonnegative')
    a.output.mkdir(parents=True,exist_ok=False)
    original={r['sequence']:r for l in (a.session/'predictions.jsonl').open() if (r:=json.loads(l))}
    alternate={r['capture_raw_ns']:r for l in a.candidate.open() if (r:=json.loads(l))}
    timing={r['sequence']:r for l in (a.session/'timing.jsonl').open() if (r:=json.loads(l)) and 'sequence' in r}
    arrivals={};interval_start={};accepted={};ages={};ticks=[]
    for line in (a.session/'render.jsonl').open():
        r=json.loads(line);seq=r['sequence']
        if r.get('kind')=='observation' and r['accepted']:accepted[seq,r['side']]=r['id']
        if r.get('kind')!='render':continue
        now=r['render_tick_s'];arrivals.setdefault(seq,now)
        if not ticks or ticks[-1]!=now:ticks.append(now)
        interval_start.setdefault(seq,ticks[-2] if len(ticks)>1 else now)
        if r['visible'] and accepted.get((seq,r['side']))==r['id'] and now==arrivals[seq]:
            ages.setdefault(seq,r['age_s'])
    if any(b<=a for a,b in zip(ticks,ticks[1:])):raise ValueError('Non-increasing viewer timeline')
    extras=[age-timing[seq]['age_upper_ms']*.001 for seq,age in ages.items() if seq in timing]
    if not extras:raise ValueError('No trace anchors for capture age')
    delay=max(0,float(np.median(extras)));origin=ticks[0]
    counts=dict(arrival_phase=a.arrival_phase,candidate_delay_ms=a.candidate_delay_ms,observed_viewer_ticks=len(ticks),paired_updates=0,trace_anchored_ages=0,estimated_ages=0,median_post_publish_delay_ms=delay*1000,post_publish_delay_ms_p05_p95=np.percentile(extras,[5,95]).tolist())
    # Convert quantiles from seconds to milliseconds too.
    counts['post_publish_delay_ms_p05_p95']=[x*1000 for x in counts['post_publish_delay_ms_p05_p95']]
    events={n:[] for n in ('baseline','candidate')}
    consume_shifts=[]
    phase={'earliest':0.,'midpoint':.5,'latest':1.}[a.arrival_phase]
    for seq,now in arrivals.items():
        if seq not in original or seq not in timing:continue
        old=original[seq];new=alternate.get(old['capture_raw_ns'])
        if new is None:raise ValueError('Candidate lacks a viewer-consumed update')
        age=ages.get(seq,timing[seq]['age_upper_ms']*.001+delay)
        counts['paired_updates']+=1
        counts['trace_anchored_ages' if seq in ages else 'estimated_ages']+=1
        assumed_arrival=interval_start[seq]+phase*(now-interval_start[seq])
        # The earliest endpoint is exclusive: publication was not consumed there.
        if phase==0. and now>interval_start[seq]:assumed_arrival+=1e-7
        candidate_arrival=assumed_arrival+a.candidate_delay_ms*.001
        next_tick=bisect.bisect_left(ticks,candidate_arrival)
        if next_tick<len(ticks):consume_shifts.append((ticks[next_tick]-now)*1000)
        for name,r in (('baseline',old),('candidate',new)):
            packet=pose_packet(r,np.eye(4),seq,old['capture_mono_ns'])
            events[name].append(dict(now=(candidate_arrival if name=='candidate' else now)-origin,capture=now-origin-age,stamp=old['capture_mono_ns']*1e-9,hands=packet['hands'],valid=old['capture_pose']['valid'] and old['current_pose']['valid']))
    counts['effective_consume_shift_ms_p50_p95']=np.percentile(consume_shifts,[50,95]).tolist()
    for name,rows in events.items():
        data=dict(name='recorded_viewer_schedule_'+name,scope=__doc__+f' Constant candidate delivery delay sensitivity: {a.candidate_delay_ms} ms; assumed original publication phase {a.arrival_phase}. Unchanged sampling/throughput, not a live timing forecast. Original first consumption only bounds publication to the preceding viewer interval.',counts=counts,ticks=[t-origin for t in ticks],events=rows)
        (a.output/(name+'.json')).write_text(json.dumps(data)+'\n')
    (a.output/'summary.json').write_text(json.dumps(counts,indent=2)+'\n');print(json.dumps(counts,indent=2))

if __name__=='__main__':main()
