"""Pair rendered gaps with exact consumed source captures, without a clock shortcut."""
import argparse,bisect,hashlib,json,shutil
from pathlib import Path
import cv2,numpy as np
from frame_data import Recording


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('session','paired','previous','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    data=json.loads((a.paired/'candidate.json').read_text());old=json.loads((a.previous/'identity.json').read_text());result=json.loads((a.paired/'candidate.json.results.json').read_text())['modes']['predict'];old_result=json.loads((a.previous/'identity.json.results.json').read_text())['modes']['predict']
    assert data['ticks']==old['ticks'] and result['hidden_gaps']==old_result['hidden_gaps'];assert len(data['events'])==len(old['events'])
    differences=[]
    for one,two in zip(data['events'],old['events']):
        assert all(one[k]==two[k] for k in ('now','capture','stamp','valid'))
        assert [(h['side'],h['source_id']) for h in one['hands']]==[(h['side'],h['source_id']) for h in two['hands']]
        for x,y in zip(one['hands'],two['hands']):differences.append([float(abs(np.asarray(x['wrist_world_m'])-y['wrist_world_m']).max()),float(abs(np.asarray(x['joint_angles'])-y['joint_angles']).max())])
    originals={}
    for line in (a.session/'predictions.jsonl').open():
        row=json.loads(line);originals[row['capture_mono_ns']*1e-9]=dict(sequence=row['sequence'],capture_raw_ns=row['capture_raw_ns'],capture_mono_ns=row['capture_mono_ns'])
    rec=Recording(a.session/'frame/sets.bin');first=rec.index[0][0];events=data['events'];times=[e['now'] for e in events];gaps=[g for g in result['hidden_gaps'] if not g['censored'] and g['duration_ms']>25];assert len(gaps)==10
    protocol=dict(session=str(a.session.resolve()),paired=str(a.paired.resolve()),selection='All ten completed predictive presentation gaps longer than25ms, including both sides when simultaneous. Select latest consumed packet at gap start, middle consumed packet during gap when present, first consumed packet at/after reappearance; deduplicate packet indices within each gap.',clock='Exact headset-mono stamp joins prediction metadata to raw capture. Render timestamps choose packet consumption, not physical camera exposure at that render instant. Capture-local age is conservative/partly estimated; no exact cross-clock alignment or visibility-throughout-gap claim.',review='Raw full-camera images only, no model/skeleton overlays. Labels are diagnostic observations to be recorded after inspection; selection is loss-based, not independent accuracy validation.',provenance=dict(exact_ticks=len(data['ticks']),exact_events=len(events),exact_gap_intervals=True,max_wrist_matrix_difference=float(np.max(differences,axis=0)[0]),max_angle_difference_rad=float(np.max(differences,axis=0)[1])),sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [a.paired/'candidate.json',a.paired/'candidate.json.results.json',a.previous/'identity.json',a.session/'predictions.jsonl']})
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');shutil.copy2(Path(__file__),a.output/'prepare_gap_source_review.py');reports=[]
    for gi,gap in enumerate(gaps):
        begin=bisect.bisect_right(times,gap['start_s'])-1;end=bisect.bisect_left(times,gap['end_s']);during=[i for i in range(max(0,begin+1),min(len(events),end)) if gap['start_s']<times[i]<gap['end_s']]
        selected=[('at_start',begin)]
        if during:selected.append(('during',during[len(during)//2]))
        if end<len(events):selected.append(('after',end))
        selected=list({i:(label,i) for label,i in selected if i>=0}.values());panels=[];records=[]
        last_side=next((events[i] for i in range(begin,-1,-1) if events[i]['valid'] and any(h['side']==gap['side'] for h in events[i]['hands'])),None)
        for label,index in selected:
            event=events[index];meta=originals[event['stamp']];seconds=(meta['capture_raw_ns']-first)/1e9;actual_time,stamp,images=next(rec.frames(seconds-1e-6,seconds+1e-6));assert stamp==meta['capture_raw_ns'];columns=[]
            for camera in ('slam_left','slam_right'):
                raw=images[camera][0];rgb=cv2.cvtColor(raw,cv2.COLOR_GRAY2BGR);rgb=cv2.resize(rgb,(768,int(raw.shape[0]*768/raw.shape[1])));cv2.putText(rgb,f'{camera} source {seconds:.6f}s',(8,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1);columns.append(rgb)
            panel=cv2.hconcat(columns);header=np.zeros((48,panel.shape[1],3),np.uint8);cv2.putText(header,f'gap {gi} side {gap["side"]} {label} | consumed {event["now"]:.3f}s | packet age {(event["now"]-event["capture"])*1000:.1f}ms',(8,28),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1);panels.append(cv2.vconcat([header,panel]))
            records.append(dict(label=label,event_index=index,consumed_render_s=event['now'],capture_local_s=event['capture'],capture_age_at_consumption_ms=(event['now']-event['capture'])*1000,**meta,source_seconds=seconds,valid=event['valid'],candidate_sides=[h['side'] for h in event['hands']],candidate_ids=[h['source_id'] for h in event['hands']]))
        cv2.imwrite(str(a.output/f'gap-{gi:02}.jpg'),cv2.vconcat(panels))
        reports.append(dict(index=gi,gap=gap,consumed_updates_strictly_inside=len(during),inside_updates_with_side=sum(any(h['side']==gap['side'] for h in events[i]['hands']) for i in during),last_side_capture_age_at_gap_start_ms=None if last_side is None else (gap['start_s']-last_side['capture'])*1000,selected=records))
    report=dict(protocol=protocol,reports=reports);(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps([dict(index=r['index'],gap=r['gap'],updates=r['consumed_updates_strictly_inside'],with_side=r['inside_updates_with_side'],selected_seconds=[v['source_seconds'] for v in r['selected']]) for r in reports],indent=2))


if __name__=='__main__':main()
