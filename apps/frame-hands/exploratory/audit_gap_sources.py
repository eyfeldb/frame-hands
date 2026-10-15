"""Join gap source review to detector and identity diagnostics; never rerun inference."""
import argparse, hashlib, json
from pathlib import Path
import cv2
from frame_data import Recording, read_audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('review','session','paired','evidence','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    report=json.loads((a.review/'report.json').read_text())
    events=json.loads((a.paired/'candidate.json').read_text())['events']
    meta={v['capture_mono_ns']*1e-9:v['capture_raw_ns'] for v in map(json.loads,(a.session/'predictions.jsonl').open())}
    evidence={v['stamp']:v for v in map(json.loads,a.evidence.open())}
    audit=read_audit(a.session/'frame/live-views.txt');rec=Recording(a.session/'frame/sets.bin')
    records=[]
    for gap in report['reports']:
        selected={s['event_index']:s['label'] for s in gap['selected']}
        inside={i for i,e in enumerate(events) if gap['gap']['start_s']<e['now']<gap['gap']['end_s']}
        for i in sorted(set(selected)|inside):
            event=events[i];stamp=meta[event['stamp']];ev=evidence[stamp];obs=audit[stamp]
            assert sorted(h['side'] for h in event['hands'])==sorted(v['side'] for v in ev['association']['assignments'])
            views=[]
            for v in obs['views']:
                match=next((x for x in ev['views'] if (x['camera'],x['source_id'])==(v['camera'],v['id'])),None)
                assigned=[x['side'] for x in ev['association']['assignments'] if any((z['camera'],z['source_id'])==(v['camera'],v['id']) for z in x['views'])]
                views.append(dict(camera=v['camera'],source_id=v['id'],presence=v['presence'],raw_right=v['right'],image_evidence=match,assigned_sides=assigned))
            records.append(dict(gap=gap['index'],event_index=i,inside=i in inside,selected=selected.get(i),stamp=stamp,valid=event['valid'],sides=[h['side'] for h in event['hands']],views=views,association=ev['association']))
            if i not in selected or gap['index']==1:continue
            seconds=(stamp-rec.index[0][0])/1e9
            _,actual,images=next(rec.frames(seconds-1e-6,seconds+1e-6));assert actual==stamp
            panels=[]
            for camera in ('slam_left','slam_right'):
                img=cv2.cvtColor(images[camera][0],cv2.COLOR_GRAY2BGR)
                for v in obs['views']:
                    if v['camera']!=camera:continue
                    color=(0,255,255) if v['id']%2 else (255,0,255)
                    pts=v['points'].round().astype(int)
                    for start in (1,5,9,13,17):
                        for x,y in zip([0,start,start+1,start+2],[start,start+1,start+2,start+3]):cv2.line(img,tuple(pts[x]),tuple(pts[y]),color,2)
                    cv2.putText(img,str(v['id']),tuple(pts[0]),cv2.FONT_HERSHEY_SIMPLEX,.7,color,2)
                cv2.putText(img,f'{camera} {seconds:.6f}s; detector IDs',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
                panels.append(img)
            assert cv2.imwrite(str(a.output/f'gap-{gap["index"]:02}-{selected[i]}.jpg'),cv2.hconcat(panels))
    out=dict(records=records,sha256={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in (a.review/'report.json',a.paired/'candidate.json',a.evidence,a.session/'frame/live-views.txt')},limits='Exact consumed packet inputs, not simultaneous display images. Detector IDs and scores are not anatomical ground truth. Sparse raw review cannot establish visibility throughout an entire gap.')
    (a.output/'audit.json').write_text(json.dumps(out,indent=2)+'\n')
    for g in range(10):
        rows=[r for r in records if r['gap']==g and r['inside']]
        print(json.dumps(dict(gap=g,inside=len(rows),view_counts=[len(r['views']) for r in rows],unassigned_view_counts=[sum(not v['assigned_sides'] for v in r['views']) for r in rows])))

if __name__=='__main__':main()
