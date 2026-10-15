"""Render exact saved raw/detector/teacher/student views for offline inspection."""
import argparse, json, sys
from pathlib import Path
import cv2
import numpy as np
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'exploratory'))
from frame_data import Recording,read_audit,load_calibration
from review_mesh_recording import render


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('session','replay','upstream','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.upstream));a.output.mkdir(parents=True,exist_ok=False)
    comparisons=[json.loads(line) for line in (a.replay/'comparison-source-views.jsonl').open()]
    matched=[(h['mean_vertex_distance_mm'],r['stamp']) for r in comparisons for h in r['matched']]
    if not matched:raise ValueError('No matched hands')
    median=float(np.median([v for v,s in matched]))
    targets={'largest_matched_difference':max(matched)[1], 'near_median_difference':min(matched,key=lambda item:abs(item[0]-median))[1]}
    changed=[r['stamp'] for r in comparisons if r['assignment_changed']]
    if changed:targets['first_assignment_difference']=changed[0]
    wanted=set(targets.values())
    predictions={label:{} for label in ('teacher','student')}
    for label in predictions:
        for line in (a.replay/(label+'.jsonl')).open():
            row=json.loads(line)
            if row['capture_raw_ns'] in wanted:predictions[label][row['capture_raw_ns']]=row
    audit=read_audit(a.session/'frame/live-views.txt')
    cameras=load_calibration(a.session/'calibration')
    triangles=np.asarray(json.loads((a.upstream/'dataset/generic_hand_model.json').read_text())['mesh_triangles'],np.int32)
    recording=Recording(a.session/'frame/sets.bin');first=recording.index[0][0]
    index={stamp for stamp,_,_ in recording.index}
    evidence=[]
    for reason,stamp in targets.items():
        if stamp not in index or stamp not in audit or any(stamp not in v for v in predictions.values()):
            raise ValueError('Missing exact-capture review input')
        seconds=(stamp-first)/1e9
        records=list(recording.frames(seconds-.000001,seconds+.000001))
        exact=[images for _,s,images in records if s==stamp]
        if len(exact)!=1:raise ValueError('Ambiguous capture')
        panels=[render(exact[0],audit[stamp],predictions[label][stamp],cameras,triangles,seconds) for label in ('teacher','student')]
        frame=np.vstack((panels[0],panels[1][-512:]))
        for y,title in ((1024,'TEACHER: accepted full encoder'),(1536,'STUDENT: unaccepted smaller encoder')):
            cv2.rectangle(frame,(0,y),(frame.shape[1],y+28),(0,0,0),-1)
            cv2.putText(frame,title,(8,y+20),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
        path=a.output/(reason+'.jpg')
        if not cv2.imwrite(str(path),frame,[cv2.IMWRITE_JPEG_QUALITY,94]):raise ValueError('Image write failed')
        evidence.append(dict(reason=reason,stamp=stamp,seconds=seconds,image=path.name))
    (a.output/'manifest.json').write_text(json.dumps(dict(scope='Exact saved capture qualitative overlays; no calibrated physical ground truth.',frames=evidence),indent=2)+'\n')
    print(json.dumps(evidence,indent=2))


if __name__=='__main__':main()
