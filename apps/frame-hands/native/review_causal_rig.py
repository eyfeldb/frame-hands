"""Exact stereo camera review of paired fresh and causal articulated fits."""
import argparse,json,sys
from pathlib import Path
import cv2
import numpy as np
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'exploratory'))
from frame_data import Recording,read_audit,load_calibration
from compare_replay import SEGMENTS
from audit_native_lengths import PARENTS,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('session','upstream','run','recording','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.upstream));a.output.mkdir(parents=True,exist_ok=False)
    rows=[json.loads(x) for x in (a.recording/'poses.jsonl').read_text().splitlines()]
    fresh=np.loadtxt(a.run/'fresh-output.txt');causal=np.loadtxt(a.run/'causal-output.txt')
    # Largest objective gain, first experiment's largest regression, earlier reviewed fist.
    first=min(read_audit(a.session/'frame/live-views.txt'))
    chosen=[int(np.argmin(causal[:,5]-fresh[:,5])),2637,min(range(len(rows)),key=lambda i:abs((rows[i]['raw_ns']-first)/1e9-58.5))]
    capture=Recording(a.session/'frame/sets.bin');audit=read_audit(a.session/'frame/live-views.txt');cams=load_calibration(a.session/'frame/snapshot/device/persist');metadata=[]
    for index in chosen:
        row=rows[index];stamp=row['raw_ns'];seconds=(stamp-capture.index[0][0])/1e9
        frames=[x for x in capture.frames(seconds-1e-6,seconds+1e-6) if x[1]==stamp]
        if len(frames)!=1:raise ValueError('Exact frame unavailable')
        canvas=np.zeros((800,1600,3),np.uint8)
        for view_index,name in enumerate(('slam_left','slam_right')):
            image,capture_ns=frames[0][2][name];cam=cams[name];view=next(v for v in audit[stamp]['views'] if v['id']==row['source_id'] and v['camera']==name)
            points=[view['points'],*[cam.eye_to_window(cam.world_to_eye(x[index,36:99].reshape(21,3))) for x in (fresh,causal)]]
            allpoints=np.concatenate(points);lo=np.maximum(np.floor(allpoints.min(0)-25).astype(int),[0,0]);hi=np.minimum(np.ceil(allpoints.max(0)+25).astype(int),[image.shape[1],image.shape[0]])
            for col,label in enumerate(('Camera','Detector','Fresh fit','Guarded history')):
                panel=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
                if col:
                    xy=points[col-1];xy=np.rint(xy).astype(int)
                    edges=[(PARENTS[j],j) for j in range(1,21)] if col==1 else [(x,y) for x,y,_,_ in SEGMENTS]+[(5,j) for j in (6,8,11,14,17)]
                    for x,y in edges:cv2.line(panel,tuple(xy[x]),tuple(xy[y]),(255,255,0) if col==1 else (255,0,255),1,cv2.LINE_AA)
                crop=panel[lo[1]:hi[1],lo[0]:hi[0]];scale=min(390/crop.shape[1],345/crop.shape[0]);tile=cv2.resize(crop,None,fx=scale,fy=scale)
                x=400*col+(400-tile.shape[1])//2;y=400*view_index+45
                canvas[y:y+tile.shape[0],x:x+tile.shape[1]]=tile
                cv2.putText(canvas,f'{seconds:.3f}s {name} {label}',(400*col+7,400*view_index+22),cv2.FONT_HERSHEY_SIMPLEX,.44,(255,255,255),1)
        filename=f'case-{index}.png'
        if not cv2.imwrite(str(a.output/filename),canvas):raise ValueError('Cannot save review')
        metadata.append(dict(index=index,raw_ns=stamp,source_id=row['source_id'],side=row['side'],seconds=seconds,
                             objective_delta=float(causal[index,5]-fresh[index,5]),used_history=bool(causal[index,100]),image=filename))
    (a.output/'selection.json').write_text(json.dumps(dict(cases=metadata,source_sha256=sha(Path(__file__)),sets_sha256=sha(a.session/'frame/sets.bin')),indent=2)+'\n')
    print(json.dumps(metadata,indent=2))
if __name__=='__main__':main()
