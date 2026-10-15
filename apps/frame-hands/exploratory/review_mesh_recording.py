"""Offline exact-capture visual review: raw stereo, detector, and projected mesh."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from frame_data import Recording,read_audit,load_calibration
from run_mesh_trial import pose_matrix

EDGES=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]
COLORS=[(255,230,0),(0,155,255)] # BGR cyan left, orange right

def render(images,observation,prediction,cameras,triangles,seconds,model_row=True):
    rows=[]
    world=pose_matrix(prediction['capture_pose']['capture'])
    for row in range(3 if model_row else 2):
        columns=[]
        for name in ('slam_left','slam_right'):
            raw=images[name][0]
            image=cv2.cvtColor(raw,cv2.COLOR_GRAY2BGR)
            labels=[]
            if row==1:
                for v in observation['views']:
                    if v['camera']!=name:continue
                    points=np.rint(v['points']).astype(np.int32)
                    color=COLORS[int(v['right']>.5)]
                    for a,b in EDGES:cv2.line(image,points[a],points[b],color,2,cv2.LINE_AA)
                    cv2.putText(image,f"ID{v['id']} R{v['right']:.2f} p{v['presence']:.2f}",tuple(points[0]),cv2.FONT_HERSHEY_SIMPLEX,.65,color,2)
            elif row==2:
                cam=cameras[name].copy(camera_to_world_xf=world@cameras[name].camera_to_world_xf)
                for h in prediction['hands']:
                    xyz=cam.world_to_eye(np.array(h['vertices_world_mm']))
                    uv=cam.eye_to_window(xyz)
                    valid=np.isfinite(uv).all(1)&(xyz[:,2]>0)&(np.abs(uv).max(1)<10000)
                    faces=triangles[valid[triangles].all(1)]
                    polys=np.rint(uv[faces]).astype(np.int32)
                    layer=image.copy();cv2.fillPoly(layer,list(polys),COLORS[h['side']])
                    image=cv2.addWeighted(image,.65,layer,.35,0)
                    wrist=np.array(h['wrist_world_mm'])[:3,3]
                    w=cam.eye_to_window(cam.world_to_eye(wrist))
                    if np.isfinite(w).all() and max(abs(w))<10000:
                        cv2.putText(image,f"{'R' if h['side'] else 'L'} ID{h['source_id']}",tuple(w.astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.9,COLORS[h['side']],2)
                    labels.append(f"{'R' if h['side'] else 'L'}={h['source_id']}")
            image=cv2.resize(image,(528,512))
            cv2.rectangle(image,(0,0),(528,28),(0,0,0),-1)
            model_title='MODEL raw mesh: '
            if prediction.get('input_control'):
                model_title='HYPOTHESIS '+prediction['input_control']+' pixels: '
            title=['RAW','DETECTOR fresh labels',model_title+','.join(labels)][row]
            cv2.putText(image,f'{seconds:.3f}s {name} | {title}',(5,19),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
            columns.append(image)
        rows.append(np.hstack(columns))
    return np.vstack(rows)

def presence_row(images,rows,seconds):
    columns=[]
    for name in ('slam_left','slam_right'):
        image=cv2.cvtColor(images[name][0],cv2.COLOR_GRAY2BGR)
        for row in rows:
            if row['camera']!=name:continue
            points=np.rint(row['model_points']).astype(np.int32);color=COLORS[row['side']]
            if not (np.abs(points)<10000).all():continue
            for a,b in EDGES:cv2.line(image,points[a],points[b],color,2,cv2.LINE_AA)
            text=f"slot {'R' if row['side'] else 'L'} p{row['presence']:.2f} qR{row['right']:.2f} e{row['normalized_disagreement']:.2f}"
            cv2.putText(image,text,tuple(points[0]),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
        image=cv2.resize(image,(528,512));cv2.rectangle(image,(0,0),(528,28),(0,0,0),-1)
        control=rows[0]['input_control'] if rows else 'no'
        cv2.putText(image,f'{seconds:.3f}s {name} | VERIFY {control} pixels',(5,19),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
        columns.append(image)
    return np.hstack(columns)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('session',type=Path);p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--seconds',type=float,nargs='+',required=True)
    p.add_argument('--predictions',type=Path)
    p.add_argument('--presence',type=Path,help='Optional independent ROI presence rows.json')
    p.add_argument('--output',type=Path)
    p.add_argument('--detector-only',action='store_true',help='Raw images and recorded detector only; no candidate output read')
    a=p.parse_args();sys.path.insert(0,str(a.upstream))
    calibration=a.session/'calibration'
    cameras=load_calibration(calibration if calibration.exists() else a.session/'frame/snapshot/device/persist')
    model=json.loads((a.upstream/'dataset/generic_hand_model.json').read_text())
    triangles=np.array(model['mesh_triangles'],dtype=np.int32)
    audit=read_audit(a.session/'frame/live-views.txt');first=min(audit)
    targets={min(audit,key=lambda t:abs((t-first)/1e9-s)) for s in a.seconds}
    predictions={}
    if a.detector_only:
        predictions={stamp:dict(capture_pose=dict(capture=np.eye(4)[:3].reshape(-1).tolist()),hands=[]) for stamp in targets}
    else:
        for line in (a.predictions or a.session/'predictions.jsonl').open():
            r=json.loads(line)
            if r['capture_raw_ns'] in targets:predictions[r['capture_raw_ns']]=r
    presence={}
    if a.presence:
        for row in json.loads(a.presence.read_text()):presence.setdefault(row['stamp'],[]).append(row)
    out=a.output or a.session/'visual-review';out.mkdir(parents=True,exist_ok=True)
    rec=Recording(a.session/'frame/sets.bin')
    for seconds,stamp,images in rec.frames(min(a.seconds)-.1,max(a.seconds)+.1):
        if stamp not in predictions:continue
        frame=render(images,audit[stamp],predictions[stamp],cameras,triangles,seconds,model_row=not a.detector_only)
        if a.presence:frame=np.vstack((frame,presence_row(images,presence.get(stamp,[]),seconds)))
        path=out/f'{seconds:06.3f}.jpg';cv2.imwrite(str(path),frame,[cv2.IMWRITE_JPEG_QUALITY,93]);print(path)
    missing=targets-predictions.keys()
    if missing:print('No exact model prediction for selected captures:',sorted(missing))
if __name__=='__main__':main()
