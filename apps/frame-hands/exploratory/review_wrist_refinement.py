"""Private exact-capture crops for comparing unchanged and corrected meshes."""
import argparse,json,sys
from pathlib import Path
import cv2,numpy as np
from frame_data import Recording,load_calibration
from run_mesh_trial import pose_matrix
from review_mesh_recording import COLORS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('session','baseline','candidate','upstream','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--seconds',type=float,required=True);p.add_argument('--side',type=int,choices=(0,1),required=True);p.add_argument('--label',default='WRIST HYPOTHESIS');a=p.parse_args()
    sys.path.insert(0,str(a.upstream))
    records=Recording(a.session/'frame/sets.bin');first=records.index[0][0]
    with a.candidate.open() as f:
        row=min((json.loads(s) for s in f),key=lambda r:abs(r.get('seconds',(r['capture_raw_ns']-first)/1e9)-a.seconds))
    with a.baseline.open() as f:original=next(r for r in map(json.loads,f) if r['capture_raw_ns']==row['capture_raw_ns'])
    baseline=next((h for h in original['hands'] if h['side']==a.side),None);candidate=next(h for h in row['hands'] if h['side']==a.side)
    _,_,images=next(r for r in records.frames(max(0,a.seconds-.2),a.seconds+.2) if r[1]==row['capture_raw_ns'])
    cal=a.session/'calibration';cameras=load_calibration(cal if cal.exists() else a.session/'frame/snapshot/device/persist');world=pose_matrix(row['capture_pose']['capture'])
    faces=np.asarray(json.loads((a.upstream/'dataset/generic_hand_model.json').read_text())['mesh_triangles'],dtype=np.int32);rows=[]
    for name,base in cameras.items():
        camera=base.copy(camera_to_world_xf=world@base.camera_to_world_xf);raw=cv2.cvtColor(images[name][0],cv2.COLOR_GRAY2BGR);panels=[raw.copy()];uvs=[]
        for h in (baseline,candidate):
            if h is None:
                panels.append(raw.copy());continue
            eye=camera.world_to_eye(np.asarray(h['vertices_world_mm']));uv=camera.eye_to_window(eye);uvs.append(uv)
            layer=raw.copy();cv2.fillPoly(layer,list(np.rint(uv[faces]).astype(np.int32)),COLORS[a.side]);panels.append(cv2.addWeighted(raw,.65,layer,.35,0))
        all_uv=np.vstack(uvs);lo=np.maximum(0,np.floor(all_uv.min(0)-50).astype(int));hi=np.minimum([raw.shape[1],raw.shape[0]],np.ceil(all_uv.max(0)+50).astype(int));cols=[]
        for label,panel in zip(('RAW','BASELINE' if baseline is not None else 'BASELINE ABSENT',a.label),panels):
            crop=panel[lo[1]:hi[1],lo[0]:hi[0]];scale=400/max(crop.shape[:2]);crop=cv2.resize(crop,None,fx=scale,fy=scale);tile=np.zeros((450,420,3),np.uint8);tile[40:40+crop.shape[0],10:10+crop.shape[1]]=crop
            cv2.putText(tile,f'{name} {label}',(6,20),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1);cols.append(tile)
        rows.append(np.hstack(cols))
    a.output.parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(a.output),np.vstack(rows),[cv2.IMWRITE_JPEG_QUALITY,95]);print(a.output)

if __name__=='__main__':main()
