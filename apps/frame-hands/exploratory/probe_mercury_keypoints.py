"""Frozen saved-frame Mercury keypoint probe; models provide hypotheses, not truth."""
import argparse,hashlib,json,shutil,sys,time
from pathlib import Path
import cv2,numpy as np,torch
from frame_data import Recording,load_calibration
from run_mesh_trial import pose_matrix
from probe_roi_presence import crop_matrix
from replay_umetrack import COMMON
from mercury_probe import MercuryModels
from mercury_keypoints import StereoCrop,estimate

CHAINS=[[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]]
def draw(image,points,color):
    out=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR) if image.ndim==2 else image.copy();points=np.asarray(points)
    for chain in CHAINS:
        for a,b in zip(chain,chain[1:]):
            if np.isfinite(points[[a,b]]).all():cv2.line(out,tuple(np.rint(points[a]).astype(int)),tuple(np.rint(points[b]).astype(int)),color,1,cv2.LINE_AA)
    for point in points:
        if np.isfinite(point).all():cv2.circle(out,tuple(np.rint(point).astype(int)),2,color,-1)
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['protocol','upstream','models','runtime','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();sys.path[:0]=[str(a.runtime.resolve()),str(a.upstream.resolve())]
    from lib.common.hand import HandModel
    from lib.common.hand_skinning import skin_landmarks
    torch.set_num_threads(1);cv2.setNumThreads(1);a.output.mkdir(parents=True,exist_ok=False)
    protocol=json.loads(a.protocol.read_text());shutil.copy2(a.protocol,a.output/'protocol.json')
    snapshot=a.output/'snapshot';snapshot.mkdir()
    for name in ['mercury_probe.py','mercury_keypoints.py','probe_mercury_keypoints.py','frame_data.py','probe_roi_presence.py']:
        shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    raw=json.loads((a.upstream/'dataset/generic_hand_model.json').read_text());hand_model=HandModel(**{k:torch.tensor(v) if isinstance(v,list) else v for k,v in raw.items()})
    model=MercuryModels(a.models);contexts={};reports=[]
    for case in protocol['cases']:
        folder=Path(protocol['case_root'])/case['name'];row=json.loads((folder/'baseline.jsonl').read_text());source_report=json.loads((folder/'report.json').read_text());h=next(h for h in row['hands'] if h['side']==case['side'])
        session=Path(case['session'])
        if str(session) not in contexts:
            cal=session/'calibration';contexts[str(session)]=(Recording(session/'frame/sets.bin'),load_calibration(cal if cal.exists() else session/'frame/snapshot/device/persist'))
        rec,cameras=contexts[str(session)];seconds=(row['capture_raw_ns']-rec.index[0][0])/1e9;_,stamp,images=next(rec.frames(seconds-.000001,seconds+.000001))
        if stamp!=row['capture_raw_ns']:raise ValueError('Capture mismatch')
        world=pose_matrix(row['capture_pose']['capture']);wrist=torch.tensor(h['wrist_world_mm'],dtype=torch.float32)
        if h['side']:wrist[:,0]*=-1
        with torch.inference_mode():xyz=skin_landmarks(hand_model,torch.tensor(h['joint_angles']),wrist).numpy()
        camera_reports=[];panels=[]
        for name,cam in cameras.items():
            camera=cam.copy(camera_to_world_xf=world@cam.camera_to_world_xf);uv=camera.eye_to_window(camera.world_to_eye(xyz));points=np.full((21,2),np.nan)
            for native,mp in COMMON:points[mp]=uv[native]
            matrix,size=crop_matrix(points);center=np.array([112.,112.,1.])@matrix.T
            crop=StereoCrop(cam,center,size,h['side']);image=images[name][0]
            start=time.perf_counter();result,pixels=estimate(model,crop,image);elapsed=(time.perf_counter()-start)*1000
            zero,_=estimate(model,crop,np.zeros_like(image));others=[]
            for background in protocol['backgrounds']:
                if background['case']==case['name'] and background['camera']==name:
                    bcrop=StereoCrop(cam,background['center'],background['size'],h['side']);bresult,bpixels=estimate(model,bcrop,image);others.append(dict(protocol=background,result=bresult));cv2.imwrite(str(a.output/(case['name']+'-'+name+'-background.png')),cv2.resize(bpixels,(384,384)))
            verifier=next(v for v in source_report['before_verifier'] if v['camera']==name)
            camera_reports.append(dict(camera=name,roi_center=center.tolist(),roi_size=size,result=result,zero=zero,backgrounds=others,keypoint_ms=elapsed,reference_ncnn_presence=verifier['presence'],raw_image_sha256=hashlib.sha256(image.tobytes()).hexdigest()))
            rawpanel=cv2.resize(pixels,(384,384));panel=draw(pixels,result.get('points_crop',[]),(0,255,0))
            cv2.imwrite(str(a.output/(case['name']+'-'+name+'-crop.png')),pixels)
            cv2.imwrite(str(a.output/(case['name']+'-'+name+'-landmarks.png')),cv2.resize(panel,(384,384)))
            whole=draw(image,result.get('points_raw',[]),(0,255,0));whole=draw(whole,verifier['model_points'],(255,0,255));whole=draw(whole,points,(0,140,255))
            # Use fixed raw ROI rectangle for comparison; no fitting to result.
            x0=max(0,int(center[0]-size*.65));x1=min(image.shape[1],int(center[0]+size*.65));y0=max(0,int(center[1]-size*.65));y1=min(image.shape[0],int(center[1]+size*.65))
            if x1<=x0 or y1<=y0:raise ValueError('Empty review crop')
            view=cv2.resize(whole[y0:y1,x0:x1],(384,384));combined=cv2.hconcat([cv2.cvtColor(rawpanel,cv2.COLOR_GRAY2BGR),cv2.resize(panel,(384,384)),view]);cv2.putText(combined,name+' p='+str(round(result.get('presence',0),3)),(8,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1);panels.append(combined)
        cv2.imwrite(str(a.output/(case['name']+'-review.jpg')),cv2.vconcat(panels));reports.append(dict(case=case,seconds=seconds,stamp=stamp,views=camera_reports));print(case['name'],[(v['camera'],v['result'].get('presence'),v['result']['active']) for v in camera_reports],flush=True)
    summary=dict(scope=protocol['scope'],legend='Rows: raw stereographic crop / greenMercury / rawcamera crop with greenMercury, magentan cnn, orangeUmeTrack. No calibrated ground truth.',runtime=model.version,threads=1,reports=reports,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.glob('*.py')})
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
