"""Paired offline learned-prior ablation using exclusively earlier stereo estimates."""
import argparse,hashlib,json,shutil,sys,time
from pathlib import Path
import cv2,numpy as np,torch
from frame_data import Recording,load_calibration
from run_mesh_trial import pose_matrix
from probe_roi_presence import crop_matrix
from replay_umetrack import COMMON
from mercury_probe import MercuryModels
from mercury_keypoints import StereoCrop,estimate
from mercury_prior import EarlierPosePrior,stereo_world
from probe_mercury_keypoints import draw


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['protocol','upstream','models','runtime','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();sys.path[:0]=[str(a.runtime.resolve()),str(a.upstream.resolve())]
    from lib.common.hand import HandModel
    from lib.common.hand_skinning import skin_landmarks
    torch.set_num_threads(1);cv2.setNumThreads(1);a.output.mkdir(parents=True,exist_ok=False)
    protocol=json.loads(a.protocol.read_text());shutil.copy2(a.protocol,a.output/'protocol.json')
    snapshot=a.output/'snapshot';snapshot.mkdir()
    for name in ['mercury_probe.py','mercury_keypoints.py','mercury_prior.py','probe_mercury_prior.py','frame_data.py','probe_roi_presence.py']:
        shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    raw=json.loads((a.upstream/'dataset/generic_hand_model.json').read_text());hand_model=HandModel(**{k:torch.tensor(v) if isinstance(v,list) else v for k,v in raw.items()})
    model=MercuryModels(a.models);contexts={};histories={};reports=[]
    # Rebuild exact native128px controls; display PNGs were enlarged for review.
    bg_protocol=json.loads((Path(protocol['background_root'])/'protocol.json').read_text())
    bg_case=next(c for c in bg_protocol['cases'] if c['name']=='latest-open-left');bg_session=Path(bg_case['session'])
    bg_row=json.loads((Path(bg_protocol['case_root'])/'latest-open-left/baseline.jsonl').read_text());bg_rec=Recording(bg_session/'frame/sets.bin')
    bg_seconds=(bg_row['capture_raw_ns']-bg_rec.index[0][0])/1e9;_,bg_stamp,bg_images=next(bg_rec.frames(bg_seconds-.000001,bg_seconds+.000001))
    if bg_stamp!=bg_row['capture_raw_ns']:raise ValueError('Background capture mismatch')
    bg_cal=bg_session/'calibration';bg_cams=load_calibration(bg_cal if bg_cal.exists() else bg_session/'frame/snapshot/device/persist');backgrounds={}
    for bg in bg_protocol['backgrounds']:
        name=bg['camera'];backgrounds[name]=StereoCrop(bg_cams[name],bg['center'],bg['size'],bg_case['side']).sample(bg_images[name][0])[0]
        cv2.imwrite(str(a.output/(name+'-native-background.png')),backgrounds[name])
    for case in protocol['cases']:
        group=case['name'].rsplit('-',1)[0];history=histories.setdefault(group,EarlierPosePrior());folder=Path(protocol['case_root'])/case['name']
        row=json.loads((folder/'baseline.jsonl').read_text());source_report=json.loads((folder/'report.json').read_text());h=next(h for h in row['hands'] if h['side']==case['side']);session=Path(case['session'])
        if str(session) not in contexts:
            cal=session/'calibration';contexts[str(session)]=(Recording(session/'frame/sets.bin'),load_calibration(cal if cal.exists() else session/'frame/snapshot/device/persist'))
        rec,cameras=contexts[str(session)];stamp=row['capture_raw_ns'];seconds=(stamp-rec.index[0][0])/1e9;_,actual,images=next(rec.frames(seconds-.000001,seconds+.000001))
        if stamp!=actual:raise ValueError('Capture mismatch')
        generation=(h['side'],h['source_id']);prior_source_stamps=[r[0] for r in history.rows];prior,prior_reason=history.predict(stamp,generation,bool(row['capture_pose']['valid']))
        if not row['capture_pose']['valid']:
            history.observe(stamp,generation,None);reports.append(dict(case=case,stamp=stamp,seconds=seconds,reason=prior_reason,views=[]));continue
        world=pose_matrix(row['capture_pose']['capture']);worldcams={name:cam.copy(camera_to_world_xf=world@cam.camera_to_world_xf) for name,cam in cameras.items()};wrist=torch.tensor(h['wrist_world_mm'],dtype=torch.float32)
        if h['side']:wrist[:,0]*=-1
        with torch.inference_mode():xyz=skin_landmarks(hand_model,torch.tensor(h['joint_angles']),wrist).numpy()
        rows=[];panels=[]
        for name,cam in cameras.items():
            camera=worldcams[name];uv=camera.eye_to_window(camera.world_to_eye(xyz));points=np.full((21,2),np.nan)
            for native,mp in COMMON:points[mp]=uv[native]
            try:
                matrix,size=crop_matrix(points);center=np.array([112.,112.,1.])@matrix.T;crop=StereoCrop(cam,center,size,h['side'])
            except ValueError as error:
                rows.append(dict(camera=name,skipped=str(error),prior_used=False));continue
            image=images[name][0];encoded=None;reason=prior_reason
            if prior is not None:
                try:
                    encoded=crop.normalized_coordinates(camera.world_to_eye(prior))
                    if np.max(np.abs(encoded))>1:encoded=None;reason='Predicted joint outside fixed current crop'
                except ValueError as error:encoded=None;reason=str(error)
            start=time.perf_counter()
            try:
                off,pixels=estimate(model,crop,image);on,_=estimate(model,crop,image,prior=encoded)
            except ValueError as error:
                rows.append(dict(camera=name,skipped=str(error),prior_used=False));continue
            elapsed=(time.perf_counter()-start)*1000
            bg_off,_=estimate(model,crop,image,pixels_override=backgrounds[name]);bg_on,_=estimate(model,crop,image,prior=encoded,pixels_override=backgrounds[name]);zero,_=estimate(model,crop,image,prior=encoded,pixels_override=np.zeros((128,128),np.uint8))
            rows.append(dict(camera=name,prior_used=encoded is not None,prior_reason=reason,prior_normalized=None if encoded is None else encoded.tolist(),off=off,on=on,background_off=bg_off,background_on=bg_on,zero=zero,paired_ms=elapsed))
            index=int(case['name'].rsplit('-',1)[1])
            if index in (0,10,20,28,29):
                panels.append(cv2.hconcat([cv2.cvtColor(pixels,cv2.COLOR_GRAY2BGR),draw(pixels,off['points_crop'],(0,140,255)),draw(pixels,on['points_crop'],(0,255,0)),draw(backgrounds[name],bg_on['points_crop'],(0,255,0))]))
        # Current independent observations enter only after all paired inference.
        verifier=source_report['before_verifier']
        if len(verifier)==2 and all(v['presence']>=.5 for v in verifier):
            observed,quality=stereo_world(worldcams,{v['camera']:v['model_points'] for v in verifier})
        else:observed=None;quality=dict(reason='Weak or missing independent stereo view')
        history.observe(stamp,generation,observed)
        reports.append(dict(case=case,stamp=stamp,seconds=seconds,generation=generation,prior_source_stamps=prior_source_stamps,prior_reason=prior_reason,prior_world_mm=None if prior is None else prior.tolist(),current_seed_quality=quality,views=rows))
        if panels:cv2.imwrite(str(a.output/(case['name']+'-review.jpg')),cv2.resize(cv2.vconcat(panels),(1536,384*len(panels))))
    summary=dict(scope=protocol['scope'],legend='Rows follow camera order; columns rawcrop/orangepriorOFF/greenpriorON/backgroundwithpriorON. Background control pixels are frozen from earlier reviewed hand-free monitor/window crops. All skeleton hypotheses are drawn even when presence gate rejects; use per-view active/presence fields for admission.',runtime=model.version,threads=1,background_sha256={name:hashlib.sha256(pixels.tobytes()).hexdigest() for name,pixels in backgrounds.items()},reports=reports,source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in snapshot.glob('*.py')})
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    views=[v for r in reports for v in r['views'] if 'off' in v];print(json.dumps(dict(frames=len(reports),camera_crops=len(views),prior_used=sum(v['prior_used'] for v in views),real_off_active=sum(v['off']['active'] for v in views),real_on_active=sum(v['on']['active'] for v in views),background_off_active=sum(v['background_off']['active'] for v in views),background_on_active=sum(v['background_on']['active'] for v in views),blank_active=sum(v['zero']['active'] for v in views))))
if __name__=='__main__':main()
