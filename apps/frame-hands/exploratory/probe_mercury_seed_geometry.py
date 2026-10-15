"""Compare external pixel-box and ray-domain seeds on exact saved captures."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
import cv2
import numpy as np
import torch
from frame_data import Recording, load_calibration
from mercury_keypoints import StereoCrop
from probe_roi_presence import crop_matrix
from replay_umetrack import COMMON
from run_mesh_trial import pose_matrix


def stats(values):
    if not values:
        return None
    return dict(count=len(values), minimum=float(min(values)), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)), maximum=float(max(values)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','upstream','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.upstream.resolve()))
    from lib.common.hand import HandModel
    from lib.common.hand_skinning import skin_landmarks
    torch.set_num_threads(1);cv2.setNumThreads(1)
    protocol=json.loads((a.source/'protocol.json').read_text())
    path=Path(protocol['baseline'])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==protocol['baseline_sha256']
    baseline={}
    with path.open() as f:
        for line in f:
            row=json.loads(line)
            baseline[row['capture_raw_ns']]=dict(capture_pose=row['capture_pose'],hands=row['hands'])
            for hand in row['hands']:hand.pop('vertices_world_mm',None)
    original={}
    with (a.source/'frames.jsonl').open() as f:
        for line in f:
            row=json.loads(line)
            for hand in row['observations']:
                for v in hand['views']:
                    original[(row['stamp'],hand['side'],v['camera'])]=dict(skipped=v.get('skipped'),crop_sha256=v.get('crop_sha256'))
    a.output.mkdir(parents=True,exist_ok=False)
    declared=dict(source=str(a.source.resolve()),baseline_sha256=protocol['baseline_sha256'],start=0,end=30,
        control='Original external distorted-pixel rectangle inverted at center/corners',
        candidate='Existing pinned native ray-domain geometry with expansion1.7 and palm floor on the same CURRENT external3Dpose; not predicted feedback',
        missing_joint='UmeTrack has no homologous thumb CMC. Mercury slot1 duplicates the wrist solely for crop bounds; it is not supplied to inference or fitting as an observation.',
        policy='Keep whole-crop forward-hemisphere and positive seed-Z requirements unchanged. No clipping, shrink sweep, model inference or size optimization.',
        outcome='Count original failures and rescued/lost views and pairs; measure seed containment/radius; no tracking accuracy claim.')
    (a.output/'protocol.json').write_text(json.dumps(declared,indent=2)+'\n')
    snapshot=a.output/'snapshot';snapshot.mkdir()
    for name in ('probe_mercury_seed_geometry.py','mercury_keypoints.py','frame_data.py','probe_roi_presence.py'):
        shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    session=Path(protocol['session']);rec=Recording(session/'frame/sets.bin');cameras=load_calibration(session/'calibration')
    raw=json.loads((a.upstream/'dataset/generic_hand_model.json').read_text())
    model=HandModel(**{k:torch.tensor(v) if isinstance(v,list) else v for k,v in raw.items()})
    counts=Counter();reasons=Counter();ratios=[];containment={'box':[],'rays':[]};saved=Counter()
    with (a.output/'frames.jsonl').open('x') as out:
        for seconds,stamp,images in rec.frames(0,30):
            row=baseline.get(stamp);counts['raw_frames']+=1
            if row is None or not row['capture_pose']['valid']:
                counts['missing_or_invalid_baseline']+=1;continue
            world=pose_matrix(row['capture_pose']['capture'])
            for hand in row['hands']:
                side=hand['side'];wrist=torch.tensor(hand['wrist_world_mm'],dtype=torch.float32)
                if side:wrist[:,0]*=-1
                with torch.inference_mode():xyz=skin_landmarks(model,torch.tensor(hand['joint_angles']),wrist).numpy()
                hand_results=[]
                for name,cam in cameras.items():
                    camera=cam.copy(camera_to_world_xf=world@cam.camera_to_world_xf)
                    eye=camera.world_to_eye(xyz);uv=camera.eye_to_window(eye)
                    points=np.full((21,2),np.nan);seed=np.zeros((21,3))
                    for native,mp in COMMON:points[mp]=uv[native];seed[mp]=eye[native]
                    seed[1]=seed[0]
                    branches={};pixels={}
                    for branch in ('box','rays'):
                        try:
                            if branch=='box':
                                matrix,size=crop_matrix(points);center=np.array([112.,112.,1.])@matrix.T;crop=StereoCrop(cam,center,size,side);info=dict(center=center.tolist(),size=size)
                            else:crop,info=StereoCrop.from_hand_geometry(cam,seed,side)
                            tile,valid=crop.sample(images[name][0]);pixels[branch]=tile
                            xy=crop.normalized_coordinates(seed);inside=(np.abs(xy)<=1).all(axis=1);inside=np.delete(inside,1)
                            containment[branch].append(float(inside.mean()))
                            result=dict(valid=True,radius=crop.radius,rotation=crop.rotation.tolist(),valid_fraction=valid,seed_containment=float(inside.mean()),info=info,crop_sha256=hashlib.sha256(tile.tobytes()).hexdigest())
                        except ValueError as e:result=dict(valid=False,reason=str(e));reasons[branch+': '+str(e)]+=1
                        branches[branch]=result;counts[branch+'_valid_views']+=int(result['valid'])
                    previous=original[(stamp,side,name)];old=branches['box']
                    assert previous['skipped']==old.get('reason'), 'Original admission changed'
                    if old['valid']:assert previous['crop_sha256']==old['crop_sha256'], 'Original pixels changed'
                    counts['exact_original_views']+=1
                    rescued=not old['valid'] and branches['rays']['valid'];lost=old['valid'] and not branches['rays']['valid']
                    counts['rescued_views']+=int(rescued);counts['lost_views']+=int(lost)
                    if all(v['valid'] for v in branches.values()):ratios.append(branches['rays']['radius']/old['radius'])
                    result=dict(seconds=seconds,stamp=stamp,side=side,camera=name,source_id=hand['source_id'],seed_eye_mm=seed.tolist(),minimum_seed_z_mm=float(seed[:,2].min()),raw_sha256=hashlib.sha256(images[name][0].tobytes()).hexdigest(),branches=branches)
                    hand_results.append(result);out.write(json.dumps(result,allow_nan=False)+'\n')
                    kind='rescued' if rescued else 'lost' if lost else 'paired'
                    if saved[kind]<6 and (kind!='paired' or seconds>9):
                        rawtile=cv2.cvtColor(images[name][0],cv2.COLOR_GRAY2BGR)
                        for point in points[np.isfinite(points).all(axis=1)]:cv2.circle(rawtile,tuple(np.rint(point).astype(int)),3,(255,255,0),-1)
                        tiles=[cv2.resize(rawtile,(512,512))]
                        for branch in ('box','rays'):
                            tile=cv2.cvtColor(pixels[branch],cv2.COLOR_GRAY2BGR) if branch in pixels else np.zeros((128,128,3),np.uint8)
                            tile=cv2.resize(tile,(512,512),interpolation=cv2.INTER_NEAREST)
                            cv2.putText(tile,branch+' '+str(round(seconds,3)),(5,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
                            if not branches[branch]['valid']:cv2.putText(tile,branches[branch]['reason'],(5,40),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,255),1)
                            tiles.append(tile)
                        cv2.imwrite(str(a.output/(kind+'-'+str(saved[kind])+'.jpg')),cv2.hconcat(tiles));saved[kind]+=1
                before=all(r['branches']['box']['valid'] for r in hand_results);after=all(r['branches']['rays']['valid'] for r in hand_results)
                counts['hand_contexts']+=1;counts['box_valid_pairs']+=int(before);counts['rays_valid_pairs']+=int(after)
                counts['rescued_pairs']+=int(after and not before);counts['lost_pairs']+=int(before and not after)
            if counts['raw_frames']%150==0:out.flush();print('frames',counts['raw_frames'],'rescued views',counts['rescued_views'],'lost views',counts['lost_views'],flush=True)
    summary=dict(protocol=declared,counts=counts,reasons=reasons,radius_ratio=stats(ratios),seed_containment={k:stats(v) for k,v in containment.items()},reviews=saved,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.glob('*.py')},limitations='Geometric availability only. Current external3Dseeds and anatomical identity remain supplied. Seed containment is not coverage of the real hand. No ML inference, acquisition, scale calibration, pose or live accuracy result.')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
