"""Replay fixed-size Mercury fits with paired externally seeded crop constructions."""
import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import sys
import cv2
import numpy as np
from frame_data import Recording, load_calibration
from mercury_keypoints import StereoCrop, estimate
from mercury_probe import MercuryModels
from mercury_native_solver import NativeSolver, camera_transforms
from probe_mercury_solver import project, stats
from probe_mercury_keypoints import draw
from replay_mercury_solver import reset_reason, to_world
from run_mesh_trial import pose_matrix


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('geometry','source','upstream','dll','models','runtime','background','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();sys.path[:0]=[str(a.runtime.resolve()),str(a.upstream.resolve())];cv2.setNumThreads(1)
    protocol=json.loads((a.source/'protocol.json').read_text());a.output.mkdir(parents=True,exist_ok=False)
    declared=dict(geometry=str(a.geometry.resolve()),source=str(a.source.resolve()),start=0,end=30,size_m=.09,smoothing=2,
        branches=['box','rays'],external_seed=True,learned_temporal_prior=False,
        acceptance='Two active model views>=0.97 and native reprojection residual<=15',
        validation='Replay every original valid model crop and fixed-size fit exactly; independent histories for the two crops. Review new output on raw saved frames. Fixed background/blank inputs remain outside history.')
    (a.output/'protocol.json').write_text(json.dumps(declared,indent=2)+'\n');snapshot=a.output/'snapshot';snapshot.mkdir()
    for name in ('replay_mercury_seed_geometry.py','mercury_keypoints.py','mercury_native_solver.py','replay_mercury_solver.py','mercury_probe.py'):
        shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    shutil.copy2(a.dll.parent/'build.json',a.output/'build.json')
    baseline_path=Path(protocol['baseline']);assert hashlib.sha256(baseline_path.read_bytes()).hexdigest()==protocol['baseline_sha256']
    baseline={}
    with baseline_path.open() as f:
        for line in f:
            row=json.loads(line);baseline[row['capture_raw_ns']]=row['capture_pose']
    geometry={}
    with (a.geometry/'frames.jsonl').open() as f:
        for line in f:
            row=json.loads(line);geometry.setdefault(row['stamp'],{}).setdefault(row['side'],{})[row['camera']]=row
    session=Path(protocol['session']);rec=Recording(session/'frame/sets.bin');cameras=load_calibration(session/'calibration');pose,head_from_left=camera_transforms(cameras);model=MercuryModels(a.models)
    background_summary=json.loads((a.background/'summary.json').read_text());background={}
    for name in cameras:
        tile=cv2.imread(str(a.background/(name+'-native-background.png')),cv2.IMREAD_GRAYSCALE)
        assert tile.shape==(128,128) and hashlib.sha256(tile.tobytes()).hexdigest()==background_summary['background_sha256'][name]
        background[name]=tile
    counts=Counter();controls={};last_review=-1
    with ExitStack() as stack:
        previous={};old_camera={};solvers={(branch,side):stack.enter_context(NativeSolver(a.dll,pose,side)) for branch in ('box','rays') for side in (0,1)}
        output=stack.enter_context((a.output/'frames.jsonl').open('x'));old=stack.enter_context((a.source/'frames.jsonl').open())
        for seconds,stamp,images in rec.frames(0,30):
            reference=json.loads(next(old));assert reference['stamp']==stamp
            capture=baseline.get(stamp);valid=capture is not None and capture['valid'];world=pose_matrix(capture['capture']) if valid else None
            contexts=geometry.get(stamp,{});observations=[];fits=[]
            if valid:
                current=world.copy();current[:3,3]*=.001;current=current@head_from_left
            for side in (0,1):
                rows=contexts.get(side,{})
                for branch in ('box','rays'):
                    key=(branch,side);views={}
                    for name,row in rows.items():
                        construction=row['branches'][branch]
                        if not construction['valid']:continue
                        image=images[name][0];assert hashlib.sha256(image.tobytes()).hexdigest()==row['raw_sha256']
                        crop=StereoCrop.from_geometry(cameras[name],construction['rotation'],construction['radius'],side)
                        result,pixels=estimate(model,crop,image);assert hashlib.sha256(pixels.tobytes()).hexdigest()==construction['crop_sha256']
                        views[name]=result;counts[branch+'_model_views']+=1;counts[branch+'_active_views']+=int(result['active'])
                        prior_hand=next(h for h in reference['observations'] if h['side']==side);prior_view=next(v for v in prior_hand['views'] if v['camera']==name)
                        if branch=='box':assert result==prior_view['model'];counts['exact_original_model_outputs']+=1
                        observations.append(dict(branch=branch,side=side,camera=name,model=result,ncnn_points=prior_view.get('ncnn_points'),crop_sha256=construction['crop_sha256']))
                        if name not in controls:
                            bg,_=estimate(model,crop,image,pixels_override=background[name]);blank,_=estimate(model,crop,image,pixels_override=np.zeros((128,128),np.uint8))
                            controls[name]=dict(background=bg,blank=blank,sha256=hashlib.sha256(background[name].tobytes()).hexdigest())
                    if len(views)!=2 or not all(v['active'] for v in views.values()):previous[key]=None;continue
                    generation=(side,next(iter(rows.values()))['source_id']);reason=reset_reason(previous.get(key),stamp,generation,valid)
                    if reason is None:solvers[key].transport(np.linalg.inv(current)@old_camera[key])
                    result=solvers[key].run(views,reset=reason is not None,smoothing=2);accepted=result['native_active'] and result['passes_native_reprojection_gate'];previous[key]=(stamp,generation) if accepted else None;old_camera[key]=current
                    projected=project(result,head_from_left,cameras);disagreement={}
                    for name,uv in projected.items():
                        points=next(v['ncnn_points'] for v in observations if v['branch']==branch and v['side']==side and v['camera']==name)
                        if points is not None:disagreement[name]=stats(np.linalg.norm(uv-np.asarray(points),axis=1))
                    if branch=='box':
                        original=next(f for f in reference['fits'] if f['side']==side and f['branch']=='fixed')
                        assert result==original['fit'];counts['exact_original_native_fits']+=1
                    counts[branch+'_fits']+=1;counts[branch+'_accepted']+=int(accepted)
                    fits.append(dict(branch=branch,side=side,generation=generation,accepted=accepted,fit=result,reset_reason=reason,
                                     world_joints_mm=to_world(result,head_from_left,world).tolist(),projected={k:v.tolist() for k,v in projected.items()},ncnn_disagreement_px=disagreement))
            for side in (0,1):
                before=any(f['side']==side and f['branch']=='box' and f['accepted'] for f in fits);after=any(f['side']==side and f['branch']=='rays' and f['accepted'] for f in fits)
                counts['rescued_hand_fits']+=int(after and not before);counts['lost_hand_fits']+=int(before and not after)
            output.write(json.dumps(dict(seconds=seconds,stamp=stamp,observations=observations,fits=fits),allow_nan=False)+'\n');counts['frames']+=1
            if int(seconds)!=last_review:
                panels=[]
                for name in cameras:
                    image=images[name][0];columns=[cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)]
                    for branch in ('box','rays'):
                        canvas=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
                        for fit in fits:
                            if fit['branch']==branch and fit['accepted']:canvas=draw(canvas,fit['projected'][name],(255,255,0) if fit['side']==0 else (0,140,255))
                        columns.append(canvas)
                    panel=cv2.hconcat([cv2.resize(c,(512,round(512*image.shape[0]/image.shape[1]))) for c in columns]);cv2.putText(panel,name+' '+str(round(seconds,3))+' raw / box / rays',(5,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);panels.append(panel)
                cv2.imwrite(str(a.output/('review-'+str(int(seconds)).zfill(2)+'.jpg')),cv2.vconcat(panels));last_review=int(seconds)
            if counts['frames']%150==0:output.flush();print('frames',counts['frames'],'rescued fits',counts['rescued_hand_fits'],'lost',counts['lost_hand_fits'],flush=True)
    summary=dict(protocol=declared,counts=counts,controls=controls,runtime=model.version,dll_sha256=hashlib.sha256(a.dll.read_bytes()).hexdigest(),source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.glob('*.py')},limitations='External current3Dseeds/identity; no native acquisition. Two fixed hand-free texture controls are not independent absence-video accuracy. Residual gates and estimator agreement are not physical truth. No live change.')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(counts))


if __name__=='__main__':main()
