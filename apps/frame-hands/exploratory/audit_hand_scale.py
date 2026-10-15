"""Offline hand-size audit: calibrated stereo palms and native learned scale.

No personalized scale is applied. Current detections must be separated, strong,
and geometrically consistent. Both estimates remain model-dependent evidence.
"""
import argparse,hashlib,itertools,json,shutil,sys
from pathlib import Path
import numpy as np,torch
from frame_data import NAMES,read_audit,load_calibration,unproject,Recording
from live_backend import Backend
from replay_raw_chirality import prepare
from probe_wrist_refinement import landmarks
from wrist_refinement import PALM_NATIVE,PALM_DETECTOR

PAIRS=np.array(list(itertools.combinations(range(5),2)))


def triangulate_palms(cameras,views):
    if len(views)!=2 or {v['camera'] for v in views}!=set(NAMES):return None,'No distinct stereo pair'
    rays=[];origins=[]
    for v in views:
        cam=cameras[v['camera']];points=np.asarray(v['points'])[PALM_DETECTOR]
        if not np.isfinite(points).all():return None,'Nonfinite landmarks'
        rays.append(unproject(cam,points)@cam.camera_to_world_xf[:3,:3].T)
        origins.append(cam.camera_to_world_xf[:3,3])
    r,s=rays;delta=origins[1]-origins[0];dot=np.sum(r*s,1);denom=1-dot*dot
    if np.any(denom<np.sin(np.deg2rad(5))**2):return None,'Weak triangulation angle'
    ra=r@delta;sb=s@delta;ta=(ra-dot*sb)/denom;tb=(dot*ra-sb)/denom
    if np.any(np.r_[ta,tb]<100) or np.any(np.r_[ta,tb]>1000):return None,'Range outside audit bounds'
    pa=origins[0]+r*ta[:,None];pb=origins[1]+s*tb[:,None];miss=np.linalg.norm(pa-pb,axis=1)
    if np.max(miss)>5:return None,'Stereo ray disagreement'
    xyz=(pa+pb)*.5
    for v in views:
        cam=cameras[v['camera']];error=np.linalg.norm(cam.eye_to_window(cam.world_to_eye(xyz))-np.asarray(v['points'])[PALM_DETECTOR],axis=1)
        if error.max()>2:return None,'Stereo pixel disagreement'
    return dict(points_mm=xyz,ray_miss_mm=miss.tolist(),min_angle_degrees=float(np.rad2deg(np.arccos(np.abs(dot))).min())),'Eligible geometry'


def scale_ratios(reference,observed):
    reference=np.asarray(reference);observed=np.asarray(observed)
    a=np.linalg.norm(reference[PAIRS[:,0]]-reference[PAIRS[:,1]],axis=1)
    b=np.linalg.norm(observed[PAIRS[:,0]]-observed[PAIRS[:,1]],axis=1)
    if not np.isfinite(np.r_[a,b]).all() or a.min()<1 or b.min()<1:raise ValueError('Degenerate palm dimensions')
    scale=float(a@b/(a@a));ratios=b/a
    return dict(least_squares_scale=scale,median_pair_scale=float(np.median(ratios)),pair_scales=ratios.tolist(),relative_shape_residual=float(np.linalg.norm(b-a*scale)/np.linalg.norm(b)),reference_lengths_mm=a.tolist(),observed_lengths_mm=b.tolist())


def select_views(hand,observation):
    selected=[]
    for source in hand['source_views']:
        matches=[v for v in observation['views'] if (v['camera'],v['id'])==(source['camera'],source['source_id'])]
        if len(matches)!=1:return None,'Missing or duplicate source view'
        v=matches[0];q=v['right'] if hand['side'] else 1-v['right']
        if v['presence']<.9 or q<.8:return None,'Weak or contradictory raw identity'
        pts=np.asarray(v['points'])[PALM_DETECTOR];center=pts.mean(0);extent=np.linalg.norm(pts.max(0)-pts.min(0))
        for other in observation['views']:
            if other['camera']!=v['camera'] or other['id']==v['id'] or other['presence']<.5:continue
            op=np.asarray(other['points'])[PALM_DETECTOR];osize=np.linalg.norm(op.max(0)-op.min(0))
            if np.linalg.norm(op.mean(0)-center)<1.5*max(extent,osize):return None,'Nearby competing hand observation'
        selected.append(v)
    if len(selected)!=2:return None,'No stereo source'
    return sorted(selected,key=lambda v:NAMES.index(v['camera'])),'Selected'


def learned_scale(backend,side,views,images):
    prepared=prepare(backend.cameras,images,dict(views=views));pixels=[];intrinsics=[];extrinsics=[]
    for v in views:
        image,meta=prepared[v['camera'],v['id']];pixels.append(image[:,::-1].copy() if side else image);intrinsics.append(meta['intrinsics'])
        xf=np.asarray(meta['camera_to_head_mm']).copy()
        if side:xf[:,0]*=-1
        xf=np.linalg.inv(xf);xf[:3,3]*=.001;extrinsics.append(xf)
    device=next(backend.model.parameters()).device
    frame=backend.Frame(torch.tensor(np.stack(pixels)/255.,dtype=torch.float32,device=device),torch.tensor(np.stack(intrinsics),dtype=torch.float32,device=device),torch.tensor(np.stack(extrinsics),dtype=torch.float32,device=device))
    idx=torch.tensor([side],device=device);desc=backend.Desc(torch.tensor([[0,2]],device=device),idx,torch.tensor([False],device=device),idx)
    with torch.inference_mode():result=backend.model.regress_pose_pred_skel_scale(frame,desc)
    value=float(result.skel_scales.item())
    if not np.isfinite(value) or value<=0:raise ValueError('Invalid native learned scale')
    return value


def summarize(rows):
    out={}
    for side in (0,1):
        selected=[r for r in rows if r['side']==side];learned=[r for r in selected if r.get('learned_scale') is not None]
        def stats(values):return dict(count=len(values),p05_p25_p50_p75_p95=np.percentile(values,[5,25,50,75,95]).tolist()) if values else dict(count=0)
        out[str(side)]=dict(geometry=stats([r['least_squares_scale'] for r in selected]),shape_residual=stats([r['relative_shape_residual'] for r in selected]),learned=stats([r['learned_scale'] for r in learned]),geometry_on_learned_samples=stats([r['least_squares_scale'] for r in learned]))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('session','baseline','upstream','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if not torch.cuda.is_available():p.error('Separate CUDA environment required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sys.path.insert(0,str(a.upstream));a.output.mkdir(parents=True,exist_ok=False);snapshot=a.output/'snapshot';snapshot.mkdir()
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,snapshot/f.name)
    cal=a.session/'calibration';cal=cal if cal.exists() else a.session/'frame/snapshot/device/persist'
    b=Backend(a.upstream,load_calibration(cal));b.model.to('cuda');audit=read_audit(a.session/'frame/live-views.txt');first=min(audit)
    recording=Recording(a.session/'frame/sets.bin');last_learned={};rows=[];reasons={};counts=dict(frames=0,hands=0,geometry=0,learned=0)
    with a.baseline.open() as source,(a.output/'rows.jsonl').open('x') as trace:
        for line in source:
            row=json.loads(line);stamp=row['capture_raw_ns'];seconds=(stamp-first)/1e9;counts['frames']+=1
            for h in row['hands']:
                counts['hands']+=1;views,reason=select_views(h,audit[stamp])
                if views is not None:
                    geometry,reason=triangulate_palms(b.cameras,views)
                    if geometry is not None:
                        xyz=landmarks(b,h)[PALM_NATIVE]
                        measured=scale_ratios(xyz,geometry['points_mm']);value=None
                        if seconds-last_learned.get(h['side'],-1e6)>=.5:
                            raw=next(recording.frames(seconds-.00001,seconds+.00001))
                            if raw[1]!=stamp:raise ValueError('Wrong saved capture')
                            value=learned_scale(b,h['side'],views,raw[2]);last_learned[h['side']]=seconds;counts['learned']+=1
                        result=dict(stamp=stamp,seconds=seconds,side=h['side'],source_id=h['source_id'],source_views=h['source_views'],learned_scale=value,ray_miss_mm=geometry['ray_miss_mm'],min_angle_degrees=geometry['min_angle_degrees'],**measured)
                        rows.append(result);counts['geometry']+=1;trace.write(json.dumps(result,allow_nan=False)+'\n')
                reasons[reason]=reasons.get(reason,0)+1
    summary=dict(scope=__doc__+' Native learned scale uses no temporal memory and samples at most once per 0.5s per side; no ground-truth crops or future images.',counts=counts,reasons=reasons,sides=summarize(rows),runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,tf32=False),args={k:str(v) for k,v in vars(a).items()},calibration_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in cal.glob('*.json')},source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in snapshot.glob('*.py')})
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:summary[k] for k in ['counts','reasons','sides']},indent=2))

if __name__=='__main__':main()
