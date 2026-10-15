"""Offline per-view left/right model-hypothesis probe with no learned memory.

Annotations select reviewed frames and score output only. Each observed crop is
inferred independently under both chiralities. Reprojection disagreement is a
hypothesis score, not anatomical 3D ground truth or calibrated confidence.
"""
import argparse,io,json,sys
from pathlib import Path
import cv2,numpy as np,torch
from frame_data import read_audit,load_calibration
from live_backend import Backend
from replay_mesh_trial import packets
from replay_umetrack import COMMON
from run_mesh_trial import pose_matrix

class ForcedSide:
    def __init__(self,side):self.side=side
    def assign(self,observation):
        view=observation['views'][0];ident=view['id']
        return {self.side:ident},{ident:[view]},{ident:view['right']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',type=Path,required=True);p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.upstream));a.output.mkdir(parents=True,exist_ok=False)
    labels={(r['raw_ns'],r['camera'],r['source_id']):r['physical_side'] for r in json.loads(a.labels.read_text())['labels']}
    stamps={k[0] for k in labels}
    records={r['capture_raw_ns']:r for line in (a.session/'predictions.jsonl').open() if (r:=json.loads(line)) and r['capture_raw_ns'] in stamps}
    backend=Backend(a.upstream,load_calibration(a.session/'calibration'))
    rows=[];landmarks=[x for x,y in COMMON];detector=[y for x,y in COMMON]
    for meta,blobs in packets(a.session/'transport.bin'):
        stamp=meta.get('stamp');record=records.get(stamp)
        if record is None or not record['capture_pose']['valid']:continue
        obs=next(iter(read_audit(io.StringIO(meta['audit'])).values()))
        prepared={(v['camera'],v['id']):(cv2.imdecode(np.frombuffer(b,np.uint8),0),v) for v,b in zip(meta['crops'],blobs)}
        world=pose_matrix(record['capture_pose']['capture'])
        for view in obs['views']:
            key=(stamp,view['camera'],view['id'])
            if key not in labels or labels[key] is None or key[1:] not in prepared:continue
            costs=[];sigmas=[]
            for side in (0,1):
                backend.ownership=ForcedSide(side);backend.previous={}
                prediction=backend.predict(dict(obs,views=[view]),{},world,prepared)
                h=prediction['hands'][0];assert not h['memory']
                wrist=torch.tensor(h['wrist_world_mm'],dtype=torch.float32)
                if side:wrist[:,0]*=-1
                with torch.inference_mode():xyz=backend.landmarks(backend.hand,torch.tensor(h['joint_angles']),wrist).numpy()
                cam=backend.cameras[view['camera']].copy(camera_to_world_xf=world@backend.cameras[view['camera']].camera_to_world_xf)
                uv=cam.eye_to_window(cam.world_to_eye(xyz))
                error=np.linalg.norm(uv[landmarks]-np.asarray(view['points'])[detector],axis=1)
                costs.append(float(np.median(error)));sigmas.append(float(np.mean(h['uncertainty_raw'])))
            if not np.isfinite(costs).all():raise ValueError('Nonfinite hypothesis score')
            rows.append(dict(stamp=stamp,camera=view['camera'],source_id=view['id'],expected=labels[key],raw_right=view['right'],raw_choice=int(view['right']>.5),reprojection_choice=int(costs[1]<costs[0]),uncertainty_choice=int(sigmas[1]<sigmas[0]),reprojection_px=costs,mean_raw_sigmas=sigmas,relative_margin=abs(costs[0]-costs[1])/max(1e-6,sum(costs))))
    eligible=sum(v is not None for v in labels.values())
    if len(rows)!=eligible:raise ValueError(f'Only {len(rows)} of {eligible} labeled views evaluated')
    summary=dict(scope=__doc__,views=len(rows),correct={method:sum(r[method]==r['expected'] for r in rows) for method in ['raw_choice','reprojection_choice','uncertainty_choice']},margin_bins=[])
    for low,high in [(0,.1),(.1,.25),(.25,.5),(.5,1.000001)]:
        subset=[r for r in rows if low<=r['relative_margin']<high]
        summary['margin_bins'].append(dict(low=low,high=high,views=len(subset),correct=sum(r['reprojection_choice']==r['expected'] for r in subset)))
    (a.output/'rows.json').write_text(json.dumps(rows,indent=2)+'\n');(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
