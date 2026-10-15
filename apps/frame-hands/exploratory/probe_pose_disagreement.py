"""Compare model projections to detector landmarks on manually identified hands.

This is a disagreement probe, NOT physical accuracy or uncertainty calibration:
both systems can be wrong and detector points supplied the model's image crops.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from frame_data import read_audit,load_calibration
from replay_umetrack import COMMON
from run_mesh_trial import pose_matrix


def ranks(values):
    _,inverse,counts=np.unique(values,return_inverse=True,return_counts=True)
    ends=np.cumsum(counts)
    return (ends-(counts+1)/2)[inverse]


def correlation(x,y):
    if len(x)<3 or np.std(x)==0 or np.std(y)==0:return None
    return float(np.corrcoef(ranks(x),ranks(y))[0,1])


def summary(rows):
    if not rows:return dict(views=0)
    error=np.array([r['median_disagreement_px'] for r in rows])
    sigma=np.array([r['mean_raw_sigma'] for r in rows])
    order=np.argsort(sigma)
    return dict(views=len(rows),median_disagreement_px_p50_p95=np.percentile(error,[50,95]).tolist(),
        raw_sigma_p05_p50_p95=np.percentile(sigma,[5,50,95]).tolist(),
        spearman_sigma_vs_disagreement=correlation(sigma,error),
        sigma_quartiles=[dict(views=len(i),mean_sigma=float(sigma[i].mean()),median_disagreement_px=float(np.median(error[i]))) for i in np.array_split(order,4) if len(i)])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',type=Path,required=True);p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True);p.add_argument('--labels',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.upstream));torch.set_num_threads(1)
    from lib.common.hand import HandModel
    from lib.common.hand_skinning import skin_landmarks
    raw=json.loads((a.upstream/'dataset/generic_hand_model.json').read_text())
    hand=HandModel(**{k:torch.tensor(v) if isinstance(v,list) else v for k,v in raw.items()})
    labels={}
    for path in a.labels:
        for l in json.loads(path.read_text())['labels']:
            key=(l['raw_ns'],l['camera'],l['source_id'])
            if key in labels and labels[key]!=l['physical_side']:raise ValueError('Conflicting labels')
            labels[key]=l['physical_side']
    stamps={k[0] for k in labels};audit=read_audit(a.session/'frame/live-views.txt');cameras=load_calibration(a.session/'calibration')
    model_indices=[x for x,y in COMMON];detector_indices=[y for x,y in COMMON]
    rows=[];rejected_wrong_side=0
    for line in a.predictions.open():
        r=json.loads(line);stamp=r['capture_raw_ns']
        if stamp not in stamps or not r['capture_pose']['valid']:continue
        world=pose_matrix(r['capture_pose']['capture']);views=audit[stamp]['views']
        for h in r['hands']:
            selected={(v['camera'],v['source_id']) for v in h['source_views']} if 'source_views' in h else {(v['camera'],v['id']) for v in views if v['id']==h['source_id']}
            identified=[v for v in views if (v['camera'],v['id']) in selected and (stamp,v['camera'],v['id']) in labels]
            correct=[v for v in identified if labels[(stamp,v['camera'],v['id'])]==h['side']]
            rejected_wrong_side+=len(identified)-len(correct)
            if not correct:continue
            wrist=torch.tensor(h['wrist_world_mm'],dtype=torch.float32)
            if h['side']:wrist[:,0]*=-1
            with torch.inference_mode():xyz=skin_landmarks(hand,torch.tensor(h['joint_angles']),wrist).numpy()
            sigma=np.asarray(h['uncertainty_raw'])[model_indices]
            for v in correct:
                cam=cameras[v['camera']].copy(camera_to_world_xf=world@cameras[v['camera']].camera_to_world_xf)
                uv=cam.eye_to_window(cam.world_to_eye(xyz))
                differences=np.linalg.norm(uv[model_indices]-np.asarray(v['points'])[detector_indices],axis=1)
                if not np.isfinite(differences).all():raise ValueError('Invalid projected pose')
                rows.append(dict(stamp=stamp,camera=v['camera'],source_id=v['id'],side=h['side'],median_disagreement_px=float(np.median(differences)),p95_disagreement_px=float(np.percentile(differences,95)),mean_raw_sigma=float(sigma.mean()),wrist_disagreement_px=float(differences[0])))
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    result=dict(scope='Model/detector disagreement on manually verified identities; not physical accuracy, calibrated uncertainty or independent landmark ground truth',excluded_wrong_side_or_nonhand_views=rejected_wrong_side,**summary(rows))
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
