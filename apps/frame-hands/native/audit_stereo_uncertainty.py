"""Diagnose current stereo sensitivity versus published native geometry, offline only."""
import argparse,json,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'exploratory'))
from frame_data import read_audit,load_calibration
from compare_replay import source_hands
from audit_native_lengths import PARENTS,FINGERS,sha,stats
from stereo_uncertainty import sensitivity

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('session','upstream','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.upstream))
    if '143912Z' in str(a.session):raise ValueError('Holdout excluded')
    a.output.mkdir(parents=True,exist_ok=False)
    source=source_hands(a.session);audits=read_audit(a.session/'frame/live-views.txt');cameras=load_calibration(a.session/'frame/snapshot/device/persist')
    rows=[];skips={};bands={k:[] for k in ('under_2','2_to_5','5_to_10','10_plus')}
    with (a.output/'geometry.jsonl').open('x') as output:
        for stamp,ob in sorted(audits.items()):
            for ident,h in source.get(ob['mono_ns'],{}).items():
                views=[v for v in ob['views'] if v['id']==ident and v['camera'] in cameras]
                if not h['stereo'] or len(views)!=2 or len({v['camera'] for v in views})!=2:continue
                try:points,cov,info=sensitivity(cameras,views)
                except ValueError as e:skips[str(e)]=skips.get(str(e),0)+1;continue
                delta=points[FINGERS]-points[PARENTS[FINGERS]];length=np.linalg.norm(delta,axis=1)
                unit=delta/np.maximum(length[:,None],1e-8)
                var=np.einsum('ni,nij,nj->n',unit,cov[FINGERS]+cov[PARENTS[FINGERS]],unit)
                sigma=np.sqrt(np.maximum(var,0))
                old=np.linalg.norm(h['points'][FINGERS]-h['points'][PARENTS[FINGERS]],axis=1)
                row=dict(stamp=stamp,mono_ns=ob['mono_ns'],source_id=ident,side=h['side'],presence=[v['presence'] for v in views],angle_degrees=info['angle_degrees'].tolist(),major_axis_mm_per_pixel=info['major_axis_mm_per_pixel'].tolist(),ray_distance_mm=info['ray_distance_mm'].tolist(),reprojection_px=info['reprojection_px'].tolist(),bone_sigma_mm_per_pixel=sigma.tolist(),bone_length_mm=length.tolist(),published_bone_length_mm=old.tolist(),source_point_difference_mm=np.linalg.norm(points-h['points'],axis=1).tolist())
                output.write(json.dumps(row)+'\n');rows.append(row)
                for j,angle in enumerate(info['angle_degrees']):
                    key='under_2' if angle<2 else ('2_to_5' if angle<5 else ('5_to_10' if angle<10 else '10_plus'))
                    bands[key].append((info['major_axis_mm_per_pixel'][j],float(np.linalg.norm(points[j]-h['points'][j])),float(info['reprojection_px'][:,j].mean())))
    if not rows:raise ValueError('No stereo geometry')
    flatten=lambda k:np.array([r[k] for r in rows]).reshape(-1)
    sigma=flatten('bone_sigma_mm_per_pixel');length=flatten('bone_length_mm');pixel=flatten('reprojection_px')
    report=dict(scope=__doc__,session=a.session.name,stereo_observations=len(rows),rejected=skips,assumption='Independent one-pixel isotropic observation noise; neither detector confidence calibration nor physical error bounds.',metrics={k:stats(flatten(k)) for k in ('angle_degrees','major_axis_mm_per_pixel','bone_sigma_mm_per_pixel','reprojection_px','source_point_difference_mm')},bone_fraction_sigma_over_10pct=float(np.mean(sigma>length*.10)),bone_fraction_sigma_over_25pct=float(np.mean(sigma>length*.25)),angle_bands={k:dict(joints=len(v),major_axis_mm_per_pixel=stats([x[0] for x in v]),published_difference_mm=stats([x[1] for x in v]),reprojection_px=stats([x[2] for x in v])) for k,v in bands.items()},provenance=dict(inputs={n:sha(a.session/n) for n in ('relay-trace.bin','frame/live-views.txt')},sources={n:sha(BASE/n) for n in ('stereo_uncertainty.py','audit_stereo_uncertainty.py')}),limitations=['Independent triangulation is not anatomical truth.','Published source includes temporal processing; disagreement contains intentional smoothing/lag.','Model-correlated errors, camera timing and calibration error are not modeled.','No recording or inference rerun; no hardware performance claim.'])
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
