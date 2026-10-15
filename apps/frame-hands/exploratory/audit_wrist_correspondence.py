"""Diagnose wrist-vs-knuckle correspondence; never change tracking geometry."""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np
from frame_data import read_audit,load_calibration
from audit_hand_scale import triangulate_palms
from wrist_refinement import PALM_NATIVE


def similarity(reference,observed):
    a=np.asarray(reference,float);b=np.asarray(observed,float)
    x=a-a.mean(0);y=b-b.mean(0);u,s,vt=np.linalg.svd(x.T@y)
    sign=np.ones(3);sign[-1]=np.linalg.det(u@vt)
    rotation=(u*sign)@vt;scale=float(s@sign/np.sum(x*x));translation=b.mean(0)-scale*a.mean(0)@rotation
    aligned=scale*a@rotation+translation
    return scale,rotation,translation,float(np.sqrt(np.mean(np.sum((aligned-b)**2,axis=1))))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    summary=json.loads((a.audit/'summary.json').read_text());session=Path(summary['args']['session']);upstream=Path(summary['args']['upstream']);sys.path.insert(0,str(upstream))
    cal=session/'calibration';cameras=load_calibration(cal if cal.exists() else session/'frame/snapshot/device/persist');obs=read_audit(session/'frame/live-views.txt')
    rest=np.asarray(json.loads((upstream/'dataset/generic_hand_model.json').read_text())['landmark_rest_positions'])[PALM_NATIVE]
    results=[]
    for line in (a.audit/'rows.jsonl').open():
        row=json.loads(line);selected=[]
        for source in row['source_views']:selected.append(next(v for v in obs[row['stamp']]['views'] if (v['camera'],v['id'])==(source['camera'],source['source_id'])))
        geometry,reason=triangulate_palms(cameras,selected)
        if geometry is None:raise ValueError('Frozen geometry changed: '+reason)
        native=rest.copy()
        if row['side']:native[:,0]*=-1
        measured=geometry['points_mm'];scale,rotation,translation,rms=similarity(native[1:],measured[1:])
        wrist=scale*native[0]@rotation+translation;axis=measured[1:].mean(0)-wrist;axis/=np.linalg.norm(axis);delta=measured[0]-wrist
        along=float(delta@axis);lateral=float(np.linalg.norm(delta-along*axis))
        results.append(dict(stamp=row['stamp'],seconds=row['seconds'],side=row['side'],knuckle_similarity_scale=scale,knuckle_rms_mm=rms,wrist_offset_toward_knuckles_mm=along,wrist_offset_lateral_mm=lateral,geometry_all_palm_scale=row['least_squares_scale'],learned_scale=row['learned_scale']))
    report={}
    for side in (0,1):
        rows=[r for r in results if r['side']==side];good=[r for r in rows if r['knuckle_rms_mm']<=3]
        report[str(side)]={}
        for label,group in [('all',rows),('knuckle_fit_rms_at_most_3mm',good)]:
            report[str(side)][label]=dict(count=len(group),**{key:np.percentile([r[key] for r in group],[5,50,95]).tolist() if group else None for key in ['knuckle_similarity_scale','knuckle_rms_mm','wrist_offset_toward_knuckles_mm','wrist_offset_lateral_mm']})
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'rows.json').write_text(json.dumps(results)+'\n')
    (a.output/'summary.json').write_text(json.dumps(dict(scope=__doc__+' Four-knuckle fit is extrapolated to wrist; noisy/nonrigid landmarks and near-planar geometry prevent a physical-origin conclusion. 3mm subgroup fixed before outcomes.',audit=str(a.audit),sides=report,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
