"""Verify correspondence controls and calibrated replay decisions independently."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
from audit_native_lengths import sha,stats
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'exploratory'))
from frame_data import load_calibration,read_audit,unproject


def synthetic(folder,profiles,evaluator):
    original=np.loadtxt(folder/'output.txt');data=np.loadtxt(folder/'output-v2.txt')
    assert data.shape==(980,97) and np.array_equal(np.delete(original,9,1),np.delete(data,9,1))
    points=np.empty((len(data),3,21,3))
    for side in (0,1):
        selected=data[:,1]==side;poses=data[selected,16:].reshape(-1,27)
        input_file=folder/f'audit-side{side}-input.txt';output_file=folder/f'audit-side{side}-points.txt'
        with input_file.open('x') as stream:np.savetxt(stream,np.c_[np.full(len(poses),side),poses],fmt='%.17g')
        subprocess.run([str(evaluator.resolve()),str((profiles/f'side{side}.fhrig').resolve()),str(input_file.resolve()),str(output_file.resolve())],check=True,capture_output=True,text=True)
        geometry=np.loadtxt(output_file);assert geometry.shape==(len(poses),64)
        points[selected]=geometry[:,1:].reshape(-1,3,21,3)
    distances=np.linalg.norm(points[:,1:]-points[:,:1],axis=-1)
    expected=np.stack((np.sqrt(np.mean(distances**2,axis=2)),distances.max(axis=2)),axis=2)
    actual=data[:,[10,11,13,14]].reshape(-1,2,2);assert np.max(abs(expected-actual))<1e-10
    groups=[]
    for family in (0,1):
        for mode in range(5):
            selected=(data[:,0]==family)&(data[:,2]==mode);x=data[selected];errors=expected[selected,:,0]
            if mode<2:
                assert np.all(x[:,5]==-1) and np.array_equal(x[:,16+27:16+54],x[:,16+54:])
            truth_step=[]
            if family:
                for side in (0,1):
                    mask=selected&(data[:,1]==side);p=points[mask];d=np.diff(p,axis=0)
                    truth_step.extend(np.sqrt(np.mean(np.sum((d[:,1:]-d[:,:1])**2,axis=3),axis=2)).tolist())
            groups.append(dict(family=('independent','causal_sequence')[family],mode=('clean','noise2px','swap_left','swap_right','swap_both')[mode],samples=len(x),chosen=int(sum(x[:,5]>=0)),correct_single_view_choices=int(sum(x[:,5]==mode-2)) if mode in (2,3) else None,truth_rmse_mm=[stats(errors[:,k]) for k in (0,1)],improvements_over1mm=int(sum(errors[:,0]-errors[:,1]>1)),regressions_over1mm=int(sum(errors[:,1]-errors[:,0]>1)),motion_error_mm=None if not family else [stats(np.array(truth_step)[:,k]) for k in (0,1)]))
    result=dict(cases=980,groups=groups,max_reconstructed_metric_difference=float(np.max(abs(expected-actual))),refactor_repeat_exact=True,output_sha256=sha(folder/'output-v2.txt'),limitations=['Truth and inverse fits share rig geometry; not real-image accuracy.','Static seeds are perturbed truth; sequence truth initializes frame0 only, then each arm feeds its own fitted history.','Sequence has moving wrist and five-frame index/middle articulation changes; no detector, head-pose noise or dropout.','Both-camera swapped labels are intentionally undetectable by epipolar disagreement.','No native performance claim.'])
    (folder/'synthetic-audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


def disagreement(cameras,points):
    names=('slam_left','slam_right');origins=[cameras[n].camera_to_world_xf[:3,3] for n in names]
    rays_eye=[unproject(cameras[n],p[5:13]) for n,p in zip(names,points)]
    if any(np.any(r[:,2]<=0) for r in rays_eye):return float('inf')
    rays=[r@cameras[n].camera_to_world_xf[:3,:3].T for n,r in zip(names,rays_eye)]
    rays=[r/np.linalg.norm(r,axis=1)[:,None] for r in rays];baseline=origins[1]-origins[0];errors=[]
    for v,n in enumerate(names):
        normal=np.cross(baseline,rays[1-v]);length=np.linalg.norm(normal,axis=1)
        if np.any(length<1e-6):return float('inf')
        normal/=length[:,None];planar=rays[v]-normal*np.sum(normal*rays[v],axis=1)[:,None]
        eye=cameras[n].world_to_eye(origins[v]+planar*1000)
        if np.any(eye[:,2]<=1e-6):return float('inf')
        projected=cameras[n].eye_to_window(eye);errors.append(np.linalg.norm(projected-points[v][5:13],axis=1))
    return float(np.sqrt(np.mean((errors[0]**2+errors[1]**2)/2))/np.hypot(3,3))


def replay(folder,session):
    cameras=load_calibration(session/'frame/snapshot/device/persist');observed=read_audit(folder/'1-candidate/views.txt');rows=[];errors=[];stamp=None
    for line in (folder/'1-candidate/rig.txt').read_text().splitlines():
        fields=line.split()
        if fields[0]=='frame':stamp=int(fields[1])
        if fields[0]!='correspondence':continue
        assert len(fields)==8
        source,choice=int(fields[1]),int(fields[2]);actual=np.array(fields[3:6],float);costs=np.array(fields[6:],float)
        views=[next(v for v in observed[stamp]['views'] if v['id']==source and v['camera']==name) for name in ('slam_left','slam_right')]
        points=[v['points'].copy() for v in views];original=disagreement(cameras,points);expected=[original,np.inf,np.inf]
        if np.isfinite(original) and original>3:
            for v in (0,1):
                alternative=[p.copy() for p in points];alternative[v][5:9],alternative[v][9:13]=points[v][9:13],points[v][5:9]
                expected[1+v]=disagreement(cameras,alternative)
        expected=np.array(expected);assert np.array_equal(np.isfinite(expected),np.isfinite(actual))
        finite=np.isfinite(expected);error=float(np.max(abs(expected[finite]-actual[finite]))) if np.any(finite) else 0;errors.append(error)
        assert error<1e-4 and choice in (-1,0,1)
        if choice>=0:assert actual[0]>3 and actual[choice+1]<=3 and np.isfinite(costs[choice])
        rows.append(dict(raw_ns=stamp,source=source,choice=choice,original=actual[0],alternative=actual[1:].tolist()))
    assert len(rows)==3057
    result=dict(rows=len(rows),choices=[sum(r['choice']==v for r in rows) for v in (-1,0,1)],ambiguous_originals=int(sum(np.isfinite(r['original']) and r['original']>3 for r in rows)),max_independent_geometry_difference=max(errors),trace_sha256=sha(folder/'1-candidate/rig.txt'))
    (folder/'correspondence-audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


def motion(folder,heads_path):
    from scipy.spatial.transform import Rotation
    records=json.loads((folder/'audit/records.json').read_text());steps=json.loads((folder/'audit/steps.json').read_text())
    heads={int(x[0]):x for x in np.loadtxt(heads_path,skiprows=1)};result_steps=[]
    for row in steps:
        a,b=records[row['previous']],records[row['index']];world=[]
        for record in (a,b):
            h=heads[record['raw_ns']];assert h[2]
            rot=Rotation.from_quat(h[3:7][[1,2,3,0]])
            world.append(rot.apply(np.array(record['points']).reshape(-1,3)).reshape(2,21,3)+h[7:10])
        movement=np.linalg.norm(world[1]-world[0],axis=2);angles=abs(np.array(b['poses'])[:,7:]-np.array(a['poses'])[:,7:])
        result_steps.append(dict(**row,max_joint_mm=movement.max(axis=1).tolist(),rms_joint_mm=np.sqrt(np.mean(movement**2,axis=1)).tolist(),max_angle_rad=angles.max(axis=1).tolist()))
    result={}
    for key in ('max_joint_mm','rms_joint_mm','max_angle_rad'):
        values=np.array([r[key] for r in result_steps])
        result[key]=[dict(p50=float(np.median(values[:,k])),p95=float(np.percentile(values[:,k],95)),maximum=float(values[:,k].max())) for k in (0,1)]
    result['angle_steps_over_half_radian']=[sum(r['max_angle_rad'][k]>.5 for r in result_steps) for k in (0,1)]
    result['limitations']=['Motion includes physical movement and error; not independent tracking accuracy or jitter alone.','Comparisons stop at missing source, invalid head, missing fits or gaps over100ms, using the parent audit step list.']
    for name,value in (('motion-audit.json',result),('motion-steps.json',result_steps)):
        path=folder/name
        if path.exists():assert json.loads(path.read_text())==value
        else:path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('synthetic','profiles','evaluator','run','session','upstream'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--heads',type=Path,required=True)
    args=parser.parse_args();sys.path.insert(0,str(args.upstream))
    a=synthetic(args.synthetic,args.profiles,args.evaluator);b=replay(args.run,args.session);motion(args.run,args.heads)
    print(json.dumps(dict(synthetic_cases=a['cases'],synthetic_metric_difference=a['max_reconstructed_metric_difference'],replay=b),indent=2))


if __name__=='__main__':main()
