"""Falsify a small conditional finger prior on sequence-grouped public labels.

This uses idealized masked joint angles, not detector visibility or independent
physical annotations. No Frame capture, private holdout, or runtime integration.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

MODEL_SHA='96649ddf7d036be3357dadd7c13db325a18638c6713583fbbbdd71446ce07564'
DATA_COMMIT='355df534a1c761ac416427925485b311f22ce50e'
SHRINKAGE=.10
OBSERVATION_SD=.10
FINGERS=('thumb','index','middle','ring','pinky')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def conditional(mean,covariance,observed,known,noise_variance=OBSERVATION_SD**2):
    observed=np.asarray(observed,dtype=float)
    known=np.asarray(known,dtype=bool)
    if observed.shape!=(20,) or known.shape!=(20,) or not np.isfinite(observed[known]).all() or noise_variance<0 or not np.isfinite(noise_variance):
        raise ValueError('Invalid known angles/mask/noise')
    hidden=~known
    result=mean.copy();uncertainty=covariance.copy()
    result[known]=observed[known]
    if np.any(known):
        weights=np.linalg.solve(covariance[np.ix_(known,known)]+np.eye(known.sum())*noise_variance,covariance[np.ix_(known,hidden)]).T
        result[hidden]+=weights@(observed[known]-mean[known])
        uncertainty[np.ix_(hidden,hidden)]-=weights@covariance[np.ix_(known,hidden)]
        uncertainty[known,:]=0;uncertainty[:,known]=0
    return result,uncertainty


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('public','upstream','msgpack-path','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(a.msgpack_path.resolve()));import msgpack
    model_path=a.upstream/'dataset/generic_hand_model.json'
    if sha(model_path)!=MODEL_SHA:raise ValueError('Unpinned rig')
    model=json.loads(model_path.read_text());limits=np.array(model['joint_limits'])[:20]
    manifest=json.loads((a.public/'manifest.json').read_text())
    if manifest['commit']!=DATA_COMMIT or not manifest['training_only']:raise ValueError('Unpinned public training subset')
    records=[];geometry_error=0.
    for item in manifest['records']:
        directory=a.public/item['directory'];archive=directory/'training.npz'
        if sha(archive)!=item['archive_sha256']:raise ValueError('Changed prepared archive')
        source=a.public/'records'/directory.name
        source_manifest=json.loads((source/'manifest.json').read_text())
        if sha(source/'labels.record')!=source_manifest['files']['labels']['range_sha256']:raise ValueError('Changed raw labels')
        labels=msgpack.unpackb((source/'labels.record').read_bytes(),raw=False,strict_map_key=True,max_array_len=1000000,max_map_len=128,max_str_len=4096,max_bin_len=8000000)
        with np.load(archive,allow_pickle=False) as data:
            angles=data['generic_joint_angles'].astype(float)
            sides=data['hand_idx'].astype(int)
        if angles.shape!=(item['frames'],22) or not np.isfinite(angles).all() or np.any(angles[:,20:]!=0) or len(set(sides))!=1:raise ValueError('Changed angle convention')
        np.testing.assert_allclose(angles,np.asarray(labels['solved_joint_angles']),rtol=0,atol=1e-7)
        # Source right-hand geometry is mirrored back to the canonical-left rig.
        for key in ('joint_rotation_axes','joint_rest_positions','landmark_rest_positions'):
            value=np.array(labels['generic_hand_model'][key],dtype=float)
            if sides[0]:
                if key=='joint_rotation_axes':value[:,1:]*=-1
                else:value[:,0]*=-1
            error=float(np.max(np.abs(value-np.array(model[key]))));geometry_error=max(geometry_error,error)
            if error>1e-6:raise ValueError('Generic model mismatch')
        # Group matching numeric IDs across interaction categories within each domain.
        pieces=item['sequence'].split('/');group=pieces[1]+'/'+pieces[-2]
        records.append(dict(sequence=item['sequence'],group=group,entry=item['entry'],angles=angles[:,:20],archive_sha256=item['archive_sha256']))
    domains=sorted({r['group'].split('/')[0] for r in records});validation_groups=set()
    for domain in domains:
        keys=sorted({r['group'] for r in records if r['group'].startswith(domain+'/')})
        validation_groups.update(keys[3::4])
    train=[r for r in records if r['group'] not in validation_groups];validation=[r for r in records if r['group'] in validation_groups]
    if not train or not validation or {r['group'] for r in train}&validation_groups:raise ValueError('Invalid split')
    training=np.concatenate([r['angles'] for r in train]);testing=np.concatenate([r['angles'] for r in validation])
    mean=training.mean(0);empirical=np.cov(training,rowvar=False)
    covariance=(1-SHRINKAGE)*empirical+SHRINKAGE*np.diag(np.diag(empirical))+np.eye(20)*1e-6
    np.linalg.cholesky(covariance)
    np.savez_compressed(a.output/'conditional-prior.npz',mean=mean,covariance=covariance,limits=limits)
    # Mask contract: hidden garbage/NaNs must not leak into a conditional estimate.
    known=np.ones(20,bool);known[8:12]=False;dirty=testing[0].copy();dirty[~known]=np.nan
    clean,cov=conditional(mean,covariance,testing[0],known);masked,cov2=conditional(mean,covariance,dirty,known)
    np.testing.assert_array_equal(clean,masked);np.testing.assert_array_equal(cov,cov2)
    np.testing.assert_array_equal(conditional(mean,covariance,testing[0],np.ones(20,bool))[0],testing[0])
    np.testing.assert_array_equal(conditional(mean,covariance,np.full(20,np.nan),np.zeros(20,bool))[0],mean)
    rejected=False
    try:conditional(mean,covariance,np.full(20,np.nan),np.ones(20,bool))
    except ValueError:rejected=True
    if not rejected:raise ValueError('Invalid observed angles accepted')
    # Static conditional prior vs unconditional mean and naive visible-finger copying.
    # A separate hold baseline gets the first frame of each16-frame record before masking.
    rng=np.random.default_rng(501051);cases=[];predictions={};std=[]
    for record_index,record in enumerate(validation):
        for frame,truth in enumerate(record['angles']):
            for finger in range(5):
                hidden=np.arange(4*finger,4*finger+4);known=np.ones(20,bool);known[hidden]=False
                others=[f for f in range(1,5) if f!=finger]
                curl=np.array([truth[4*f+1:4*f+4].mean() for f in range(1,5)])
                exception=bool(finger>0 and ((curl[finger-1]<.3 and np.median(curl[np.arange(4)!=finger-1])>.9) or (curl[finger-1]>.9 and np.median(curl[np.arange(4)!=finger-1])<.3)))
                variants={'mean':mean.copy(),'copy_visible':mean.copy(),'hold_first':mean.copy() if frame==0 else record['angles'][0].copy()}
                if finger>0:
                    variants['copy_visible'][4*finger+1:4*finger+4]=np.mean([truth[4*f+1:4*f+4] for f in others],axis=0)
                ideal,uncertainty=conditional(mean,covariance,truth,known)
                noisy=truth.copy();noisy[known]+=rng.normal(0,OBSERVATION_SD,known.sum())
                variants['conditional']=ideal;variants['conditional_noisy']=conditional(mean,covariance,noisy,known)[0]
                for name,value in variants.items():
                    pose=truth.copy();pose[hidden]=np.clip(value[hidden],limits[hidden,0],limits[hidden,1])
                    predictions.setdefault(name,[]).append(pose)
                cases.append(dict(record=record_index,frame=frame,finger=finger,exception=exception,truth=truth,raw_conditional=ideal[hidden]))
                std.append(np.sqrt(np.maximum(0,np.diag(uncertainty)[hidden])))
    truth=np.array([x['truth'] for x in cases]);finger=np.array([x['finger'] for x in cases]);frame=np.array([x['frame'] for x in cases]);exceptions=np.array([x['exception'] for x in cases]);hidden=4*finger[:,None]+np.arange(4);row=np.arange(len(cases))[:,None];std=np.array(std)
    # Independent upstream CPU skinning measures the consequence on a generic fingertip.
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'exploratory'))
    sys.path.insert(0,str(a.upstream.resolve()));import torch
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    from lib.common.hand import HandModel
    from lib.common.hand_skinning import skin_landmarks
    hand=HandModel(**{k:torch.tensor(v) if isinstance(v,list) else v for k,v in model.items()})
    def tips(poses):
        batch=hand._replace(**{k:v.unsqueeze(0).expand(len(poses),*v.shape) for k,v in hand._asdict().items() if isinstance(v,torch.Tensor)})
        with torch.inference_mode():points=skin_landmarks(batch,torch.tensor(np.pad(poses,((0,0),(0,2)))),torch.eye(4).expand(len(poses),4,4)).numpy()
        return points[np.arange(len(poses)),finger]
    target=tips(truth);errors={};masks={'all':np.ones(len(cases),bool),'history_available':frame>0,'independent_finger_exceptions':exceptions}
    masks.update({name:finger==i for i,name in enumerate(FINGERS)})
    for name,value in predictions.items():
        value=np.array(value);predictions[name]=value;angular=value[row,hidden]-truth[row,hidden];tip_error=np.linalg.norm(tips(value)-target,axis=1)
        errors[name]={}
        for group,mask in masks.items():
            if not mask.any():errors[name][group]=dict(cases=0);continue
            errors[name][group]=dict(cases=int(mask.sum()),angle_rmse_rad=float(np.sqrt(np.mean(angular[mask]**2))),tip_error_mm=dict(p50=float(np.median(tip_error[mask])),p95=float(np.percentile(tip_error[mask],95)),mean=float(tip_error[mask].mean())))
    raw=np.array([x['raw_conditional'] for x in cases]);coverage=np.abs(raw-truth[row,hidden])<=1.96*std
    clipped_truth=truth.copy();clipped_truth[row,hidden]=np.clip(truth[row,hidden],limits[hidden,0],limits[hidden,1]);clip_error=np.linalg.norm(tips(clipped_truth)-target,axis=1)
    all_angles=np.concatenate([r['angles'] for r in records]);violation=np.maximum.reduce([limits[:,0]-all_angles,all_angles-limits[:,1],np.zeros_like(all_angles)])
    report=dict(scope=__doc__,seed=501051,settings=dict(shrinkage=SHRINKAGE,observation_sd_rad=OBSERVATION_SD),split=dict(train_records=len(train),train_frames=len(training),validation_records=len(validation),validation_frames=len(testing),validation_groups=sorted(validation_groups),train_groups=sorted({r['group'] for r in train}),policy='Every fourth sorted domain/numeric-ID group held out; both entries and interaction categories stay together.'),metrics=errors,nominal_95pct_angle_interval_coverage=float(coverage.mean()),mean_conditional_angle_std_rad=float(std.mean()),limits_audit=dict(frames=len(all_angles),outside_over_1e_6=int((violation.max(1)>1e-6).sum()),outside_over_point05=int((violation.max(1)>.05).sum()),max_violation_rad=float(violation.max()),hidden_tip_error_from_limits_only_mm=dict(p50=float(np.median(clip_error)),p95=float(np.percentile(clip_error,95)),maximum=float(clip_error.max()))),geometry_max_difference=geometry_error,controls=dict(hidden_garbage_invariant=True,all_observed_unchanged=True,all_unknown_returns_prior=True,invalid_observed_rejected=True),provenance=dict(public_manifest_sha256=sha(a.public/'manifest.json'),model_sha256=sha(model_path),prior_sha256=sha(a.output/'conditional-prior.npz'),source_sha256=sha(Path(__file__)),upstream_skinning_sha256=sha(a.upstream/'lib/common/hand_skinning.py'),rodrigues_adapter_sha256=sha(Path(__file__).resolve().parent.parent/'exploratory/rotations.py'),archives=[dict(sequence=r['sequence'],entry=r['entry'],sha256=r['archive_sha256']) for r in records]),limitations=['Generic-skeleton solved labels and perfect finger masks; not camera occlusion detection or physical ground truth.','Visible angles are ideal or independently corrupted by0.10rad; real detector errors are correlated.','Hold baseline sees each record first frame; compare only history_available group, not cold-start cases.','Current native limits clamp inferred angles only; observed angles are never changed by the prior.','Internal grouped holdout from a previously used public training subset, not a new external benchmark or guaranteed subject-independent split.','One Gaussian can average incompatible poses; uncertainty is not calibrated confidence.','Private recordings/latest accepted holdout unused; no runtime export, Frame execution, integration or wearer trial.'])
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(a.output/'validation.npz',truth=truth,finger=finger,frame=frame,exception=exceptions,std=std,**predictions)
    (a.output/'cases.json').write_text(json.dumps([{k:v for k,v in x.items() if k not in ('truth','raw_conditional')} for x in cases])+'\n')
    print(json.dumps({k:report[k] for k in ('split','metrics','nominal_95pct_angle_interval_coverage','limits_audit')},indent=2))

if __name__=='__main__':main()
