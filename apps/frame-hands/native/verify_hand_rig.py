"""Verify portable rig against pinned CPU Torch skinning and malformed-input controls."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'exploratory'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('upstream','profile','executable','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.upstream));a.output.mkdir(parents=True,exist_ok=False)
    torch.set_default_dtype(torch.float64)
    from lib.common.hand import HandModel
    from lib.common.hand_skinning import skin_landmarks
    d=json.loads((a.upstream/'dataset/generic_hand_model.json').read_text())
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if sha(a.upstream/'dataset/generic_hand_model.json')!='96649ddf7d036be3357dadd7c13db325a18638c6713583fbbbdd71446ce07564':raise ValueError('Changed anatomy')
    hand=HandModel(**{k:torch.from_numpy(np.asarray(v)) if isinstance(v,list) else v for k,v in d.items()})
    hand=hand._replace(**{k:v.double() for k,v in hand._asdict().items() if isinstance(v,torch.Tensor) and v.is_floating_point()})
    rng=np.random.default_rng(50107);limits=np.array(d['joint_limits'])[:20]
    angles=rng.uniform(limits[:,0],limits[:,1],size=(512,20));angles[:2]=0;angles[2]=limits[:,0];angles[3]=limits[:,1]
    q=Rotation.random(512,random_state=rng).as_quat();q[:4]=[0,0,0,1];translation=rng.uniform(-600,600,size=(512,3));translation[:4]=0
    sides=np.arange(512)%2;wrist=np.tile(np.eye(4),(512,1,1));wrist[:,:3,:3]=Rotation.from_quat(q).as_matrix();wrist[:,:3,3]=translation
    wrist[sides==1,:,0]*=-1
    batch=hand._replace(**{k:v.unsqueeze(0).expand(512,*v.shape) for k,v in hand._asdict().items() if isinstance(v,torch.Tensor)})
    with torch.inference_mode():expected=skin_landmarks(batch,torch.tensor(np.pad(angles,((0,0),(0,2)))),torch.tensor(wrist)).numpy()
    source=a.output/'poses.txt'
    with source.open('x') as f:
        for i in range(512):f.write(' '.join(map(str,[sides[i],q[i,3],*q[i,:3],*translation[i],*angles[i]]))+'\n')
    result=subprocess.run([str(a.executable.resolve()),str(a.profile.resolve()),str(source.resolve()),str((a.output/'native.txt').resolve())],capture_output=True,text=True)
    if result.returncode:raise RuntimeError(result.stderr)
    actual=np.loadtxt(a.output/'native.txt')
    if actual.shape!=(512,64) or not np.array_equal(actual[:,0],sides):raise ValueError('Incomplete pose output')
    actual=actual[:,1:].reshape(512,21,3)
    error=float(np.max(np.abs(actual-expected)))
    if not np.isfinite(actual).all() or error>1e-6:raise ValueError(f'Forward kinematics mismatch {error}mm')
    # Rest lengths are invariant across wrist poses, sides and all admissible angles.
    segments=[(6,7),(7,0),(8,9),(9,10),(10,1),(11,12),(12,13),(13,2),(14,15),(15,16),(16,3),(17,18),(18,19),(19,4)]
    lengths=np.stack([np.linalg.norm(actual[:,x]-actual[:,y],axis=1) for x,y in segments],axis=1)
    variation=float(np.max(np.ptp(lengths,axis=0)))
    if variation>1e-4:raise ValueError(f'Articulated segment length varies {variation}mm')
    # Identity, pure translation and reflection checked independently of Torch.
    np.testing.assert_allclose(actual[0],np.array(d['landmark_rest_positions']),atol=1e-10)
    mirror=actual[0].copy();mirror[:,0]*=-1;np.testing.assert_allclose(actual[1],mirror,atol=1e-10)
    tests=[]
    def rejects(name,profile,pose):
        pf=a.output/(name+'.rig');pf.write_text(profile);inp=a.output/(name+'.input');inp.write_text(pose)
        proc=subprocess.run([str(a.executable.resolve()),str(pf.resolve()),str(inp.resolve()),str((a.output/(name+'.out')).resolve())],capture_output=True,text=True)
        if proc.returncode==0:raise ValueError('Malformed case accepted: '+name)
        tests.append(dict(name=name,exit_code=proc.returncode,message=proc.stderr.strip()))
    text=a.profile.read_text();line=source.read_text().splitlines()[0]+'\n';fields=line.split()
    rejects('wrong_units',text.replace(' mm ',' meters ',1),line)
    rejects('wrong_version',text.replace('FHRIG 1','FHRIG 2',1),line)
    rejects('truncated', '\n'.join(text.splitlines()[:-1]),line)
    rejects('trailing_data',text+'unexpected\n',line)
    rows=text.splitlines();bad=rows[1].split();bad[:3]=['0','0','0'];rows[1]=' '.join(bad);rejects('zero_axis','\n'.join(rows),line)
    rows=text.splitlines();bad=rows[21].split();bad[3:]=['0']*17;rows[21]=' '.join(bad);rejects('zero_weights','\n'.join(rows),line)
    rejects('false_calibration',text.replace('generic -1','calibrated 0',1),line)
    wrong=fields.copy();wrong[0]='2';rejects('wrong_side',text,' '.join(wrong)+'\n')
    wrong=fields.copy();wrong[1]='2';rejects('nonunit_wrist',text,' '.join(wrong)+'\n')
    wrong=fields.copy();wrong[8]=str(limits[0,1]+.1);rejects('outside_limits',text,' '.join(wrong)+'\n')
    wrong=fields.copy();wrong[-1]='nan';rejects('nonfinite_pose',text,' '.join(wrong)+'\n')
    # A synthetic calibrated-left header must refuse right-side use.
    personal=text.replace('generic -1','calibrated 0',1).replace(' - 20 21 17',' '+'a'*64+' 20 21 17',1)
    wrong=fields.copy();wrong[0]='1';rejects('wrong_personal_side',personal,' '.join(wrong)+'\n')
    report=dict(scope=__doc__,poses=512,seed=50107,maximum_native_reference_coordinate_difference_mm=error,maximum_segment_length_variation_mm=variation,compared_segments=len(segments),invalid_cases=tests,profile_status='Generic only; no user scan or calibration performed.',limitations=['Forward-kinematic correctness is not image tracking accuracy.','21model landmarks are not identical to MediaPipe21; thumb CMC mapping is intentionally absent.','No skin mesh rendering, inverse fit, source integration, Frame timing or wearer validation.','Generic joint limits do not establish individual physical ranges.'],provenance=dict(model_sha256=sha(a.upstream/'dataset/generic_hand_model.json'),profile_sha256=sha(a.profile),executable_sha256=sha(a.executable),sources={n:sha(BASE/n) for n in ('hand_rig.h','evaluate_hand_rig.cpp','export_hand_rig.py','verify_hand_rig.py')},torch_version=torch.__version__,reference_device='CPU float64'))
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(a.output/'reference.npz',expected=expected,angles=angles,wrist=wrist)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
