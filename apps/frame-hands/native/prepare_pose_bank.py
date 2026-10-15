"""Export eight actual in-limit public-training poses as bounded fitter starts."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from audit_native_lengths import sha,stats

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('public','split','profile','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();manifest=json.loads((a.public/'manifest.json').read_text());split=json.loads(a.split.read_text())
 if sha(a.public/'manifest.json')!=split['provenance']['public_manifest_sha256'] or manifest['commit']!='355df534a1c761ac416427925485b311f22ce50e' or not manifest['training_only']:raise ValueError('Unpinned dataset')
 header=a.profile.read_text().splitlines()[0].split();limits=np.loadtxt(a.profile,skiprows=1,max_rows=20)[:,6:8];model=header[6]
 if model!='96649ddf7d036be3357dadd7c13db325a18638c6713583fbbbdd71446ce07564':raise ValueError('Wrong model')
 train_groups=set(split['split']['train_groups']);validation_groups=set(split['split']['validation_groups']);assert not train_groups&validation_groups
 train=[];validation=[];identities=[];archives=[];rejected=0
 for record in manifest['records']:
  archive=a.public/record['directory']/'training.npz'
  if sha(archive)!=record['archive_sha256']:raise ValueError('Changed archive')
  with np.load(archive,allow_pickle=False) as z:angles=z['generic_joint_angles'].astype(float)
  assert angles.shape==(record['frames'],22) and np.isfinite(angles).all() and np.all(angles[:,20:]==0)
  angles=angles[:,:20];pieces=record['sequence'].split('/');group=pieces[1]+'/'+pieces[-2]
  if group in validation_groups:validation.extend(angles);continue
  assert group in train_groups
  archives.append(dict(sequence=record['sequence'],entry=record['entry'],sha256=record['archive_sha256']))
  for frame,row in enumerate(angles):
   if np.any(row<limits[:,0]-1e-6) or np.any(row>limits[:,1]+1e-6):rejected+=1;continue
   train.append(np.clip(row,limits[:,0],limits[:,1]));identities.append(dict(sequence=record['sequence'],group=group,entry=record['entry'],frame=frame,directory=record['directory'],archive_sha256=record['archive_sha256']))
 poses=np.array(train);scale=limits[:,1]-limits[:,0];assert len(poses)>=8 and np.all(scale>0);x=poses/scale
 # Deterministic medoids: every exported row is an observed complete pose.
 pair=np.sum((x[:,None]-x[None,:])**2,axis=2);selected=[int(np.argmin(pair.sum(1)))];nearest=pair[:,selected[0]]
 for _ in range(7):
  j=int(np.argmax(nearest));assert j not in selected;selected.append(j);nearest=np.minimum(nearest,pair[:,j])
 iterations=0
 for _ in range(20):
  assignments=np.argmin(pair[:,selected],axis=1);replacement=[]
  for k,old in enumerate(selected):
   members=np.flatnonzero(assignments==k);replacement.append(int(members[np.argmin(pair[np.ix_(members,members)].sum(1))]) if len(members) else old)
  iterations+=1
  if replacement==selected:break
  selected=replacement
 assert len(set(selected))==8
 bank=poses[selected];a.output.mkdir(parents=True,exist_ok=False);path=a.output/'poses.fhbank'
 with path.open('x',encoding='utf-8',newline='\n') as f:
  f.write(f'FHPOSEBANK 1 8 20 {model} {sha(a.public/"manifest.json")}\n');np.savetxt(f,bank,fmt='%.17g')
 np.savez(a.output/'training-poses.npz',poses=poses,selected=selected)
 validation=np.array(validation);nearest_validation=np.sqrt(np.mean(((validation[:,None]-bank[None])/scale)**2,axis=2)).min(1)
 report=dict(scope=__doc__,poses=8,training_frames_retained=len(poses),training_frames_outside_limits_excluded=rejected,validation_frames=len(validation),validation_used_for_selection=False,selection='Fixed8medoids of normalized20angles; actual in-limit training samples, deterministic greedy coverage initialization and maximum20updates. No camera/private-pose selection.',iterations=iterations,exemplars=[identities[i] for i in selected],validation_nearest_normalized_angle_rmse=stats(nearest_validation),limitations=['Small previously collected public training subset, internally grouped split, not an external accuracy benchmark.','Majority of public solved labels exceed generic limits; excluded from selection rather than materially clamped. This narrows pose coverage.','Observed training pose is not guaranteed collision-free on another subject profile. Bank supplies starts, not forced outputs or confidence.','Eight exemplars and two fitted candidates are a single bounded experiment, not a tuned optimum.'],input_sha256=dict(public_manifest=sha(a.public/'manifest.json'),split=sha(a.split),profile=sha(a.profile)),model_sha256=model,bank_sha256=sha(path),source_sha256=sha(Path(__file__)),training_archives=archives)
 (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k not in ('training_archives','exemplars')},indent=2))
if __name__=='__main__':main()
