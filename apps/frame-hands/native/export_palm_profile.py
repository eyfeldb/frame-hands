"""Export unaccepted, side-specific palm-shape candidates without changing finger lengths."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def shape_profile(text,shape,side,calibration_sha):
 s=np.asarray(shape,float)
 if s.shape!=(2,) or not np.isfinite(s).all() or np.any(s<.65) or np.any(s>1.3) or side not in (0,1) or len(calibration_sha)!=64:raise ValueError('Invalid experimental shape provenance')
 rows=text.splitlines();header=rows[0].split()
 if header[:5]!=['FHRIG','1','mm','left','generic'] or header[-3:]!=['20','21','17']:raise ValueError('Expected original generic profile')
 joints=np.array([list(map(float,r.split())) for r in rows[1:21]]);landmarks=np.array([list(map(float,r.split())) for r in rows[21:]])
 if joints.shape!=(20,8) or landmarks.shape!=(21,20):raise ValueError('Invalid source rig dimensions')
 groups=((0,6,7),(1,8,9,10),(2,11,12,13),(3,14,15,16),(4,17,18,19))
 for finger,group in enumerate(groups):
  base=joints[4*finger,3:6];delta=base*np.array([s[0]-1,0,s[1]-1]);joints[4*finger:4*finger+4,3:6]+=delta;landmarks[list(group),:3]+=delta
 landmarks[20,0]*=s[0];landmarks[20,2]*=s[1] # Unobserved palm-center approximation, not calibrated mesh deformation.
 header[4]='experimental';header[5]=str(side);header[7]=calibration_sha
 return ' '.join(header)+'\n'+'\n'.join(' '.join(format(v,'.17g') for v in row) for row in list(joints)+list(landmarks))+'\n'

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('profile','calibration','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);meta=json.loads(a.calibration.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 if sha(a.profile)!=meta['provenance']['profile_sha256'] or not meta.get('visual_review'):raise ValueError('Original profile and explicit visual review required')
 result={}
 for side,data in meta['results'].items():
  candidate=data.get('candidate')
  if candidate is None or not candidate['success'] or candidate['near_shape_bound']:raise ValueError('Unresolved/bounded calibration')
  target=a.output/f'side{side}.fhrig';target.write_text(shape_profile(a.profile.read_text(),candidate['shape'],int(side),sha(a.calibration)));result[side]=dict(shape=candidate['shape'],profile_sha256=sha(target))
 (a.output/'manifest.json').write_text(json.dumps(dict(status='Experimental per-side palm profile. No accepted personal calibration or mesh.',calibration_sha256=sha(a.calibration),source_profile_sha256=sha(a.profile),profiles=result,source_sha256=sha(Path(__file__)),limitations=['Finger lengths and axes unchanged; only rigid offsets of each finger chain change.','Thumb chain base shifts with palm dimensions; no independent thumb calibration.','Palm-center position approximate and mesh deformation not implemented.']),indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
