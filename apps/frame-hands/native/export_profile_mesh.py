"""Export fixed experimental palm meshes matching the exact native joint profiles.

Dense surface deformation is approximate: palm-weighted vertices receive the palm
scale; finger-weighted vertices receive the matching chain translation. This is
not a scanned surface, collision solution, or accepted personal calibration.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from export_hand_rig import MODEL_SHA
from export_palm_profile import shape_profile

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def deform(vertices,weights,pivots,shape):
 scale=np.array([shape[0]-1,0,shape[1]-1]);delta=weights[:,:2].sum(1)[:,None]*(vertices*scale)
 for finger in range(5):delta+=weights[:,2+3*finger:5+3*finger].sum(1)[:,None]*(pivots[4*finger]*scale)
 return vertices+delta

def asset(model,profile,vertices,side,profile_sha):
 joints=np.array([list(map(float,row.split())) for row in profile.splitlines()[1:21]])
 landmarks=np.array([list(map(float,row.split())) for row in profile.splitlines()[21:]])
 triangles=np.asarray(model['mesh_triangles'],int);faces=np.cross(vertices[triangles[:,1]]-vertices[triangles[:,0]],vertices[triangles[:,2]]-vertices[triangles[:,0]])
 normals=np.zeros_like(vertices)
 for i in range(3):np.add.at(normals,triangles[:,i],faces)
 lengths=np.linalg.norm(normals,axis=1)
 if np.any(lengths<1e-9):raise ValueError('Degenerate normals')
 normals/=lengths[:,None]
 return dict(version=1,profile_side=side,profile_sha256=profile_sha,source_model_sha256=MODEL_SHA,status='experimental approximate surface, fixed profile pivots',vertices_m=(vertices*.001).tolist(),normals=normals.tolist(),triangles=triangles.reshape(-1).tolist(),weights=model['dense_bone_weights'],axes=joints[:,:3].tolist(),rest_m=(joints[:,3:6]*.001).tolist(),limits=joints[:,6:8].tolist(),landmarks_m=(landmarks[:,:3]*.001).tolist(),landmark_weights=landmarks[:,3:].tolist())

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('model','generic','profiles','output'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();meta=json.loads((a.profiles/'manifest.json').read_text())
 if sha(a.model)!=MODEL_SHA or sha(a.generic)!=meta['source_profile_sha256']:raise ValueError('Changed source geometry')
 model=json.loads(a.model.read_text());v=np.array(model['mesh_vertices']);w=np.array(model['dense_bone_weights']);pivots=np.array(model['joint_rest_positions']);tri=np.asarray(model['mesh_triangles'],int)
 if v.shape!=(788,3) or w.shape!=(788,17) or not np.isfinite(v).all() or not np.isfinite(w).all() or np.any(w<0) or np.max(np.abs(w.sum(1)-1))>1e-6 or np.any((w>0).sum(1)>4):raise ValueError('Changed mesh contract')
 np.testing.assert_array_equal(deform(v,w,pivots,[1,1]),v)
 a.output.mkdir(parents=True,exist_ok=False);generic=a.generic.read_text();basefaces=np.cross(v[tri[:,1]]-v[tri[:,0]],v[tri[:,2]]-v[tri[:,0]]);results={}
 (a.output/'generic.json').write_text(json.dumps(asset(model,generic,v,-1,sha(a.generic)),separators=(',',':'))+'\n')
 for side in (0,1):
  profile=a.profiles/f'side{side}.fhrig';data=meta['profiles'][str(side)];shape=data['shape']
  if sha(profile)!=data['profile_sha256'] or profile.read_text()!=shape_profile(generic,shape,side,meta['calibration_sha256']):raise ValueError('Profile/shape mismatch')
  vertices=deform(v,w,pivots,shape);faces=np.cross(vertices[tri[:,1]]-vertices[tri[:,0]],vertices[tri[:,2]]-vertices[tri[:,0]])
  flipped=int(np.sum(np.sum(basefaces*faces,axis=1)<=0));area_ratio=np.linalg.norm(faces,axis=1)/np.linalg.norm(basefaces,axis=1)
  if flipped or area_ratio.min()<.1:raise ValueError('Folded or collapsed rest surface')
  pure_count=0;pure_error=0.
  for finger in range(5):
   mass=w[:,2+3*finger:5+3*finger].sum(1);pure=mass>1-1e-6;pure_count+=int(pure.sum())
   delta=pivots[4*finger]*np.array([shape[0]-1,0,shape[1]-1]);pure_error=max(pure_error,float(np.max(np.linalg.norm(vertices[pure]-v[pure]-delta,axis=1))))
  if pure_error>1e-4:raise ValueError('Pure finger vertices changed shape')
  target=a.output/f'side{side}.json';target.write_text(json.dumps(asset(model,profile.read_text(),vertices,side,sha(profile)),separators=(',',':'))+'\n')
  results[str(side)]=dict(shape=shape,mesh_sha256=sha(target),profile_sha256=sha(profile),max_rest_displacement_mm=float(np.max(np.linalg.norm(vertices-v,axis=1))),flipped_rest_triangles=flipped,min_rest_triangle_area_ratio=float(area_ratio.min()),pure_finger_vertices=pure_count,max_pure_finger_translation_error_mm=pure_error)
 report=dict(scope=__doc__,sides=results,zero_shape_identity_exact=True,source_sha256=sha(Path(__file__)),model_sha256=sha(a.model),profile_manifest_sha256=sha(a.profiles/'manifest.json'),limitations=['Skinning-weight surface warp is approximate, not personal surface calibration.','No self-intersection or collision correction, especially for tightly flexed poses.','Only fixed assets change; inference, source identity and temporal behavior unchanged.'])
 (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
