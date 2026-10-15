"""Stage per-profile Godot geometry checks against independent Torch skinning and C++ landmarks."""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from export_profile_mesh import sha

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('assets','profiles','generic','upstream','evaluator','output'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();root=a.output;root.mkdir(parents=True,exist_ok=False);source=Path(__file__).parent
 for name in ('generic.json','side0.json','side1.json'):shutil.copy2(a.assets/name,root/name)
 shutil.copy2(source.parent/'exploratory/mesh-viewer/ume_rig.gd',root/'base_rig.gd')
 for name in ('profile_rig.gd','test_mesh.gd','render_mesh.gd'):shutil.copy2(source/'mesh'/name,root/name)
 (root/'project.godot').write_text('config_version=5\n[application]\nconfig/name="Native profile mesh offline check"\n[display]\nwindow/size/viewport_width=1280\nwindow/size/viewport_height=900\n[rendering]\nrenderer/rendering_method="gl_compatibility"\n[xr]\nopenxr/enabled=false\n')
 sys.path.insert(0,str(source.parent/'exploratory'));sys.path.insert(0,str(a.upstream.resolve()))
 import torch
 from lib.common.hand_skinning import _skin_points
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
 rng=np.random.default_rng(501056);cases=[];meshes=[];landmarks=[];cpp_max=0.
 for side in (0,1):
  for variant in ('generic','side'+str(side)):
   asset=json.loads((root/(variant+'.json')).read_text());limits=np.array(asset['limits']);angles=[np.clip(np.zeros(20),limits[:,0],limits[:,1])]
   fist=angles[0].copy()
   for f in range(1,5):fist[4*f+1:4*f+4]=[1.2,1.2,.7]
   fist=np.clip(fist,limits[:,0],limits[:,1]);point=fist.copy();point[4:8]=0;angles += [fist,np.clip(point,limits[:,0],limits[:,1])]
   angles += list(rng.uniform(limits[:,0],limits[:,1],(16,20)))
   poses=[];expected_lm=[]
   for index,angle in enumerate(angles):
    wrist=np.eye(4);rotation=Rotation.random(random_state=rng);q=rotation.as_quat();wrist[:3,:3]=rotation.as_matrix();wrist[:3,3]=rng.uniform(-.3,.3,3);wrist[2,3]-=.8
    poses.append([side,q[3],*q[:3],*(wrist[:3,3]*1000),*angle]);mirror=wrist.copy()
    if side:mirror[:,0]*=-1
    args=[torch.tensor(asset['rest_m']),torch.tensor(asset['axes']),None,torch.tensor(angle),None,torch.tensor(mirror)]
    def skin(points,weights):
     args[2]=torch.tensor(weights).unsqueeze(0);args[4]=torch.tensor(points)
     with torch.inference_mode():return _skin_points(*args).numpy()
    mesh=skin(asset['vertices_m'],asset['weights']);lm=skin(asset['landmarks_m'],asset['landmark_weights']);meshes.append(mesh);landmarks.append(lm);expected_lm.append(lm)
    cases.append(dict(side=side,asset=variant+'.json',angles=angle.tolist(),wrist=wrist.tolist(),kind=('open','fist','point')[index] if index<3 else 'random_bounded'))
   prefix=root/(f'{side}-{variant}');inp=prefix.with_suffix('.txt');out=prefix.with_suffix('.cpp.txt');np.savetxt(inp,poses,fmt='%.17g')
   profile=a.generic if variant=='generic' else a.profiles/(variant+'.fhrig')
   subprocess.run([str(a.evaluator.resolve()),str(profile.resolve()),str(inp.resolve()),str(out.resolve())],check=True,capture_output=True,text=True,timeout=20)
   cpp=np.loadtxt(out)[:,1:].reshape(-1,21,3)*.001;delta=float(np.max(np.linalg.norm(cpp-np.array(expected_lm),axis=2)));cpp_max=max(cpp_max,delta)
 if cpp_max>1e-8:raise ValueError('Torch landmarks disagree with native profile FK')
 (root/'fixtures.json').write_text(json.dumps(dict(cases=cases),separators=(',',':'))+'\n');np.savez_compressed(root/'expected.npz',vertices=np.array(meshes),landmarks=np.array(landmarks))
 meta=dict(scope=__doc__,cases=len(cases),seed=501056,max_cpp_torch_landmark_difference_mm=cpp_max*1000,source_sha256=sha(Path(__file__)),evaluator_sha256=sha(a.evaluator),upstream_skinning_sha256=sha(a.upstream/'lib/common/hand_skinning.py'),sha256={p.name:sha(p) for p in root.iterdir() if p.is_file()},limitations=['Random bounded angles are algebra checks, not physically valid gesture/collision labels.','Torch CPU only; Godot checks/render and native execution are separate.','Private generic mesh retains upstream license; assets are not committed.'])
 (root/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(dict(cases=len(cases),cpp_torch_max_mm=cpp_max*1000)))
if __name__=='__main__':main()
