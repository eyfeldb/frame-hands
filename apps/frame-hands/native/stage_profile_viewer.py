"""Stage a separate native diagnostic viewer with fixed per-side profiles and accepted motion."""
import argparse,json,shutil
from pathlib import Path
from export_profile_mesh import sha
from stage_tracker_rig import replace_once

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('assets','output'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);base=Path(__file__).parent;viewer=base.parent/'exploratory/mesh-viewer';manifest=json.loads((a.assets/'manifest.json').read_text(encoding="utf-8"))
 for n in ('project.godot','main.tscn','openxr_action_map.tres'):shutil.copy2(viewer/n,a.output/n)
 shutil.copy2(viewer/'ume_rig.gd',a.output/'base_rig.gd');shutil.copy2(viewer/'hand_motion.gd',a.output/'baseline_motion.gd');shutil.copy2(base/'presentation/hand_motion.gd',a.output/'hand_motion.gd')
 for n in ('profile_rig.gd','profile_assets.gd','test_assets.gd'):shutil.copy2(base/'mesh'/n,a.output/n)
 index=dict(version=2,profiles=[])
 for side in (0,1):
  n=f'side{side}.json';entry=manifest['sides'][str(side)]
  if sha(a.assets/n)!=entry['mesh_sha256']:raise ValueError('Changed mesh')
  shutil.copy2(a.assets/n,a.output/n);index['profiles'].append(dict(side=side,file=n,sha256=sha(a.output/n),profile_sha256=entry['profile_sha256']))
 (a.output/'profile-rigs.json').write_text(json.dumps(index,indent=2)+'\n')
 s=(viewer/'viewer.gd').read_text(encoding="utf-8");s=replace_once(s,'const Rig = preload("res://ume_rig.gd")','const Rig = preload("res://profile_rig.gd")\nconst Assets = preload("res://profile_assets.gd")')
 s=replace_once(s,'var rig_path := OS.get_environment("FRAME_UME_RIG")','var rig_path := "res://profile-rigs.json"')
 s=replace_once(s,'        var rig = JSON.parse_string(FileAccess.get_file_as_string(rig_path))\n        assert(rig is Dictionary and rig.version == 1)','        var profiles = Assets.load_verified(rig_path)\n        if profiles.size() != 2:\n            push_error("Native profile asset verification failed")\n            get_tree().quit(4)\n            return')
 s=replace_once(s,'            hand.configure(rig,side,materials[side])','            if not hand.configure(profiles[side],side,materials[side]):\n                push_error("Native profile side mismatch")\n                get_tree().quit(5)\n                return')
 s=s.replace('UME TRACK \u2022 MESH LAB','FRAME-NATIVE HANDS \u2022 EXPERIMENTAL').replace('UmeTrack mesh viewer ready; one solid layer; cyan left / orange right','Frame-native profile viewer ready; cyan left / orange right')
 (a.output/'viewer.gd').write_text(s,encoding="utf-8")
 p=a.output/'project.godot';s=p.read_text(encoding="utf-8").replace('Frame Hands \u2014 UmeTrack Mesh Lab','Frame Native Hands Lab').replace('FrameHandsUmeMeshLab','FrameHandsNativeProfileLab');p.write_text(s,encoding="utf-8")

 meta=dict(scope=__doc__,assets_manifest_sha256=sha(a.assets/'manifest.json'),source_sha256=sha(Path(__file__)),sha256={p.name:sha(p) for p in a.output.iterdir() if p.is_file()},limitations=['Staging only, no XR scene launched.','Pose service bridge still needs a supervised native launcher.','Side profiles and mesh are experimental, not wearer accepted or collision constrained.'])
 (a.output/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(dict(manifest_sha256=sha(a.output/'manifest.json'),files=len(meta['sha256']))))
if __name__=='__main__':main()
