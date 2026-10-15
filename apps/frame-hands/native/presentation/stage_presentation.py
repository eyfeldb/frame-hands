"""Stage the unchanged accepted motion base plus optional native correction presentation."""
import argparse,hashlib,json,shutil
from pathlib import Path

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('fixtures','native-poses','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();source=Path(__file__).resolve().parent;root=a.output;root.mkdir(parents=True,exist_ok=False)
    baseline=source.parents[1]/'exploratory/mesh-viewer/hand_motion.gd'
    files=[(baseline,'baseline_motion.gd'),(a.fixtures/'observed.json','observed.json'),(a.fixtures/'assumed-30hz.json','assumed-30hz.json'),(a.native_poses,'native-rig-motion.json'),(source.parent/'workflow_guard.py','workflow_guard.py')]
    files += [(source/name,name) for name in ('hand_motion.gd','test_motion.gd','replay_motion.gd','run_native_check.py')]
    for original,name in files:shutil.copyfile(original,root/name)
    (root/'project.godot').write_text('config_version=5\n[application]\nconfig/name="Frame Hands offline motion check"\n[rendering]\nrenderer/rendering_method="gl_compatibility"\n')
    metadata=dict(scope=__doc__,baseline_commit='d5ea73b2ce32fcfade111b520a8527c9847d3cb5',correction_seconds=.06,sha256={p.name:sha(p) for p in root.iterdir() if p.is_file()},source_sha256=sha(Path(__file__)))
    (root/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(dict(manifest_sha256=sha(root/'manifest.json')),indent=2))
if __name__=='__main__':main()
