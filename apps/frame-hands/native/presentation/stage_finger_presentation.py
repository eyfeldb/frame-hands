"""Stage an isolated finger-continuity diagnostic; no viewer installation or XR."""
import argparse,hashlib,json,shutil
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();root=a.output;root.mkdir(parents=True,exist_ok=False);src=Path(__file__).parent
 for name in ('baseline_motion.gd','observed.json','assumed-30hz.json','native-rig-motion.json','project.godot'):shutil.copy2(a.fixtures/name,root/name)
 for source,target in [('hand_motion.gd','correction_motion.gd'),('finger_motion.gd','hand_motion.gd'),('test_motion.gd','test_motion.gd'),('test_finger_motion.gd','test_finger_motion.gd')]:shutil.copy2(src/source,root/target)
 s=(src/'replay_motion.gd').read_text()
 replacements={
  'const Baseline = preload("res://baseline_motion.gd")':'const Baseline = preload("res://correction_motion.gd")',
  '"correcting":sample.get("correcting",false)':'"correcting":sample.get("correcting",false),"inferred_fingers":sample.get("inferred_fingers",[]),"finger_hold_ages":sample.get("finger_hold_ages",{}),"finger_correcting":sample.get("finger_correcting",[])',
  '"paired_replay_wall_ms":':'"finger_holds":states[1][0].finger_holds+states[1][1].finger_holds,"finger_confirmations":states[1][0].finger_confirmations+states[1][1].finger_confirmations,"finger_forced_corrections":states[1][0].finger_forced_corrections+states[1][1].finger_forced_corrections,"paired_replay_wall_ms":'}
 for before,after in replacements.items():
  if s.count(before)!=1:raise ValueError('Changed replay contract')
  s=s.replace(before,after)
 (root/'replay_motion.gd').write_text(s,newline='\n')
 report=dict(scope=__doc__,source_sha256=sha(Path(__file__)),sha256={p.name:sha(p) for p in root.iterdir() if p.is_file()},limitations=['Private offline staging only. No camera, model, input or installed-renderer changes.','Fixture manifest alone does not prove tracking or wearer quality.'])
 (root/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(dict(manifest_sha256=sha(root/'manifest.json'))))


if __name__=='__main__':main()
