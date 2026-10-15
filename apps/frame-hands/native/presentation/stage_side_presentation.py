"""Stage paired full-path presentation of an audited native side-policy experiment."""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_tracker_rig_check import sha

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('run','first-heads','second-heads','output'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();src=Path(__file__).resolve().parent;a.output.mkdir(parents=True,exist_ok=False)
 shutil.copy2(src.parents[1]/'exploratory/mesh-viewer/hand_motion.gd',a.output/'baseline_motion.gd')
 for n in ('hand_motion.gd','replay_side_motion.gd'):shutil.copy2(src/n,a.output/n)
 (a.output/'project.godot').write_text('config_version=5\n[application]\nconfig/name="Offline side-policy presentation"\n[rendering]\nrenderer/rendering_method="gl_compatibility"\n')
 fixtures={}
 for label,heads in [('first',a.first_heads),('second',a.second_heads)]:
  inputs=[]
  for variant in ('baseline','candidate'):
   output=a.output/(label+'-'+variant+'.json')
   result=subprocess.run([sys.executable,str(src/'prepare_tracker_presentation.py'),'--run',str(a.run/label),'--heads',str(heads),'--variant',variant,'--output',str(output)],capture_output=True,text=True,check=True);print(label,variant,result.stdout.strip())
   inputs.append(json.loads(output.read_text()))
  events=[]
  for old,new in zip(inputs[0]['events'],inputs[1]['events']):
   assert all(old[k]==new[k] for k in ('raw_ns','now','capture','stamp','valid'))
   event=old.copy();event['hands_by_variant']=[old['hands'],new['hands']];events.append(event)
  assert len(events)==len(inputs[0]['events'])==len(inputs[1]['events'])==1800
  combined=dict(name=label+'-provisional-side',scope='Audited full-neural baseline/candidate poses; identical original native motion; assumed50ms delivery/90Hz ticks. No actual native latency or wearer validation.',events=events,provenance=dict(baseline=inputs[0]['provenance'],candidate=inputs[1]['provenance']))
  output=a.output/(label+'.json');output.write_text(json.dumps(combined,separators=(',',':'))+'\n');fixtures[label]=sha(output)
 (a.output/'stage-manifest.json').write_text(json.dumps(dict(source_sha256={n:sha(a.output/n) for n in ('baseline_motion.gd','hand_motion.gd','replay_side_motion.gd','project.godot')},fixtures_sha256=fixtures,runner_sha256=sha(Path(__file__))),indent=2)+'\n')
if __name__=='__main__':main()
