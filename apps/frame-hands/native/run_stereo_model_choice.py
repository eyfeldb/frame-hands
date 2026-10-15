"""Run a pinned offline stereo-model fallback and unchanged constrained fitter."""
import argparse,json,subprocess
from pathlib import Path
from run_tracker_rig_check import sha


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('models','rig','profiles','selector','fitter','output'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--independent-history',action='store_true')
 a=p.parse_args();rig=json.loads((a.rig/'result.json').read_text());models=json.loads((a.models/'result.json').read_text())
 if sha(a.models/'result.json')!=rig['model_result_sha256'] or sha(a.rig/'cameras.txt')!=rig['camera_sha256']:raise ValueError('Changed input provenance')
 for n,h in rig['profile_sha256'].items():
  if sha(a.profiles/n)!=h:raise ValueError('Changed anatomy')
 a.output.mkdir(parents=True,exist_ok=False);lookup={(r['raw_ns'],r['source'],r['camera']):r for r in models['records']}
 with (a.output/'presence.txt').open('w') as f:
  for row in rig['records']:
   values=[]
   for m in ('full','lite'):
    for cam in ('slam_left','slam_right'):
     entry=lookup.get((row['raw_ns'],row['source'],cam));values.append(entry['models'][m]['presence'] if entry else 0.)
   f.write(f"{row['raw_ns']} {row['source']} "+' '.join(format(v,'.17g') for v in values)+'\n')
 commands=[]
 for repeat in ('first','repeat'):
  selected=a.output/(repeat+'-input.txt');decisions=a.output/(repeat+'-decisions.txt');fit=a.output/(repeat+'-fitted.txt')
  cmd=[a.selector,a.rig/'cameras.txt',a.rig/'full-input.txt',a.rig/'lite-input.txt',a.output/'presence.txt',selected,decisions]
  cmd=list(map(lambda p:str(p.resolve()),cmd));r=subprocess.run(cmd,capture_output=True,text=True,timeout=60);commands.append(cmd);(a.output/(repeat+'-selection.log')).write_text(r.stdout+r.stderr)
  if r.returncode:raise RuntimeError(r.stderr)
  cmd=[a.fitter,*[a.profiles/n for n in ('generic.fhrig','side0.fhrig','side1.fhrig')],a.rig/'cameras.txt',selected,fit];cmd=[str(p.resolve()) for p in cmd]
  if a.independent_history:cmd += [str((a.rig/'full-input.txt').resolve()),str(decisions.resolve()),str((a.output/(repeat+'-baseline.txt')).resolve())]
  r=subprocess.run(cmd,capture_output=True,text=True,timeout=120);commands.append(cmd);(a.output/(repeat+'-fit.log')).write_text(r.stdout+r.stderr)
  if r.returncode:raise RuntimeError(r.stderr)
 for name in ('input','decisions','fitted'):
  assert (a.output/('first-'+name+'.txt')).read_bytes()==(a.output/('repeat-'+name+'.txt')).read_bytes()
 if a.independent_history:
  for repeat in ('first','repeat'):assert (a.output/(repeat+'-baseline.txt')).read_bytes()==(a.rig/'full-first.txt').read_bytes()
 source=Path(__file__).parent;result=dict(scope=__doc__,repeat_exact=True,independent_history=a.independent_history,baseline_exact=a.independent_history,records=rig['records'],commands=commands,rig_result_sha256=sha(a.rig/'result.json'),model_result_sha256=sha(a.models/'result.json'),sources={n:sha(source/n) for n in ['stereo_model_choice.h','probe_stereo_model_choice.cpp','run_stereo_model_choice.py','probe_hand_scale_rig.cpp','rig_stage.h','causal_rig_fit.h']},executables=dict(selector=sha(a.selector),fitter=sha(a.fitter)),outputs={n:sha(a.output/('first-'+n+'.txt')) for n in ('input','decisions','fitted')},limitations=rig['limitations']+['Full is retained when every finger-chain stereo RMS is within3 normalized units; alternatives additionally require presence>=0.8 in both views. These are experimental gates, not calibrated probabilities.','Alternative full-camera observations preserve each model skeleton; no per-joint mixing.','Both models execute on all source crops for this diagnostic. Native extra inference cost is unknown.'])
 (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('Repeated selection/input/fits exact')
if __name__=='__main__':main()
