"""Fit frozen Lite/Full sequences with identical anatomy and source assignments.
Fresh unfiltered stereo triangulation, causal angle initialization, then optional
existing index/middle correspondence hypothesis. No whole-wrist history/rendering.
"""
import argparse,json,sys,subprocess
from pathlib import Path
from collections import defaultdict
import numpy as np
from run_tracker_rig_check import sha
from rig_pixel_data import write_cameras
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'exploratory'))
from frame_data import read_audit,load_calibration
MAP=[5,-1,6,7,0,8,9,10,1,11,12,13,2,14,15,16,3,17,18,19,4]

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('run','trace','session','upstream','profiles','executable','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();sys.path.insert(0,str(a.upstream));report=json.loads((a.run/'result.json').read_text());request=json.loads((a.run/'requests.json').read_text());audit=read_audit(a.trace/'views.txt')
 if sha(a.trace/'views.txt')!=request['views_sha256'] or sha(a.run/'requests.json')!=report['provenance']['requests_sha256']:raise ValueError('Changed inputs')
 if not report['native_geometry_reproduced'] or not report['repeat_exact']:raise ValueError('Unverified model outputs')
 a.output.mkdir(parents=True,exist_ok=False);cameras=load_calibration(a.session/'frame/snapshot/device/persist');write_cameras(cameras,a.output/'cameras.txt')
 records=defaultdict(dict)
 for row in report['records']:
  key=(row['raw_ns'],row['source']);assert row['camera'] not in records[key];records[key][row['camera']]=row
 metadata=[]
 for model in ('lite','full'):
  with (a.output/(model+'-input.txt')).open('w') as out:
   for frame in request['sets']:
    raw=frame['raw_ns'];hands=audit[raw]['hands'];out.write(f'set {raw} {len(hands)}\n')
    for source,hand in hands.items():
     side=int(hand['right']>=.5);views=records[(raw,source)];pixels=np.zeros((21,2,3))
     if model=='lite':metadata.append(dict(raw_ns=raw,source=source,side=side,window=frame['window'],case=next(iter(views.values()))['case'] if views else None))
     for v,cam in enumerate(('slam_left','slam_right')):
      if cam not in views:continue
      measured=np.array(views[cam]['models'][model]['points_raw']);assert measured.shape==(21,2) and np.isfinite(measured).all()
      for m,r in enumerate(MAP):
       if r>=0:pixels[r,v]=[1,*measured[m]]
     out.write(f'hand {source} {side} '+' '.join(format(x,'.17g') for x in pixels.ravel())+'\n')
 commands=[]
 for model in ('lite','full'):
  command=[str(a.executable.resolve()),*[str((a.profiles/n).resolve()) for n in ('generic.fhrig','side0.fhrig','side1.fhrig')],str((a.output/'cameras.txt').resolve()),str((a.output/(model+'-input.txt')).resolve())]
  for repeat in ('first','repeat'):
   target=a.output/(model+'-'+repeat+'.txt');cmd=command+[str(target.resolve())];run=subprocess.run(cmd,capture_output=True,text=True,timeout=120);commands.append(cmd)
   (a.output/(model+'-'+repeat+'.log')).write_text(run.stdout+run.stderr)
   if run.returncode:raise RuntimeError(model+': '+run.stderr)
  assert (a.output/(model+'-first.txt')).read_bytes()==(a.output/(model+'-repeat.txt')).read_bytes()
  lines=(a.output/(model+'-first.txt')).read_text().splitlines();assert len(lines)==len(metadata)
  for line,m in zip(lines,metadata):
   f=line.split();assert len(f)==198 and list(map(int,f[:3]))==[m['raw_ns'],m['source'],m['side']] and np.isfinite(np.array(f[3:],float)).all()
 sources={n:sha(BASE/n) for n in ['probe_hand_scale_rig.cpp','rig_stage.h','causal_rig_fit.h','finger_correspondence.h','fit_hand_rig.h','fit_rig_pixels.h','hand_rig.h','stereo_disagreement.h','run_hand_scale_rig.py']}
 result=dict(scope=__doc__,records=metadata,repeat_exact=True,commands=commands,source_sha256=sources,model_result_sha256=sha(a.run/'result.json'),profile_sha256={n:sha(a.profiles/n) for n in ('generic.fhrig','side0.fhrig','side1.fhrig')},camera_sha256=sha(a.output/'cameras.txt'),executable_sha256=sha(a.executable),output_sha256={m:sha(a.output/(m+'-first.txt')) for m in ('lite','full')},limitations=['Existing detector source IDs and sides fixed, including errors; no new detection or association.','No presence threshold applied: geometry diagnostic only, not visibility.','Unfiltered stereo midpoint replaces tracker filtered3D initializer. Not exact reproduction of live native pipeline.','Existing causal angle stage is per source, reset on missing source/side change/insufficient stereo;100ms angle history bound unchanged.','Correspondence correction is output-only; it does not feed back into baseline causal history.','No recorded whole-wrist history, output smoothing, native timing or live acceptance.'])
 (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(hands=len(metadata),repeat_exact=True)))
if __name__=='__main__':main()
