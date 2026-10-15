"""Compare complete original-detector rig traces; no inferred ground-truth labels."""
import argparse,json,subprocess,sys
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from audit_native_lengths import sha,stats
BASE=Path(__file__).resolve().parent

def read_trace(path):
 frames=[];frame=None
 for line in path.read_text(encoding='utf-8').splitlines():
  x=line.split()
  if x[0]=='frame':
   if frame is not None:assert len(frame['rows'])==frame['count']
   frame=dict(raw=int(x[1]),count=int(x[2]),rows={});frames.append(frame)
  elif x[0]=='rig':
   assert frame is not None and len(x) in (4,33)
   source,side=int(x[1]),int(x[2]);assert source not in frame['rows'] and side in (0,1)
   values=np.array(x[4:],float);assert np.isfinite(values).all()
   assert (x[3]=='fitted')==(len(values)==29)
   frame['rows'][source]=dict(side=side,status=x[3],values=values)
 assert frame is not None and len(frame['rows'])==frame['count']
 assert all(a['raw']<b['raw'] for a,b in zip(frames,frames[1:]));return frames

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('run','heads','profiles','evaluator','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
 reports=json.loads((a.run/'result.json').read_text(encoding='utf-8'))
 same_objective=reports.get('comparison')=='fixed_rig_vs_whole_pose'
 same_correspondences=reports.get('comparison') not in ('whole_pose_vs_finger_correspondence','finger_correspondence_vs_continuity','correspondence_continuity_vs_renewal')
 assert reports['original_detector_outputs_identical'] and reports['rig_outputs_repeat_exactly'] and reports['environment']['machine']=='x86_64'
 for r in reports['results']:
  for name in ('poses','views','rig'):assert sha(a.run/r['name']/(name+'.txt'))==r[name+'_sha256']
 variants=[read_trace(a.run/name/'rig.txt') for name in ('0-baseline','1-candidate')]
 assert len(variants[0])==len(variants[1])==reports['results'][0]['processed_frames']
 heads={int(x[0]):x for x in np.loadtxt(a.heads,skiprows=1)}
 records=[];steps=[];previous={};counts=Counter()
 for base,cand in zip(*variants):
  assert base['raw']==cand['raw'] and base['rows'].keys()==cand['rows'].keys();raw=base['raw'];now={}
  for source,b in base['rows'].items():
   c=cand['rows'][source];assert (b['side'],b['status'])==(c['side'],c['status']);counts[b['status']]+=1
   if b['status']!='fitted':continue
   rows=np.stack([b['values'],c['values']])
   if same_objective:assert rows[1,0]<=rows[0,0]+1e-8
   record=dict(index=len(records),raw_ns=raw,source=source,side=b['side'],cost=rows[:,0].tolist(),rmse=rows[:,1].tolist(),poses=rows[:,2:].tolist());records.append(record)
   head=heads[raw]
   if not head[2]:continue
   rotation=Rotation.from_quat(head[3:7][[1,2,3,0]])
   wrists=[rotation*Rotation.from_quat(row[2:6][[1,2,3,0]]) for row in rows]
   translations=[rotation.apply(row[6:9])+head[7:10] for row in rows]
   key=(source,b['side']);now[key]=(raw,record['index'],wrists,translations)
   if key in previous:
    stamp,idx,pw,pt=previous[key]
    if 0<raw-stamp<=100000000:
     steps.append(dict(index=record['index'],previous=idx,wrist_degrees=[float((pw[k].inv()*wrists[k]).magnitude()*180/np.pi) for k in (0,1)],root_mm=[float(np.linalg.norm(translations[k]-pt[k])) for k in (0,1)]))
  previous=now
 # Evaluate exact rig geometry using the same portable C++ anatomy, separately per side.
 for side in (0,1):
  selected=[r for r in records if r['side']==side];input_path=a.output/f'side{side}-input.txt';output_path=a.output/f'side{side}-points.txt'
  input_path.write_text(''.join(str(side)+' '+' '.join(format(v,'.17g') for v in pose)+'\n' for r in selected for pose in r['poses']),encoding='utf-8')
  run=subprocess.run([str(a.evaluator.resolve()),str((a.profiles/f'side{side}.fhrig').resolve()),str(input_path.resolve()),str(output_path.resolve())],capture_output=True,text=True,check=True)
  points=np.loadtxt(output_path);assert points.shape==(2*len(selected),64) and np.isfinite(points).all()
  for i,r in enumerate(selected):r['points']=points[2*i:2*i+2,1:].reshape(2,21,3).tolist()
 metric=dict(frames=len(variants[0]),status_counts=dict(counts),fits=len(records),cost_gains_over1=sum(r['cost'][0]-r['cost'][1]>1 for r in records) if same_objective else None,objectives_comparable=same_objective,pixel_correspondences_comparable=same_correspondences,image_rmse_px=[stats([r['rmse'][k] for r in records]) for k in (0,1)] if same_correspondences else None,fit_reported_rmse_px=[stats([r['rmse'][k] for r in records]) for k in (0,1)],world_wrist_steps_degrees=[stats([r['wrist_degrees'][k] for r in steps]) for k in (0,1)],world_wrist_steps_over45=[sum(r['wrist_degrees'][k]>45 for r in steps) for k in (0,1)],world_root_steps_mm=[stats([r['root_mm'][k] for r in steps]) for k in (0,1)])
 (a.output/'records.json').write_text(json.dumps(records)+'\n',encoding='utf-8');(a.output/'steps.json').write_text(json.dumps(steps)+'\n',encoding='utf-8')
 result=dict(comparison=reports.get('comparison'),metrics=metric,source_sha256=sha(Path(__file__)),input_result_sha256=sha(a.run/'result.json'),heads_sha256=sha(a.heads),record_sha256=sha(a.output/'records.json'),steps_sha256=sha(a.output/'steps.json'),limitations=['Within-executable Linux PC CPU comparison, not Frame execution or timing.','Motion counts include true motion and errors; current fitted pixel RMSE is not independent accuracy.','History comparisons stop at every missing/invalid fitted frame, source/side change or missing head.','Actual fresh neural inference and tracker state exercised; no rendering or optical ground truth.'])
 (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(metric,indent=2))
if __name__=='__main__':main()
