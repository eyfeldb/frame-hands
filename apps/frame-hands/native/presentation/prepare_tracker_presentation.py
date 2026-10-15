"""Convert full-neural rig traces to world-space motion fixtures, preserving viewer ambiguity.
50ms delivery and90Hz sampling are assumptions, never native latency measurements.
"""
import argparse,json,sys
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.spatial.transform import Rotation
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
from run_tracker_rig_check import sha,verify_head_evidence
from audit_host_tracker import read_trace

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('run','heads','output'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument("--variant",choices=("baseline","candidate"),default="candidate")
 a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 if '143912Z' in str(a.run)+str(a.heads):raise ValueError('Accepted holdout excluded')
 report=json.loads((a.run/'result.json').read_text());selected=report['results'][0 if a.variant=='baseline' else 1];trace=a.run/selected['name']/'rig.txt'
 assert report['rig_outputs_repeat_exactly']
 if report.get('comparison')=='provisional_side_hysteresis':
  audit=json.loads((a.run/'side-audit.json').read_text())
  assert report['neural_views_exact'] and audit['raw_detector_geometry_exact'] and audit['input_result_sha256']==sha(a.run/'result.json')
  assert audit[a.variant+'_rig_sha256']==sha(trace)
 else:assert report['original_detector_outputs_identical']
 assert sha(trace)==selected['rig_sha256']
 meta=json.loads((a.heads/'manifest.json').read_text());head_evidence=verify_head_evidence(a.heads,meta['recording_sha256'])
 if 'provenance' in report:assert report['provenance']['recording_sha256']==meta['recording_sha256']
 # Integer clocks must not take a float64 roundtrip.
 heads={}
 for line in (a.heads/'heads.txt').read_text().splitlines()[1:]:
  x=line.split();raw,mono,valid=map(int,x[:3]);assert raw not in heads and valid in (0,1)
  values=np.array(x[3:],float);assert len(values)==7 and np.isfinite(values).all()
  heads[raw]=(mono,valid,values)
 for line in trace.read_text().splitlines():
  x=line.split()
  if x[0]=='head_recorded':
   mono,valid,values=heads[int(x[1])];assert int(x[6])==valid
   if valid:assert int(x[2])==mono and np.max(abs(np.array(x[7:],float)-values))<1e-10
 frames=read_trace(trace);origin=heads[frames[0]['raw']][0];events=[];missing=0;duplicates=0;fit_count=0;roundtrip=0.
 for frame in frames:
  mono,valid,h=heads[frame['raw']];stamp=(mono-origin)/1e9;hands=[]
  if not valid:missing+=1
  if valid:
   hr=Rotation.from_quat(h[:4][[1,2,3,0]]);ht=h[4:]
   for source,row in frame['rows'].items():
    if row['status']!='fitted':continue
    values=row['values'];pose=values[2:];local_q=Rotation.from_quat(pose[:4][[1,2,3,0]]);world=np.eye(4);world[:3,:3]=(hr*local_q).as_matrix();world[:3,3]=(hr.apply(pose[4:7])+ht)*.001
    roundtrip=max(roundtrip,float(np.max(abs(hr.inv().apply(world[:3,3]*1000-ht)-pose[4:7]))))
    hands.append(dict(side=row['side'],source_id=source,wrist_world_m=world.tolist(),joint_angles=pose[7:].tolist(),fit_rmse_px=float(values[1])))
   fit_count+=len(hands)
  ambiguous=any(n>1 for n in Counter(h['side'] for h in hands).values());duplicates+=ambiguous
  events.append(dict(now=stamp+.05,capture=stamp,stamp=stamp,raw_ns=frame['raw'],hands=hands,valid=bool(valid),ambiguous=ambiguous))
 assert all(a['stamp']<b['stamp'] for a,b in zip(events,events[1:])) and roundtrip<1e-9
 output=dict(name=a.run.name,scope=__doc__,events=events,missing_head_joins=missing,ambiguous_fitted_packets=duplicates,hands=fit_count,max_world_roundtrip_mm=roundtrip,provenance=dict(rig_sha256=sha(trace),result_sha256=sha(a.run/'result.json'),head_evidence=head_evidence,source_sha256=sha(Path(__file__))),limitations=['Motion state and packet validation mirror the existing viewer; native publication is not yet integrated.','Fitted hands retain their original source order, including duplicate sides. Replay clears both states on duplicates as the viewer does.','Missing head joins clear states; absent or insufficient hands are not fabricated. No interpolation of head poses.','Raw rig errors are not physical truth.50ms delivery and90Hz ticks are assumptions.'])
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(output,separators=(',',':'))+'\n',encoding='utf-8')
 print(json.dumps({k:output[k] for k in ('hands','missing_head_joins','ambiguous_fitted_packets','max_world_roundtrip_mm')}))
if __name__=='__main__':main()
