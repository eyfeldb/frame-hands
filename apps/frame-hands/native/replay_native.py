"""Replay compact saved camera crops entirely on Frame. No live camera or XR access."""
import argparse,fcntl,io,json,platform,sys,time,subprocess
from pathlib import Path
import cv2,numpy as np
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE.parent/'exploratory'))
from frame_data import read_audit,load_calibration
from replay_mesh_trial import packets
from run_mesh_trial import pose_matrix
from native_pipeline import NativeIdentityPipeline

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('fixture','upstream','blocks','output'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--require-stereo-exit',action='store_true');p.add_argument('--early-hypothesis',action='store_true');p.add_argument('--cached-skinning',action='store_true');p.add_argument('--mode',choices=('cpu','hybrid_fp32','hybrid_fp16','hybrid_fp16_arithmetic'),default='hybrid_fp32');p.add_argument('--threads',type=int,default=4);a=p.parse_args()
 if not 1<=a.threads<=8:p.error('threads must be1..8')
 locks=[]
 for path in (Path('/home/steamos/devkit-game/FrameEyes/workflow.lock'),Path('/home/steamos/devkit-game/FrameHands/workflow.lock')):
  if path.exists():
   lock=path.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
 if subprocess.run(['systemctl','is-active','--quiet','frame-eyes-reader-brightness.service']).returncode==0:raise RuntimeError('Eye workflow active')
 from workflow_guard import require_no_optical_processes
 require_no_optical_processes()
 a.output.mkdir(exist_ok=False)
 old={r['capture_raw_ns']:r for line in (a.fixture/'reference.jsonl').open() if (r:=json.loads(line))}
 pipe=NativeIdentityPipeline(a.upstream,a.blocks,a.mode,a.threads,early_hypothesis=a.early_hypothesis,require_stereo_exit=a.require_stereo_exit);pipe.bind_calibration(load_calibration(a.fixture/'calibration'))
 if a.cached_skinning:
  from cached_skinning import bind_cached_skinning
  bind_cached_skinning(pipe)
 times=[];vertex=[];differences=[];counts=[]
 with (a.output/'predictions.jsonl').open('w') as predictions,(a.output/'identity.jsonl').open('w') as trace:
  for meta,blobs in packets(a.fixture/'transport.bin'):
   if meta.get('waiting'):continue
   ref=old.get(meta['stamp'])
   if ref is None:pipe.reset('Missing replay context');continue
   observation=next(iter(read_audit(io.StringIO(meta['audit'])).values()))
   prepared={(v['camera'],v['id']):(cv2.imdecode(np.frombuffer(blob,np.uint8),0),v) for v,blob in zip(meta['crops'],blobs)}
   world=pose_matrix(ref['capture_pose']['capture']) if ref['capture_pose']['valid'] else None
   start=time.perf_counter();result=pipe.predict(observation,world,prepared);times.append((time.perf_counter()-start)*1000)
   if not ref['current_pose']['valid']:result['hands']=[]
   counts.append(len(result['hands']));actual=[(h['side'],h['source_id']) for h in result['hands']];expected=[(h['side'],h['source_id']) for h in ref['hands']]
   if actual!=expected:differences.append({'stamp':meta['stamp'],'actual':actual,'expected':expected})
   else:
    for h,r in zip(result['hands'],ref['hands']):vertex.append(float(abs(np.asarray(h['vertices_world_mm'])-r['vertices_world_mm']).max()))
   predictions.write(json.dumps(dict(stamp=meta['stamp'],**result),allow_nan=False)+'\n');trace.write(json.dumps(dict(stamp=meta['stamp'],**pipe.last_debug),allow_nan=False)+'\n')
   if len(times)%50==0:print(len(times),'frames',flush=True)
 result={'scope':'Native saved-crop replay, not camera-to-render latency or live acceptance. No external inference service.','mode':a.mode,'threads':a.threads,'cached_skinning':a.cached_skinning,'early_hypothesis':a.early_hypothesis,'require_stereo_exit':a.require_stereo_exit,'completed_views':getattr(pipe.evidence,'completed_views',0),'completed_frames':getattr(pipe.evidence,'completed_frames',0),'early_exit_views':getattr(pipe.evidence,'exit_count',0),'platform':platform.platform(),'frames':len(times),'hands':sum(counts),'assignment_differences':differences,'max_vertex_difference_mm':max(vertex,default=None),'processing_ms_p50_p95_max':np.percentile(times,[50,95,100]).tolist(),'nonempty_processing_ms_p50_p95_max':np.percentile([t for t,c in zip(times,counts) if c],[50,95,100]).tolist(),'block_calls':pipe.bank.counts,'block_total_ms':pipe.bank.elapsed}
 (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
