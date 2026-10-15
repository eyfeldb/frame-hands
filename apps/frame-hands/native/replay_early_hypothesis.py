"""Paired PC numerical check of early-hypothesis inference; no device access.

Uses saved development crops to verify actual changed neural execution before
native timing. PC timings are diagnostic only, never a Frame speed claim.
"""
import argparse,io,json,sys,time,hashlib
from pathlib import Path
import cv2,numpy as np,torch
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE.parent/'exploratory'))
from frame_data import read_audit,load_calibration
from replay_mesh_trial import packets
from run_mesh_trial import pose_matrix
from identity_pipeline import ImageIdentityPipeline
from early_evidence import EarlyHypothesisEvidence
from compare_native import compare


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('upstream','session','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--require-stereo-exit',action='store_true');a=p.parse_args()
    if '143912Z' in str(a.session):raise ValueError('Latest holdout excluded')
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    pipes=[ImageIdentityPipeline(a.upstream),ImageIdentityPipeline(a.upstream)]
    pipes[1].evidence=EarlyHypothesisEvidence(a.upstream,{},optimized=True,device='cuda')
    pipes[1].evidence.require_stereo_exit=a.require_stereo_exit;pipes[1].evidence.ownership=pipes[1].backend.ownership
    pipes[1].evidence.regressor.warmup()
    passes=[0,0]
    for n,pipe in enumerate(pipes):
        method=pipe.evidence.regressor.extract
        def extract(images,n=n,method=method):
            passes[n]+=len(images);return method(images)
        pipe.evidence.regressor.extract=extract
        pipe.bind_calibration(load_calibration(a.session/'calibration'))
    refs={r['capture_raw_ns']:r for line in (a.session/'predictions.jsonl').open() if (r:=json.loads(line))}
    frames=late=0;times=[[],[]]
    from contextlib import ExitStack
    with ExitStack() as stack:
        outputs=[stack.enter_context((a.output/name).open('w')) for name in ('control.jsonl','candidate.jsonl')]
        traces=[stack.enter_context((a.output/name).open('w')) for name in ('control-identity.jsonl','candidate-identity.jsonl')]
        for meta,blobs in packets(a.session/'transport.bin'):
            if meta.get('waiting'):continue
            ref=refs.get(meta['stamp'])
            if ref is None:
                for pipe in pipes:pipe.reset('Missing saved context')
                continue
            observation=next(iter(read_audit(io.StringIO(meta['audit'])).values()))
            prepared={(v['camera'],v['id']):(cv2.imdecode(np.frombuffer(blob,np.uint8),0),v) for v,blob in zip(meta['crops'],blobs)}
            world=pose_matrix(ref['capture_pose']['capture']) if ref['capture_pose']['valid'] else None
            for n,pipe in enumerate(pipes):
                started=time.perf_counter();result=pipe.predict(observation,world,prepared);times[n].append((time.perf_counter()-started)*1000)
                if not ref['current_pose']['valid']:result['hands']=[]
                outputs[n].write(json.dumps(dict(stamp=meta['stamp'],**result),allow_nan=False)+'\n')
                traces[n].write(json.dumps(dict(stamp=meta['stamp'],**pipe.last_debug),allow_nan=False)+'\n')
            if pipes[1].evidence.features is not None:late+=pipes[1].evidence.features.late_passes
            frames+=1
            if frames%100==0:print(frames,'frames',flush=True)
    result=compare(a.output/'control.jsonl',a.output/'candidate.jsonl')
    result.update(scope='Actual paired neural execution on PC saved inputs. Teacher agreement, not physical truth or native speed.',session=a.session.name,require_stereo_exit=a.require_stereo_exit,completed_views=pipes[1].evidence.completed_views,completed_frames=pipes[1].evidence.completed_frames,encoder_passes=passes,encoder_pass_saving_fraction=1-passes[1]/passes[0],early_exit_views=pipes[1].evidence.exit_count,late_opposite_passes=late,pc_processing_ms_p50_p95_max=[np.percentile(t,[50,95,100]).tolist() for t in times],torch=torch.__version__,cuda=torch.version.cuda,tf32=False,source_sha256={name:hashlib.sha256((BASE/name).read_bytes()).hexdigest() for name in ('early_hypothesis.py','early_evidence.py','replay_early_hypothesis.py','stereo_exit.py')})
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
