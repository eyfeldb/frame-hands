"""Replay exact recorded inputs through stateless chirality evidence and temporal pose tracking.

No annotations, headset, driver or future frames are read by this candidate.
"""
import argparse,io,json,sys,time,shutil,hashlib
from pathlib import Path
import cv2,numpy as np,torch
from chirality_evidence import ChiralityEvidence
from live_backend import Backend
from frame_data import read_audit,load_calibration
from replay_mesh_trial import packets
from run_mesh_trial import pose_matrix


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',type=Path,required=True);p.add_argument('--upstream',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--optimized',action='store_true',help='Reuse same-frame image features and batch skinning; offline only')
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--warmup',action='store_true',help='Initialize synthetic inference shapes before replay, then reset memory')
    p.add_argument('--live-pipeline',action='store_true',help='Replay the explicit supervised CUDA pipeline without hardware')
    p.add_argument('--fixed-hand-scale',type=float,default=1.,help='Offline fixed pose-skeleton/mesh scale; image identity remains unscaled')
    a=p.parse_args();
    if not np.isfinite(a.fixed_hand_scale) or not .5<=a.fixed_hand_scale<=1.5:p.error('Invalid fixed scale')
    if a.live_pipeline and a.fixed_hand_scale!=1.:p.error('Live identity pipeline does not expose hand scaling')
    if a.live_pipeline and not (a.optimized and a.warmup and a.device=='cuda'):p.error('Live pipeline requires optimized CUDA with warmup')
    if (a.device!='cpu' or a.warmup) and not a.optimized:p.error('CUDA/warmup requires --optimized')
    if a.device=='cuda':
        if not torch.cuda.is_available():p.error('CUDA is unavailable in this environment')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sys.path.insert(0,str(a.upstream));a.output.mkdir(parents=True,exist_ok=False)
    snapshot=a.output/'snapshot';snapshot.mkdir()
    for source in Path(__file__).parent.glob('*.py'):shutil.copy2(source,snapshot/source.name)
    (snapshot/'manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.glob('*.py')},indent=2)+'\n')
    # Retain only timing/pose metadata. Keeping every baseline vertex list alive
    # makes Python scan millions of irrelevant objects during timed inference.
    old={}
    with (a.session/'predictions.jsonl').open() as source:
        for line in source:
            row=json.loads(line)
            old[row['capture_raw_ns']]={key:row[key] for key in ('capture_mono_ns','capture_pose','current_pose')}
    del row
    pipeline=None
    if a.live_pipeline:
        from identity_pipeline import ImageIdentityPipeline
        pipeline=ImageIdentityPipeline(a.upstream)
    cameras=load_calibration(a.session/'calibration')
    if pipeline:
        pipeline.bind_calibration(cameras);evidence=pipeline.evidence;backend=pipeline.backend
    else:
        evidence=ChiralityEvidence(a.upstream,cameras,optimized=a.optimized,device=a.device)
        from fixed_scale_backend import ScaledBackend
        backend=ScaledBackend(a.upstream,cameras,association='temporal',hand_scale=a.fixed_hand_scale)
    warmup_seconds=pipeline.preparation_seconds if pipeline else 0.
    if a.warmup and not pipeline:
        from feature_reuse import PreparedRegressor
        warmup_started=time.perf_counter()
        backend.prepared_regressor=PreparedRegressor(backend.model,backend.skeleton,device=a.device)
        evidence.regressor.warmup();backend.prepared_regressor.warmup()
        warmup_seconds=time.perf_counter()-warmup_started
    frames=hands=agreements=views=0;times=[];evidence_times=[]
    with (a.output/'predictions.jsonl').open('x') as predictions,(a.output/'evidence.jsonl').open('x') as trace:
        for meta,blobs in packets(a.session/'transport.bin'):
            if meta.get('waiting'):continue
            reference=old.get(meta['stamp'])
            if reference is None:
                if pipeline:pipeline.reset('Missing replay capture context')
                else:backend.previous={}
                continue
            obs=next(iter(read_audit(io.StringIO(meta['audit'])).values()))
            prepared={(v['camera'],v['id']):(cv2.imdecode(np.frombuffer(blob,np.uint8),0),v) for v,blob in zip(meta['crops'],blobs)}
            world=pose_matrix(reference['capture_pose']['capture']) if reference['capture_pose']['valid'] else None
            start=time.perf_counter();scores=[];extra_ms=0.
            if pipeline:
                result=pipeline.predict(obs,world,prepared)
                scores=pipeline.last_debug.get('views',[]);extra_ms=pipeline.last_debug.get('evidence_ms',0.)
            else:
                if world is not None:obs,scores,extra_ms=evidence.score(obs,world,prepared)
                result=backend.predict(obs,{},world,prepared,feature_cache=evidence.features if a.optimized and world is not None else None)
            elapsed=(time.perf_counter()-start)*1000
            if not reference['current_pose']['valid']:result['hands']=[]
            result['processing_ms']=elapsed
            predictions.write(json.dumps(dict(capture_raw_ns=meta['stamp'],capture_mono_ns=reference['capture_mono_ns'],capture_pose=reference['capture_pose'],current_pose=reference['current_pose'],**result),allow_nan=False)+'\n')
            trace.write(json.dumps(dict(stamp=meta['stamp'],views=scores,evidence_ms=extra_ms,association=backend.ownership.last_debug),allow_nan=False)+'\n')
            frames+=1;hands+=len(result['hands']);views+=len(scores);agreements+=sum(r['agreement'] for r in scores);times.append(elapsed);evidence_times.append(extra_ms)
    summary=dict(fixed_hand_scale=a.fixed_hand_scale,live_pipeline=a.live_pipeline,warmup_seconds=warmup_seconds,device=a.device,torch_version=torch.__version__,cuda_version=torch.version.cuda,tf32_enabled=False if a.device=='cuda' else None,optimized=a.optimized,frames=frames,hands=hands,view_hypothesis_pairs=views,agreed_pairs=agreements,processing_ms_p50_p95_max=np.percentile(times,[50,95,100]).tolist(),evidence_ms_p50_p95_max=np.percentile(evidence_times,[50,95,100]).tolist(),scope=__doc__+' Synchronized replay processing cost is not achieved live latency or physical accuracy.')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
