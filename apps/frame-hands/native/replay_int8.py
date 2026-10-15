"""Paired PC numerical check of full-encoder INT8 inference; no device access.

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
from int8_encoder import Int8Encoder
from compare_native import compare


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('upstream','session','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--encoder',type=Path,required=True);a=p.parse_args()
    if '143912Z' in str(a.session):raise ValueError('Latest holdout excluded')
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    pipes=[ImageIdentityPipeline(a.upstream),ImageIdentityPipeline(a.upstream)]
    encoder=Int8Encoder(a.encoder)
    for model in (pipes[1].evidence.backend.model,pipes[1].backend.model):model._feature_extractor._image_backbone=encoder
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
            frames+=1
            if frames%100==0:print(frames,'frames',flush=True)
    result=compare(a.output/'control.jsonl',a.output/'candidate.jsonl')
    result.update(scope='PC causal numerical screen of full-width calibrated INT8 encoder versus full FP32 teacher. No native timing, physical truth or live acceptance.',session=a.session.name,encoder_manifest_sha256=hashlib.sha256((a.encoder/'manifest.json').read_bytes()).hexdigest(),encoder_passes=passes,pc_processing_ms_p50_p95_max=[np.percentile(t,[50,95,100]).tolist() for t in times],torch=torch.__version__,cuda=torch.version.cuda,tf32=False,source_sha256={name:hashlib.sha256((BASE/name).read_bytes()).hexdigest() for name in ('int8_encoder.py','replay_int8.py')})
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
