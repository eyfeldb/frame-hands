"""Offline paired recovery using exact current baseline captures and saved pixels.

Baseline poses are precomputed causal input, not ground truth. No future hand
poses enter recovery. No device, camera, XR scene or driver access.
"""
import argparse,hashlib,json,shutil,sys,time
from pathlib import Path
import numpy as np,torch
from frame_data import Recording,load_calibration
from live_backend import Backend
from run_mesh_trial import pose_matrix
from roi_recovery import RoiRecovery,RecoveryState
from replay_capture_join import exact_capture_pairs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('session','baseline','upstream','models','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu');p.add_argument('--pixels',choices=('real','zero'),default='real',help='Zero affects fallback only; current baseline is unchanged')
    p.add_argument('--reuse-crop-geometry',action='store_true',help='Reuse current/neutral/open geometry across cameras within this capture only')
    p.add_argument('--delivered-frames-only',action='store_true',help='Only exact baseline-delivered captures; never fill skipped raw images')
    p.add_argument('--start',type=float,default=0.);p.add_argument('--duration',type=float)
    a=p.parse_args()
    if a.device=='cuda':
        if not torch.cuda.is_available():p.error('CUDA unavailable')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sys.path.insert(0,str(a.upstream));a.output.mkdir(parents=True,exist_ok=False)
    snapshot=a.output/'snapshot';snapshot.mkdir()
    for path in Path(__file__).parent.glob('*.py'):shutil.copy2(path,snapshot/path.name)
    calibration=a.session/'calibration'
    backend=Backend(a.upstream,load_calibration(calibration if calibration.exists() else a.session/'frame/snapshot/device/persist'))
    engine=RoiRecovery(backend,a.models,a.device,a.reuse_crop_geometry)
    manifest=dict(scope=__doc__,args={k:str(v) for k,v in vars(a).items()},torch=torch.__version__,cuda=torch.version.cuda,ncnn=engine.verifier.version,source_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in snapshot.glob('*.py')},limits_ns=dict(unsupported=RecoveryState.MAX_UNSUPPORTED_NS,anchor=RecoveryState.MAX_ANCHOR_AGE_NS,frame_gap=RecoveryState.MAX_FRAME_GAP_NS))
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    counts=dict(frames=0,baseline_hands=0,output_hands=0,proposals=0,recovered=0,ownership_rejected=0,invalid_pose=0,attempt_frames=0);durations=[];attempts=[]
    recording=Recording(a.session/'frame/sets.bin');end=float('inf') if a.duration is None else a.start+a.duration
    with a.baseline.open() as source,(a.output/'predictions.jsonl').open('x') as out,(a.output/'recovery.jsonl').open('x') as trace:
        delivery={}
        baseline_rows=(json.loads(line) for line in source)
        for (seconds,stamp,images),current in exact_capture_pairs(recording.frames(a.start,end),baseline_rows,a.delivered_frames_only,delivery):
            valid=current['capture_pose']['valid'];world=pose_matrix(current['capture_pose']['capture']) if valid else None
            started=time.perf_counter();hands,debug=engine.step(stamp,world,images,current['hands'],a.pixels);elapsed=(time.perf_counter()-started)*1000
            # The baseline is authoritative wherever a current detector pose exists.
            for h in current['hands']:
                if h not in hands:raise AssertionError('Current baseline hand changed or suppressed')
            result={k:v for k,v in current.items() if k not in ('hands','processing_ms')}
            result.update(hands=hands,processing_ms=current.get('processing_ms',0)+elapsed,recovery_processing_ms=elapsed,recovery_pixel_control=a.pixels)
            out.write(json.dumps(result,allow_nan=False)+'\n')
            trace.write(json.dumps(dict(seconds=seconds,stamp=stamp,**debug),allow_nan=False)+'\n')
            counts['frames']+=1;counts['baseline_hands']+=len(current['hands']);counts['output_hands']+=len(hands)
            counts['proposals']+=len(debug['proposals']);counts['recovered']+=len(hands)-len(current['hands']);counts['invalid_pose']+=not valid
            counts['ownership_rejected']+=sum(d['presence_supported'] and not d['supported'] for d in debug['decisions'])
            durations.append(elapsed)
            if debug['attempted']:counts['attempt_frames']+=1;attempts.append(elapsed)
    summary=dict(delivery=delivery,scope=__doc__+' Output counts are not accuracy. Recovery cost excludes precomputed baseline; summed processing_ms is an offline serial estimate, not live latency.',counts=counts,recovery_ms_p50_p95_max=np.percentile(durations,[50,95,100]).tolist(),attempt_ms_p50_p95_max=np.percentile(attempts,[50,95,100]).tolist() if attempts else [])
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
