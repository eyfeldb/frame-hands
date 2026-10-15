"""Replay an existing crop-trace binary; require exact detector output on new data."""
import argparse,json,os,platform,subprocess
from pathlib import Path
from run_tracker_rig_check import sha,verify_head_evidence

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('diagnostic','reference','session','heads','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args()
 if platform.system()!='Linux' or platform.machine()!='x86_64':raise ValueError('Host Linux CPU only')
 if '143912Z' in str(a.session) or a.output.exists():raise ValueError('Holdout or existing output')
 evidence=json.loads((a.diagnostic/'result.json').read_text());reference=json.loads((a.reference/'result.json').read_text())
 assert reference['original_detector_outputs_identical'] and reference['rig_outputs_repeat_exactly']
 source=(a.diagnostic/'source').resolve();exe=(a.diagnostic/'replay').resolve()
 assert sha(exe)==evidence['executable_sha256'] and sha(source/'manifest.json')==evidence['bundle_sha256']
 for name,digest in json.loads((source/'manifest.json').read_text())['sha256'].items():
  if name=='track/tracker.cpp':digest=evidence['patched_source_sha256']
  path=(source/name).resolve();assert source in path.parents and sha(path)==digest,name
 recording=a.session/'frame';recording_sha=sha(recording/'sets.bin');assert recording_sha==reference['provenance']['recording_sha256']
 head=verify_head_evidence(a.heads,recording_sha)
 for name in ('device_config.json','xrservice.json'):assert sha(source/'device/persist'/name)==sha(recording/'snapshot/device/persist'/name)
 a.output.mkdir(parents=True)
 env={k:v for k,v in os.environ.items() if not k.startswith('FRAME_HANDS_')}
 for key in ('PAIR_CONTINUITY','OPEN_SIDE','IDENTITY_LATCH','ROTATION_SEARCH','FULL_RATE','FILTER_GUARD','OUTPUT_RECOVERY','JOINT_IDENTITY','RIG_WHOLE_POSE'):env['FRAME_HANDS_'+key]='1'
 env.update(FRAME_HANDS_PALM_THRESHOLD='0.2',FRAME_JOB_DEVICE_ROOT=str(source/'device'),FRAME_HANDS_RIG_DIR=str(source/'profiles'),FRAME_HANDS_AUDIT=str(a.output.resolve()/'views.txt'),FRAME_HANDS_RIG_TRACE=str(a.output.resolve()/'rig.txt'),FRAME_HANDS_RIG_REPLAY_HEADS=str((a.heads/'heads.txt').resolve()),FRAME_HANDS_LANDMARK_DIAGNOSTIC=str(a.output.resolve()/'crops.txt'))
 command=[str(exe),str(recording.resolve()),'--threads','3','--models',str(source/'models/ncnn'),'--to',str(reference['seconds']),'--all-sets','--poses',str(a.output.resolve()/'poses.txt')]
 run=subprocess.run(command,env=env,capture_output=True,text=True,timeout=300)
 for stream in ('stdout','stderr'):(a.output/(stream+'.txt')).write_text(getattr(run,stream),encoding='utf-8')
 if run.returncode:raise RuntimeError('Trace replay failed')
 for name in ('views','poses'):assert sha(a.output/(name+'.txt'))==reference['results'][0][name+'_sha256'],name+' changed'
 result=dict(scope=__doc__,outputs_match=True,matched_outputs=['views','poses'],rig_match_required=False,trace_sha256=sha(a.output/'crops.txt'),reference_sha256=sha(a.reference/'result.json'),diagnostic_sha256=sha(a.diagnostic/'result.json'),executable_sha256=sha(exe),runner_sha256=sha(Path(__file__)),head_evidence=head,recording_sha256=recording_sha,limitations=['Exact detector output parity permits inspecting crop traces only. Diagnostic has original whole-pose rig, not correspondence candidate; its rig output is not used.','No Frame execution, camera capture, live head queries or input.'])
 (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print('Exact views and poses reproduced; crop trace ready.',flush=True)
if __name__=='__main__':main()
