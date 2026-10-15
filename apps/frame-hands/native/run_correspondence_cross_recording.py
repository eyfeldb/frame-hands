"""Frozen correspondence-renewal replay on a separately pinned older recording.
Host CPU only: reuse the exact previously tested binary; no tuning, camera or XR.
"""
import argparse,json,os,platform,subprocess,time
from pathlib import Path
from run_tracker_rig_check import sha,stats,verify_head_evidence,recorded_trace_stats

RECORDING_SHA='5833dcb93dc43bd85413915ba6192d55102f8c896d67971db28b4667e55e91d4'
FROZEN_RESULT_SHA='a8cc3db463b00e1cefd3888591db9c304322478141f6caa70d18cea579a212ae'

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('frozen','session','heads','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args()
 if platform.system()!='Linux' or platform.machine()!='x86_64':raise ValueError('Host Linux CPU only')
 if a.output.exists():raise FileExistsError(a.output)
 if '143912Z' in str(a.session):raise ValueError('Accepted holdout excluded')
 if sha(a.frozen/'result.json')!=FROZEN_RESULT_SHA:raise ValueError('Changed reference experiment')
 reference=json.loads((a.frozen/'result.json').read_text());source=(a.frozen/'source').resolve();exe=(a.frozen/'replay').resolve()
 if sha(exe)!=reference['executable_sha256']:raise ValueError('Changed frozen executable')
 for name,digest in reference['source_sha256'].items():
  path=(source/name).resolve()
  if source not in path.parents or sha(path)!=digest:raise ValueError('Changed frozen source: '+name)
 recording=a.session/'frame'
 if sha(recording/'sets.bin')!=RECORDING_SHA:raise ValueError('Wrong independent recording')
 head=verify_head_evidence(a.heads,RECORDING_SHA)
 calibration={}
 for name in ('device_config.json','xrservice.json'):
  observed=recording/'snapshot/device/persist'/name
  if sha(observed)!=sha(source/'device/persist'/name):raise ValueError('Frozen calibration does not match recording')
  calibration[name]=sha(observed)
 a.output.mkdir(parents=True)
 envbase={k:v for k,v in os.environ.items() if not k.startswith('FRAME_HANDS_')}
 for key in ('PAIR_CONTINUITY','OPEN_SIDE','IDENTITY_LATCH','ROTATION_SEARCH','FULL_RATE','FILTER_GUARD','OUTPUT_RECOVERY','JOINT_IDENTITY','RIG_FINGER_CORRESPONDENCE','RIG_CORRESPONDENCE_CONTINUITY','RIG_WHOLE_POSE'):envbase['FRAME_HANDS_'+key]='1'
 envbase.update(FRAME_HANDS_PALM_THRESHOLD='0.2',FRAME_JOB_DEVICE_ROOT=str(source/'device'),FRAME_HANDS_RIG_DIR=str(source/'profiles'),FRAME_HANDS_RIG_REPLAY_HEADS=str((a.heads/'heads.txt').resolve()))
 results=[]
 for index,variant in enumerate(('baseline','candidate','candidate','baseline')):
  name=str(index)+'-'+variant;folder=(a.output/name).resolve();folder.mkdir();env=envbase.copy()
  env.update(FRAME_HANDS_AUDIT=str(folder/'views.txt'),FRAME_HANDS_RIG_TRACE=str(folder/'rig.txt'))
  if variant=='candidate':env['FRAME_HANDS_RIG_CORRESPONDENCE_RENEWAL']='1'
  command=[str(exe),str(recording.resolve()),'--threads','3','--models',str(source/'models/ncnn'),'--to','60','--all-sets','--poses',str(folder/'poses.txt'),'--step-times',str(folder/'times.txt')]
  started=time.monotonic();run=subprocess.run(command,env=env,capture_output=True,text=True,timeout=300)
  for stream in ('stdout','stderr'):(folder/(stream+'.txt')).write_text(getattr(run,stream),encoding='utf-8')
  if run.returncode:raise RuntimeError('Replay failed: '+name)
  times=[float(line.split()[1]) for line in (folder/'times.txt').read_text().splitlines()]
  if len(times)!=1800:raise ValueError('Unexpected coverage')
  row=dict(name=name,variant=variant,processed_frames=len(times),step_ms=stats(times),wall_seconds=time.monotonic()-started,**{n+'_sha256':sha(folder/(n+'.txt')) for n in ('views','poses','rig')},**recorded_trace_stats(folder/'rig.txt'));results.append(row)
  if row['recorded_head_frames']!=len(times) or row['whole_eligible']<=0:raise ValueError('Recorded whole-pose path not exercised')
  print(json.dumps(row),flush=True)
 same=all((r['poses_sha256'],r['views_sha256'])==(results[0]['poses_sha256'],results[0]['views_sha256']) for r in results)
 repeat=results[0]['rig_sha256']==results[3]['rig_sha256'] and results[1]['rig_sha256']==results[2]['rig_sha256']
 report=dict(scope=__doc__,comparison='correspondence_continuity_vs_renewal',seconds=60,original_detector_outputs_identical=same,rig_outputs_repeat_exactly=repeat,results=results,environment=dict(machine=platform.machine(),kernel=platform.release(),os_release=Path('/etc/os-release').read_text()),provenance=dict(frozen_result_sha256=FROZEN_RESULT_SHA,executable_sha256=sha(exe),recording_sha256=RECORDING_SHA,calibration_sha256=calibration,head_evidence=head,runner_sha256=sha(Path(__file__))),limitations=['Same frozen binary and thresholds as first recording; no changes selected from this recording.','Host CPU only, not Frame timing, native integration or wearer acceptance.','Recorded-head missing joins remain invalid. No live queries, cameras or input injection.','Unchanged original detector output is a comparison control, not proof of detector accuracy.','No optical ground truth. Motion measures include real movement.'])
 (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
 if not same or not repeat:raise ValueError('Changed detector output or nondeterministic fits')
 print('Independent-recording ABBA passed',flush=True)

if __name__=='__main__':main()
