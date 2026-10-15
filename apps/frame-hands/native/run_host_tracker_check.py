"""Host-only Linux CPU camera replay of a frozen native bundle; never native acceptance.

No cameras, OpenVR queries, OS input, service changes, or device communication.
The native builder rejects this result because its machine is x86_64.
"""
import argparse,json,os,platform,subprocess,time
from pathlib import Path
from run_tracker_rig_check import sha,stats,verify_head_evidence,recorded_trace_stats

RECORDING_SHA='34eb755f1f5285cdf0c94c6d45250ccb3739fe2409abbb958485ea13ef40d3d4'
BUNDLE_SHA='c26c4f1b19d60e797f40ba261b9bde3e2d682314f30140383492ac14fe166aad'

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('bundle','recording','heads','ncnn','json-include','json-library','output'):
  p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--seconds',type=float,default=10)
 a=p.parse_args()
 if platform.system()!='Linux' or platform.machine()!='x86_64':raise ValueError('Host-only x86_64 Linux experiment')
 if not 1<=a.seconds<=60:raise ValueError('Replay must be bounded to1..60seconds')
 if a.output.exists():raise FileExistsError(a.output)
 root=a.bundle.resolve()
 if sha(root/'manifest.json')!=BUNDLE_SHA:raise ValueError('Wrong frozen bundle')
 manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
 for name,digest in manifest['sha256'].items():
  f=(root/name).resolve()
  if root not in f.parents or sha(f)!=digest:raise ValueError('Changed bundle: '+name)
 if sha(a.recording/'sets.bin')!=RECORDING_SHA:raise ValueError('Only the pinned older recording is permitted')
 head=verify_head_evidence(a.heads,RECORDING_SHA)
 library=(a.ncnn/'lib/libncnn.a').resolve()
 if not library.is_file() or not (a.json_include/'json/json.h').is_file() or not a.json_library.is_file():raise ValueError('Existing local dependencies required')
 a.output.mkdir(parents=True);exe=(a.output/'ft-handreplay-host').resolve()
 command=['c++','-std=c++17','-O2','-fopenmp','-Wno-psabi','-DFRAME_HANDS_WITH_OPENVR','-I'+str((a.ncnn/'include/ncnn').resolve()),'-I'+str(a.json_include.resolve()),*[str(root/'track'/name) for name in ('replay.cpp','calib.cpp','nets.cpp','tracker.cpp','io.cpp','record.cpp','pinch.cpp')],str(library),str(a.json_library.resolve()),'-lpthread','-ldl','-o',str(exe)]
 build=subprocess.run(command,capture_output=True,text=True,timeout=180)
 (a.output/'build.stdout').write_text(build.stdout,encoding='utf-8');(a.output/'build.stderr').write_text(build.stderr,encoding='utf-8')
 if build.returncode:raise RuntimeError('Host integration link failed')
 print('host replay linked',flush=True)
 base={k:v for k,v in os.environ.items() if not k.startswith('FRAME_HANDS_')}
 for key in ('PAIR_CONTINUITY','OPEN_SIDE','IDENTITY_LATCH','ROTATION_SEARCH','FULL_RATE','FILTER_GUARD','OUTPUT_RECOVERY','JOINT_IDENTITY'):base['FRAME_HANDS_'+key]='1'
 base['FRAME_HANDS_PALM_THRESHOLD']='0.2';base['FRAME_JOB_DEVICE_ROOT']=str(root/'device')
 results=[]
 for index,variant in enumerate(('baseline','candidate','candidate','baseline')):
  name=str(index)+'-'+variant;folder=(a.output/name).resolve();folder.mkdir();env=base.copy()
  env['FRAME_HANDS_AUDIT']=str(folder/'views.txt');env['FRAME_HANDS_RIG_DIR']=str(root/'profiles');env['FRAME_HANDS_RIG_TRACE']=str(folder/'rig.txt')
  if variant=='candidate':env['FRAME_HANDS_RIG_WHOLE_POSE']='1';env['FRAME_HANDS_RIG_REPLAY_HEADS']=str(Path(head['path'])/'heads.txt')
  cmd=[str(exe),str(a.recording.resolve()),'--threads','3','--models',str(root/'models/ncnn'),'--to',str(a.seconds),'--all-sets','--poses',str(folder/'poses.txt'),'--step-times',str(folder/'times.txt')]
  started=time.monotonic();run=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=300)
  (folder/'stdout.txt').write_text(run.stdout,encoding='utf-8');(folder/'stderr.txt').write_text(run.stderr,encoding='utf-8')
  if run.returncode:raise RuntimeError('Host replay failed: '+name)
  times=[line.split() for line in (folder/'times.txt').read_text().splitlines()]
  if not times:raise ValueError('Empty replay')
  row=dict(name=name,variant=variant,processed_frames=len(times),step_ms=stats([float(x[1]) for x in times]),wall_seconds=time.monotonic()-started,**{n+'_sha256':sha(folder/(n+'.txt')) for n in ('poses','views','rig')},**recorded_trace_stats(folder/'rig.txt'))
  if variant=='candidate' and (row['recorded_head_frames']!=len(times) or row['whole_eligible']<=0):raise ValueError('Whole-pose path was not exercised')
  results.append(row);print(json.dumps(row),flush=True)
 same=all((r['poses_sha256'],r['views_sha256'])==(results[0]['poses_sha256'],results[0]['views_sha256']) for r in results)
 repeat=results[0]['rig_sha256']==results[3]['rig_sha256'] and results[1]['rig_sha256']==results[2]['rig_sha256']
 result=dict(scope=__doc__,comparison='fixed_rig_vs_whole_pose',scheduling='all saved sets',seconds=a.seconds,original_detector_outputs_identical=same,rig_outputs_repeat_exactly=repeat,results=results,provenance=dict(head_evidence=head,manifest_sha256=sha(root/'manifest.json'),recording_sha256=RECORDING_SHA,executable_sha256=sha(exe),ncnn_library_sha256=sha(library),jsoncpp_library_sha256=sha(a.json_library),runner_sha256=sha(Path(__file__)),build=command),environment=dict(machine=platform.machine(),kernel=platform.release(),os_release=Path('/etc/os-release').read_text(),compiler=subprocess.run(['c++','--version'],capture_output=True,text=True,check=True).stdout),limitations=['PC Linux CPU inference integration, not Frame performance or hardware validation.','No live camera, headset pose queries, XR rendering, input injection or accepted tracker changes.','Same saved inputs suppress busy frame skipping. Three host threads; times are not Frame estimates.','Current detector may differ numerically from older ARM detector; comparisons are within this executable only.','Recorded head transforms are saved estimates, not independent motion ground truth.'])
 (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
 if not same or not repeat:raise ValueError('Outputs changed across equal-input runs')
 print('host integration comparison passed',flush=True)
if __name__=='__main__':main()
