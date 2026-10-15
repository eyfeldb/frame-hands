"""Build and run an isolated original-detector/rig replay on Frame; no cameras or XR."""
import argparse,hashlib,json,os,platform,subprocess,time
from pathlib import Path
from workflow_guard import native_workflow

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()

def stats(values):
 values=sorted(values)
 def q(p):
  x=(len(values)-1)*p;i=int(x);return values[i]+(values[min(i+1,len(values)-1)]-values[i])*(x-i)
 return dict(p50=q(.5),p95=q(.95),maximum=max(values))

def verify_head_evidence(root, recording_sha256):
 root=Path(root).resolve();meta=json.loads((root/'manifest.json').read_text())
 if meta.get('version')!=1 or meta.get('recording_sha256')!=recording_sha256 or meta.get('clock')!='CLOCK_MONOTONIC' or meta.get('space')!='raw_world_mm':raise ValueError('Wrong head evidence recording/contract')
 if not 0<meta.get('valid_frames',0)<=meta.get('frames',0)<=100000:raise ValueError('Unbounded or empty head evidence')
 path=root/'heads.txt'
 if sha(path)!=meta.get('heads_sha256'):raise ValueError('Changed recorded heads')
 with path.open() as f:header=f.readline().strip()
 if header!='FHHEAD 1 CLOCK_MONOTONIC raw_world_mm '+recording_sha256:raise ValueError('Wrong recorded head header')
 return dict(path=str(root),heads_sha256=sha(path),manifest_sha256=sha(root/'manifest.json'),recording_sha256=recording_sha256)


def recorded_trace_stats(path):
 result=dict(recorded_head_frames=0,valid_head_frames=0,whole_eligible=0,whole_selected=0)
 with Path(path).open() as stream:
  for line in stream:
   x=line.split()
   if not x:continue
   if x[0]=='head':raise ValueError('Live head query in recorded replay')
   if x[0]=='head_recorded':
    if len(x)!=14 or int(x[1])<=0 or int(x[3])!=0 or int(x[4])!=0 or x[6] not in ('0','1'):raise ValueError('Invalid recorded head trace')
    if x[6]=='1' and (int(x[2])<=0 or int(x[5])<=0):raise ValueError('Missing valid head context')
    if not all(__import__('math').isfinite(float(v)) for v in x[7:]):raise ValueError('Invalid recorded transform')
    result['recorded_head_frames']+=1;result['valid_head_frames']+=int(x[6])
   elif x[0]=='whole':
    if len(x)!=6 or x[2] not in ('0','1') or x[3] not in ('0','1') or int(x[3])>int(x[2]):raise ValueError('Invalid whole-pose decision')
    result['whole_eligible']+=int(x[2]);result['whole_selected']+=int(x[3])
 return result


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('bundle','output','recording','ncnn'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--recording-sha256',required=True);p.add_argument('--seconds',type=float,default=10)
 p.add_argument('--head-evidence',type=Path);p.add_argument('--scheduled',action='store_true');a=p.parse_args();root=a.bundle.resolve()
 if platform.machine()!='aarch64' or not 1<=a.seconds<=65:raise ValueError('Wrong platform or unbounded duration')
 if '143912Z' in str(a.recording):raise ValueError('Accepted holdout excluded')
 if a.output.exists():raise FileExistsError(a.output)
 manifest=json.loads((root/'manifest.json').read_text())
 for name,value in manifest['sha256'].items():
  path=(root/name).resolve()
  if root not in path.parents or sha(path)!=value:raise ValueError('Changed/uncontained source '+name)
 if sha(a.recording/'sets.bin')!=a.recording_sha256:raise ValueError('Recording differs')
 head_evidence=verify_head_evidence(a.head_evidence,a.recording_sha256) if a.head_evidence else None
 library=a.ncnn/'lib/libncnn.a'
 if not library.is_file():raise ValueError('Existing ncnn build required')
 with native_workflow():
  a.output.mkdir(parents=True);exe=(a.output/'ft-handreplay-rig').resolve()
  build=['c++','-std=c++17','-O2','-fopenmp','-Wno-psabi','-I'+str(a.ncnn/'include/ncnn'),*[str(root/'track'/n) for n in ('replay.cpp','calib.cpp','nets.cpp','tracker.cpp','io.cpp','record.cpp','pinch.cpp')],str(library),'-ljsoncpp','-lpthread','-o',str(exe)]
  if manifest.get('capture_head_support'):build[1:1]=['-DFRAME_HANDS_WITH_OPENVR'];build.insert(-2,'-ldl')
  compiled=subprocess.run(build,capture_output=True,text=True,timeout=120)
  (a.output/'build.stdout').write_text(compiled.stdout);(a.output/'build.stderr').write_text(compiled.stderr)
  if compiled.returncode:raise RuntimeError('Native tracker build failed')
  base={k:v for k,v in os.environ.items() if not k.startswith('FRAME_HANDS_')}
  for key in ('PAIR_CONTINUITY','OPEN_SIDE','IDENTITY_LATCH','ROTATION_SEARCH','FULL_RATE','FILTER_GUARD','OUTPUT_RECOVERY','JOINT_IDENTITY'):base['FRAME_HANDS_'+key]='1'
  base['FRAME_HANDS_PALM_THRESHOLD']='0.2';base['FRAME_JOB_DEVICE_ROOT']=str(root/'device')
  results=[]
  for index,variant in enumerate(('baseline','candidate','candidate','baseline')):
   name=str(index)+'-'+variant;folder=a.output/name;folder.mkdir();env=base.copy();env['FRAME_HANDS_AUDIT']=str((folder/'views.txt').resolve())
   if variant=='candidate' or head_evidence:env['FRAME_HANDS_RIG_DIR']=str(root/'profiles');env['FRAME_HANDS_RIG_TRACE']=str((folder/'rig.txt').resolve())
   if variant=='candidate' and head_evidence:env['FRAME_HANDS_RIG_WHOLE_POSE']='1';env['FRAME_HANDS_RIG_REPLAY_HEADS']=str(Path(head_evidence['path'])/'heads.txt')
   command=[str(exe),str(a.recording.resolve()),'--threads','3','--models',str(root/'models/ncnn'),'--to',str(a.seconds),'--poses',str((folder/'poses.txt').resolve()),'--step-times',str((folder/'times.txt').resolve())]
   if not a.scheduled:command+=['--all-sets']
   started=time.monotonic();run=subprocess.run(command,env=env,capture_output=True,text=True,timeout=240)
   (folder/'stdout.txt').write_text(run.stdout);(folder/'stderr.txt').write_text(run.stderr)
   if run.returncode:raise RuntimeError('Replay failed: '+name)
   times=[line.split() for line in (folder/'times.txt').read_text().splitlines()]
   if not times:raise ValueError('No processed input')
   results.append(dict(name=name,variant=variant,processed_frames=len(times),step_ms=stats([float(x[1]) for x in times]),wall_seconds=time.monotonic()-started,poses_sha256=sha(folder/'poses.txt'),views_sha256=sha(folder/'views.txt')))
   if head_evidence:
    results[-1].update(rig_sha256=sha(folder/'rig.txt'),**recorded_trace_stats(folder/'rig.txt'))
    if variant=='candidate' and (results[-1]['recorded_head_frames']!=len(times) or not results[-1]['valid_head_frames'] or not results[-1]['whole_eligible']):raise ValueError('Recorded whole-pose path not exercised')
  same=all(r['poses_sha256']==results[0]['poses_sha256'] and r['views_sha256']==results[0]['views_sha256'] for r in results)
  report=dict(scope=__doc__,comparison='fixed_rig_vs_whole_pose' if head_evidence else 'detector_vs_fixed_rig',seconds=a.seconds,scheduling='measured busy scheduling' if a.scheduled else 'all saved sets',results=results,original_detector_outputs_identical=same,provenance=dict(head_evidence=head_evidence,executable_sha256=sha(exe),ncnn_library_sha256=sha(library),recording_sha256=a.recording_sha256,manifest_sha256=sha(root/'manifest.json'),build=build),environment=dict(machine=platform.machine(),kernel=platform.release(),os_release=Path('/etc/os-release').read_text(),compiler=subprocess.run(['c++','--version'],capture_output=True,text=True,check=True).stdout),limitations=['Native saved camera inference plus fitting; no live camera service, XR, physical latency or wearer acceptance. Recorded-head mode never queries OpenVR and does not measure runtime HMD query cost.','Step timing includes optional rig trace buffering; ABBA is sequential, not simultaneous.','All-sets mode suppresses busy drops for identical input comparison; scheduled mode measures counterfactual replay skipping, not actual live capture.','Rig outputs are separate; existing point publication, source tracking and controls stay unchanged.','No monocular rig prediction or native mesh publication yet.'])
  (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
  if not a.scheduled and not same:raise ValueError('Adapter changed original detector outputs or replay is nondeterministic')
if __name__=='__main__':main()
