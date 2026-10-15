"""Private full-neural replay of index/middle correspondence alternatives; no live inputs."""
import argparse,json,os,shutil,subprocess,time
from pathlib import Path
from run_tracker_rig_check import sha,stats,recorded_trace_stats
from run_host_tracker_check import BUNDLE_SHA

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('bundle','reference','recording','heads','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();reference=json.loads((a.reference/'result.json').read_text())
 assert reference['environment']['machine']=='x86_64' and sha(a.bundle/'manifest.json')==BUNDLE_SHA
 for name,digest in json.loads((a.bundle/'manifest.json').read_text())['sha256'].items():assert sha(a.bundle/name)==digest
 assert sha(a.recording/'sets.bin')==reference['provenance']['recording_sha256'];assert sha(a.heads)==reference['provenance']['head_evidence']['heads_sha256']
 a.output.mkdir(parents=True,exist_ok=False);source=a.output/'source';shutil.copytree(a.bundle,source);
 for filename in ('stereo_disagreement.h','finger_correspondence.h'):shutil.copy2(Path(__file__).with_name(filename),source/'native'/filename)
 path=source/'native/whole_pose_fitter.h';text=path.read_text();text=text.replace('#include "whole_pose_seed.h"','#include "whole_pose_seed.h"\n#include "finger_correspondence.h"\n#include <cstdlib>')
 text=text.replace('struct WholePoseResult {','struct WholePoseResult {CorrespondenceResult correspondence{};')
 marker='        // Mutate only after validating/fitting.'
 assert text.count(marker)==1
 insert='        if(const char* flag=std::getenv("FRAME_HANDS_RIG_FINGER_CORRESPONDENCE");flag&&std::string(flag)=="1"&&fresh_valid){out.correspondence=refine_index_middle_hypotheses(rig,out.fit.pose,out.fit,cameras,pixels,fresh.angles);out.fit=out.correspondence.fit;}\n'
 text=text.replace(marker,insert+marker);path.write_text(text)
 path=source/'native/tracker_rig_adapter.h';text=path.read_text();text=text.replace('RigStageResult result{};','CorrespondenceResult correspondence{};RigStageResult result{};');text=text.replace('output.whole_alternate_cost=full.alternate_cost;','output.whole_alternate_cost=full.alternate_cost;output.correspondence=full.correspondence;');path.write_text(text)
 path=source/'native/tracker_rig_hook.h';text=path.read_text();marker='            if(head_source_||recorded_heads_)trace_<<"whole "'
 assert text.count(marker)==1
 insert='            if(o.result.initialized&&std::getenv("FRAME_HANDS_RIG_FINGER_CORRESPONDENCE")){const auto& c=o.correspondence;trace_<<"correspondence "<<o.source<<\' \'<<c.chosen<<\' \'<<c.original_disagreement;for(double v:c.alternative_disagreement)trace_<<\' \'<<v;for(double v:c.alternative_cost)trace_<<\' \'<<v;trace_<<\'\\n\';}\n'
 text=text.replace(marker,insert+marker);path.write_text(text)
 exe=(a.output/'replay').resolve();command=[s.replace(str(a.bundle.resolve()),str(source.resolve())) for s in reference['provenance']['build']];command[-1]=str(exe)
 build=subprocess.run(command,capture_output=True,text=True,timeout=180);(a.output/'build.stderr').write_text(build.stderr);assert build.returncode==0,build.stderr;print('correspondence replay linked',flush=True)
 envbase={k:v for k,v in os.environ.items() if not k.startswith('FRAME_HANDS_')}
 for key in ('PAIR_CONTINUITY','OPEN_SIDE','IDENTITY_LATCH','ROTATION_SEARCH','FULL_RATE','FILTER_GUARD','OUTPUT_RECOVERY','JOINT_IDENTITY'):envbase['FRAME_HANDS_'+key]='1'
 envbase.update(FRAME_HANDS_PALM_THRESHOLD='0.2',FRAME_JOB_DEVICE_ROOT=str(source.resolve()/'device'),FRAME_HANDS_RIG_DIR=str(source.resolve()/'profiles'),FRAME_HANDS_RIG_WHOLE_POSE='1',FRAME_HANDS_RIG_REPLAY_HEADS=str(a.heads.resolve()))
 results=[]
 for index,variant in enumerate(('baseline','candidate','candidate','baseline')):
  name=str(index)+'-'+variant;folder=(a.output/name).resolve();folder.mkdir();env=envbase.copy();env.update(FRAME_HANDS_AUDIT=str(folder/'views.txt'),FRAME_HANDS_RIG_TRACE=str(folder/'rig.txt'))
  if variant=='candidate':env['FRAME_HANDS_RIG_FINGER_CORRESPONDENCE']='1'
  cmd=[str(exe),str(a.recording.resolve()),'--threads','3','--models',str(source.resolve()/'models/ncnn'),'--to',str(reference['seconds']),'--all-sets','--poses',str(folder/'poses.txt'),'--step-times',str(folder/'times.txt')]
  start=time.monotonic();run=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=300);(folder/'stdout.txt').write_text(run.stdout);(folder/'stderr.txt').write_text(run.stderr);assert run.returncode==0,run.stderr
  for filename in ('views','poses'):assert sha(folder/(filename+'.txt'))==reference['results'][1][filename+'_sha256']
  if variant=='baseline':assert sha(folder/'rig.txt')==reference['results'][1]['rig_sha256']
  times=[float(line.split()[1]) for line in (folder/'times.txt').read_text().splitlines()];row=dict(name=name,variant=variant,processed_frames=len(times),step_ms=stats(times),wall_seconds=time.monotonic()-start,**{n+'_sha256':sha(folder/(n+'.txt')) for n in ('views','poses','rig')},**recorded_trace_stats(folder/'rig.txt'));results.append(row);print(json.dumps(row),flush=True)
 repeat=results[0]['rig_sha256']==results[3]['rig_sha256'] and results[1]['rig_sha256']==results[2]['rig_sha256'];assert repeat
 report=dict(scope=__doc__,comparison='whole_pose_vs_finger_correspondence',original_detector_outputs_identical=True,rig_outputs_repeat_exactly=repeat,results=results,seconds=reference['seconds'],environment=reference['environment'],source_sha256={str(f.relative_to(source)):sha(f) for f in source.rglob('*') if f.is_file()},reference_sha256=sha(a.reference/'result.json'),runner_sha256=sha(Path(__file__)),executable_sha256=sha(exe),build=command,limitations=['Host CPU experiment; not Frame runtime or wearer validation.','Index/middle chain may swap in one camera only when original normalized paired-ray RMS exceeds3 and corrected RMS is at most3.','Complete fit alternatives share angle prior and start from selected whole pose. Selected correction feeds whole-pose history, not detector/identity/crop history.','Different correspondence objectives cannot certify unmodified-pixel accuracy or true finger identity.','No both-camera permutation, learned alternative model, presence extension, injected input or accepted source change.'])
 (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print('correspondence ABBA passed',flush=True)
if __name__=='__main__':main()
