"""Private full-neural continuity comparison, optionally testing fresh evidence renewal."""
import argparse,json,os,shutil,subprocess,time
from pathlib import Path
from run_tracker_rig_check import sha,stats,recorded_trace_stats
from run_host_tracker_check import BUNDLE_SHA

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('bundle','reference','strict-reference','recording','heads','output'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--renewal-reference',type=Path)
 a=p.parse_args();renewal_reference=json.loads((a.renewal_reference/'result.json').read_text()) if a.renewal_reference else None;strict_reference=json.loads((a.strict_reference/'result.json').read_text());reference=json.loads((a.reference/'result.json').read_text())
 assert reference['environment']['machine']=='x86_64' and sha(a.bundle/'manifest.json')==BUNDLE_SHA
 for name,digest in json.loads((a.bundle/'manifest.json').read_text())['sha256'].items():assert sha(a.bundle/name)==digest
 assert sha(a.recording/'sets.bin')==reference['provenance']['recording_sha256'];assert sha(a.heads)==reference['provenance']['head_evidence']['heads_sha256']
 a.output.mkdir(parents=True,exist_ok=False);source=a.output/'source';shutil.copytree(a.bundle,source);
 for filename in ('stereo_disagreement.h','finger_correspondence.h','correspondence_continuity.h'):shutil.copy2(Path(__file__).with_name(filename),source/'native'/filename)
 path=source/'native/whole_pose_fitter.h';text=path.read_text();text=text.replace('#include "whole_pose_seed.h"','#include "whole_pose_seed.h"\n#include "correspondence_continuity.h"\n#include <cstdlib>')
 text=text.replace('struct WholePoseResult {','struct WholePoseResult {CorrespondenceResult correspondence{};CorrespondenceContinuation continuation{};')
 text=text.replace('    std::array<History,2> history_{};','    std::array<History,2> history_{};std::array<CorrespondenceMemory,2> correspondence_memory_{};')
 text=text.replace('history_[side]={};}', 'history_[side]={};correspondence_memory_[side].clear();}')
 marker='        // Mutate only after validating/fitting.'
 assert text.count(marker)==1
 insert='        if(const char* flag=std::getenv("FRAME_HANDS_RIG_FINGER_CORRESPONDENCE");flag&&std::string(flag)=="1"&&fresh_valid){out.correspondence=refine_index_middle_hypotheses(rig,out.fit.pose,out.fit,cameras,pixels,fresh.angles);if(std::getenv("FRAME_HANDS_RIG_CORRESPONDENCE_CONTINUITY")){correspondence_memory_[side].set_relative_renewal(bool(std::getenv("FRAME_HANDS_RIG_CORRESPONDENCE_RENEWAL")));out.continuation=continue_correspondence(correspondence_memory_[side],rig,out.fit.pose,out.correspondence,cameras,pixels,fresh.angles,stamp,source,bool(capture_head)&&continuous);out.correspondence=out.continuation.result;}out.fit=out.correspondence.fit;}\n'
 text=text.replace(marker,insert+marker);path.write_text(text)
 path=source/'native/tracker_rig_adapter.h';text=path.read_text();text=text.replace('RigStageResult result{};','CorrespondenceResult correspondence{};CorrespondenceContinuation continuation{};RigStageResult result{};');text=text.replace('output.whole_alternate_cost=full.alternate_cost;','output.whole_alternate_cost=full.alternate_cost;output.correspondence=full.correspondence;output.continuation=full.continuation;');path.write_text(text)
 path=source/'native/tracker_rig_hook.h';text=path.read_text();marker='            if(head_source_||recorded_heads_)trace_<<"whole "'
 assert text.count(marker)==1
 insert='            if(o.result.initialized&&std::getenv("FRAME_HANDS_RIG_FINGER_CORRESPONDENCE")){const auto& c=o.correspondence;trace_<<"correspondence "<<o.source<<\' \'<<c.chosen<<\' \'<<c.original_disagreement;for(double v:c.alternative_disagreement)trace_<<\' \'<<v;for(double v:c.alternative_cost)trace_<<\' \'<<v;trace_<<\'\\n\';}\n            if(o.result.initialized&&std::getenv("FRAME_HANDS_RIG_CORRESPONDENCE_CONTINUITY")){trace_<<"continuity "<<o.source<<\' \'<<o.continuation.continued<<\' \'<<o.continuation.evidence_age<<\' \'<<o.continuation.strong_stamp<<\'\\n\';}\n            if(o.result.initialized&&std::getenv("FRAME_HANDS_RIG_CORRESPONDENCE_RENEWAL")){trace_<<"renewal "<<o.source<<\' \'<<o.continuation.renewed<<\'\\n\';}\n'
 text=text.replace(marker,insert+marker);path.write_text(text)
 exe=(a.output/'replay').resolve();command=[s.replace(str(a.bundle.resolve()),str(source.resolve())) for s in reference['provenance']['build']];command[-1]=str(exe)
 build=subprocess.run(command,capture_output=True,text=True,timeout=180);(a.output/'build.stderr').write_text(build.stderr);assert build.returncode==0,build.stderr;print('correspondence replay linked',flush=True)
 envbase={k:v for k,v in os.environ.items() if not k.startswith('FRAME_HANDS_')}
 for key in ('PAIR_CONTINUITY','OPEN_SIDE','IDENTITY_LATCH','ROTATION_SEARCH','FULL_RATE','FILTER_GUARD','OUTPUT_RECOVERY','JOINT_IDENTITY'):envbase['FRAME_HANDS_'+key]='1'
 envbase['FRAME_HANDS_RIG_FINGER_CORRESPONDENCE']='1'
 if renewal_reference:envbase['FRAME_HANDS_RIG_CORRESPONDENCE_CONTINUITY']='1'
 envbase.update(FRAME_HANDS_PALM_THRESHOLD='0.2',FRAME_JOB_DEVICE_ROOT=str(source.resolve()/'device'),FRAME_HANDS_RIG_DIR=str(source.resolve()/'profiles'),FRAME_HANDS_RIG_WHOLE_POSE='1',FRAME_HANDS_RIG_REPLAY_HEADS=str(a.heads.resolve()))
 results=[]
 for index,variant in enumerate(('baseline','candidate','candidate','baseline')):
  name=str(index)+'-'+variant;folder=(a.output/name).resolve();folder.mkdir();env=envbase.copy();env.update(FRAME_HANDS_AUDIT=str(folder/'views.txt'),FRAME_HANDS_RIG_TRACE=str(folder/'rig.txt'))
  if variant=='candidate':env['FRAME_HANDS_RIG_CORRESPONDENCE_RENEWAL' if renewal_reference else 'FRAME_HANDS_RIG_CORRESPONDENCE_CONTINUITY']='1'
  cmd=[str(exe),str(a.recording.resolve()),'--threads','3','--models',str(source.resolve()/'models/ncnn'),'--to',str(reference['seconds']),'--all-sets','--poses',str(folder/'poses.txt'),'--step-times',str(folder/'times.txt')]
  start=time.monotonic();run=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=300);(folder/'stdout.txt').write_text(run.stdout);(folder/'stderr.txt').write_text(run.stderr);assert run.returncode==0,run.stderr
  for filename in ('views','poses'):assert sha(folder/(filename+'.txt'))==reference['results'][1][filename+'_sha256']
  if variant=='baseline':assert sha(folder/'rig.txt')==(renewal_reference or strict_reference)['results'][1]['rig_sha256']
  times=[float(line.split()[1]) for line in (folder/'times.txt').read_text().splitlines()];row=dict(name=name,variant=variant,processed_frames=len(times),step_ms=stats(times),wall_seconds=time.monotonic()-start,**{n+'_sha256':sha(folder/(n+'.txt')) for n in ('views','poses','rig')},**recorded_trace_stats(folder/'rig.txt'));results.append(row);print(json.dumps(row),flush=True)
 repeat=results[0]['rig_sha256']==results[3]['rig_sha256'] and results[1]['rig_sha256']==results[2]['rig_sha256'];assert repeat
 report=dict(scope=__doc__,comparison='correspondence_continuity_vs_renewal' if renewal_reference else 'finger_correspondence_vs_continuity',original_detector_outputs_identical=True,rig_outputs_repeat_exactly=repeat,results=results,seconds=reference['seconds'],environment=reference['environment'],source_sha256={str(f.relative_to(source)):sha(f) for f in source.rglob('*') if f.is_file()},reference_sha256=sha(a.reference/'result.json'),strict_reference_sha256=sha(a.strict_reference/'result.json'),renewal_reference_sha256=sha(a.renewal_reference/'result.json') if renewal_reference else None,runner_sha256=sha(Path(__file__)),executable_sha256=sha(exe),build=command,limitations=['Host CPU experiment; not Frame runtime or wearer validation.','Strict admission remains original disagreement>3 and alternative<=3. Previously selected camera may continue up to100ms from latest qualifying evidence only while original>3 and alternative is finite, above3 but better than original, with better current fit cost.','Fresh fit each frame, shared angle prior, no pose holding. With renewal enabled, an established correction renews its100ms evidence deadline only when current original-minus-alternative normalized disagreement is at least3 and the fitted alternative wins. Other weak frames never renew. Head/source/side/missing resets remain enforced.','Different correspondence objectives cannot certify unmodified-pixel accuracy or true finger identity.','No both-camera permutation, learned alternative model, presence extension, injected input or accepted source change.'])
 (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print('correspondence ABBA passed',flush=True)
if __name__=='__main__':main()
