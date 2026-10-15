"""Isolate exact live-path landmark crops during saved-image host replay.

Adds observation-only trace statements to a private copy of frozen tracker-rig-v9.
Requires output parity with its previous host candidate before interpreting crops.
"""
import argparse,json,os,shutil,subprocess
from pathlib import Path
from run_tracker_rig_check import sha
from run_host_tracker_check import BUNDLE_SHA

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('bundle','reference','recording','heads','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args()
 result=json.loads((a.reference/'result.json').read_text())
 assert result['environment']['machine']=='x86_64' and result['rig_outputs_repeat_exactly'] and result['original_detector_outputs_identical']
 assert sha(a.bundle/'manifest.json')==BUNDLE_SHA
 for name,digest in json.loads((a.bundle/'manifest.json').read_text())['sha256'].items():assert sha(a.bundle/name)==digest
 assert sha(a.recording/'sets.bin')==result['provenance']['recording_sha256']
 assert sha(a.heads)==result['provenance']['head_evidence']['heads_sha256']
 a.output.mkdir(parents=True,exist_ok=False);src=a.output/'source';shutil.copytree(a.bundle,src)
 path=src/'track/tracker.cpp';text=path.read_text();original=text
 helper='''\n#include <fstream>\n#include <iomanip>\nnamespace {\nstd::ofstream& landmark_diagnostic(){static std::ofstream out(std::getenv("FRAME_HANDS_LANDMARK_DIAGNOSTIC"));if(!out)throw std::runtime_error("Cannot write landmark diagnostic");out<<std::setprecision(17);return out;}\n}\n'''
 text=helper+text
 before='    pool_.run(jobs);\n    stats.hand_calls += int(views.size());'
 after='''    pool_.run(jobs);
    if(std::getenv("FRAME_HANDS_LANDMARK_DIAGNOSTIC"))for(const View* v:views){auto& out=landmark_diagnostic();out<<"crop "<<v->cam->name<<' '<<v->hand<<' '<<v->roi.center[0]<<' '<<v->roi.center[1]<<' '<<v->roi.size<<' '<<v->roi.rotation<<' '<<v->lm.presence<<' '<<v->lm.right;for(const auto& p:v->lm.pts)out<<' '<<p[0]<<' '<<p[1];out<<'\\n';}
    stats.hand_calls += int(views.size());'''
 assert text.count(before)==1;text=text.replace(before,after)
 before='    const auto t_step = std::chrono::steady_clock::now();'
 assert text.count(before)==1;text=text.replace(before,before+'\n    if(std::getenv("FRAME_HANDS_LANDMARK_DIAGNOSTIC"))landmark_diagnostic()<<"set "<<t_ns<<\'\\n\';')
 path.write_text(text);exe=(a.output/'replay').resolve()
 oldroot=str(a.bundle.resolve());command=[v.replace(oldroot,str(src.resolve())) for v in result['provenance']['build']];command[-1]=str(exe)
 build=subprocess.run(command,capture_output=True,text=True,timeout=180)
 (a.output/'build.stderr').write_text(build.stderr);assert build.returncode==0,build.stderr
 print('diagnostic linked',flush=True)
 env={k:v for k,v in os.environ.items() if not k.startswith('FRAME_HANDS_')}
 for key in ('PAIR_CONTINUITY','OPEN_SIDE','IDENTITY_LATCH','ROTATION_SEARCH','FULL_RATE','FILTER_GUARD','OUTPUT_RECOVERY','JOINT_IDENTITY'):env['FRAME_HANDS_'+key]='1'
 env.update(FRAME_HANDS_PALM_THRESHOLD='0.2',FRAME_JOB_DEVICE_ROOT=str(src.resolve()/'device'),FRAME_HANDS_AUDIT=str(a.output.resolve()/'views.txt'),FRAME_HANDS_RIG_DIR=str(src.resolve()/'profiles'),FRAME_HANDS_RIG_TRACE=str(a.output.resolve()/'rig.txt'),FRAME_HANDS_RIG_WHOLE_POSE='1',FRAME_HANDS_RIG_REPLAY_HEADS=str(a.heads.resolve()),FRAME_HANDS_LANDMARK_DIAGNOSTIC=str(a.output.resolve()/'crops.txt'))
 command_run=[str(exe),str(a.recording.resolve()),'--threads','3','--models',str(src.resolve()/'models/ncnn'),'--to',str(result['seconds']),'--all-sets','--poses',str(a.output.resolve()/'poses.txt')]
 run=subprocess.run(command_run,env=env,capture_output=True,text=True,timeout=300);(a.output/'stdout.txt').write_text(run.stdout);(a.output/'stderr.txt').write_text(run.stderr);assert run.returncode==0
 for name in ('views','rig','poses'):assert sha(a.output/(name+'.txt'))==result['results'][1][name+'_sha256'],name+' changed'
 evidence=dict(scope=__doc__,seconds=result['seconds'],reference_sha256=sha(a.reference/'result.json'),bundle_sha256=sha(a.bundle/'manifest.json'),patched_source_sha256=sha(path),original_source_sha256=sha(a.bundle/'track/tracker.cpp'),runner_sha256=sha(Path(__file__)),executable_sha256=sha(exe),trace_sha256=sha(a.output/'crops.txt'),outputs_match=True,command=command,run_command=command_run)
 (a.output/'result.json').write_text(json.dumps(evidence,indent=2)+'\n');print('Exact views, rig and poses reproduced with diagnostic enabled.',flush=True)
if __name__=='__main__':main()
