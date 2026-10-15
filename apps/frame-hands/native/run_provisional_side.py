"""Private full-neural provisional-side ABBA, preserving the original joint-recovery trigger."""
import argparse,json,os,platform,shutil,subprocess,time
from pathlib import Path
from run_tracker_rig_check import sha,stats,verify_head_evidence,recorded_trace_stats
FROZEN_SHA='a8cc3db463b00e1cefd3888591db9c304322478141f6caa70d18cea579a212ae'
RECORDINGS=['34eb755f1f5285cdf0c94c6d45250ccb3739fe2409abbb958485ea13ef40d3d4','5833dcb93dc43bd85413915ba6192d55102f8c896d67971db28b4667e55e91d4']

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('frozen','second-reference','first-session','second-session','first-heads','second-heads','output'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument("--stereo-first",action="store_true")
 a=p.parse_args()
 if platform.system()!='Linux' or platform.machine()!='x86_64':raise ValueError('Host Linux CPU only')
 if a.output.exists():raise FileExistsError(a.output)
 assert sha(a.frozen/'result.json')==FROZEN_SHA
 reference=json.loads((a.frozen/'result.json').read_text());original=(a.frozen/'source').resolve()
 for name,digest in reference['source_sha256'].items():assert sha(original/name)==digest
 a.output.mkdir(parents=True);source=(a.output/'source').resolve();shutil.copytree(original,source);shutil.copy2(Path('apps/frame-hands/native/provisional_side.h'),source/'native/provisional_side.h')
 path=source/'track/tracker.h';s=path.read_text();s=s.replace('#include "joint_identity.h"','#include "joint_identity.h"\n#include "../native/provisional_side.h"')
 s=s.replace('    HandIdentity identity;', '    fhn::ProvisionalSide provisional;\n    bool provisional_enabled=false;int provisional_vote=-1,provisional_strong=-1;\n    HandIdentity identity;')
 before='bool right() const { return right_score > 0.5; }';assert s.count(before)==1
 s=s.replace(before,'bool right() const { return provisional_enabled&&provisional.active ? (identity.latched?identity.right:provisional.right) : right_score>0.5; }');path.write_text(s)
 path=source/'track/tracker.cpp';s=path.read_text();s='#include <fstream>\n#include <iomanip>\n'+s
 before='    if(identity_latch && hand.identity.latched)\n        hand.right_score=hand.identity.right ? 0.99 : 0.01;\n    return true;';assert s.count(before)==1
 after=before.replace('    return true;','''    int provisional_vote=-1;bool provisional_reliable=views.size()>=2;
    for(const View* v:views){int side=v->lm.right<.35?0:(v->lm.right>.65?1:-1);
        if(!v->fresh||v->lm.presence<.8||side<0||(provisional_vote>=0&&side!=provisional_vote))provisional_reliable=false;
        if(side>=0)provisional_vote=side;}
    hand.provisional_vote=provisional_reliable?provisional_vote:-1;
    hand.provisional_strong=identity_latch&&hand.identity.latched?int(hand.identity.right):-1;
    const char* provisional_flag=std::getenv("FRAME_HANDS_PROVISIONAL_SIDE");
    hand.provisional_enabled=provisional_flag&&std::string(provisional_flag)=="1";
    const char* stereo_flag=std::getenv("FRAME_HANDS_PROVISIONAL_STEREO_FIRST");
    hand.provisional.stereo_first=stereo_flag&&std::string(stereo_flag)=="1";
    if(hand.provisional_enabled)hand.provisional.observe(hand.right_score,hand.provisional_vote,t_ns,hand.provisional_strong);
    else hand.provisional.clear();
    return true;''');s=s.replace(before,after)
 trigger='out[0]->right()==out[1]->right()';assert s.count(trigger)==1
 s=s.replace(trigger,'(out[0]->right_score>.5)==(out[1]->right_score>.5)')
 marker='    // Fit current unfiltered evidence after final source-side recovery.';assert s.count(marker)==1
 diagnostic='''    if(const char* path=std::getenv("FRAME_HANDS_SIDE_DIAGNOSTIC")){
        static std::ofstream side_out(path);if(!side_out)throw std::runtime_error("Cannot write side trace");
        side_out<<std::setprecision(17)<<"frame "<<t_ns<<' '<<out.size()<<'\\n';
        for(const Hand* h:out)side_out<<"side "<<h->id<<' '<<h->right_score<<' '<<h->right()<<' '
            <<h->identity.latched<<' '<<h->identity.right<<' '<<h->identity.candidate<<' '<<h->identity.streak<<' '<<h->identity.stereo_streak<<' '
            <<h->provisional_enabled<<' '<<h->provisional.active<<' '<<h->provisional.right<<' '<<h->provisional.last<<' '<<h->provisional_vote<<' '<<h->provisional_strong<<' '<<h->nviews<<' '<<h->provisional.stereo_first<<' '<<h->provisional.stereo_supported<<'\\n';
    }
'''
 s=s.replace(marker,diagnostic+marker);path.write_text(s)
 exe=(a.output/'replay').resolve();build=[v.replace(str(original),str(source)) for v in reference['build']];build[-1]=str(exe)
 linked=subprocess.run(build,capture_output=True,text=True,timeout=180);(a.output/'build.stderr').write_text(linked.stderr);assert linked.returncode==0,linked.stderr;print('provisional replay linked',flush=True)
 envbase={k:v for k,v in os.environ.items() if not k.startswith('FRAME_HANDS_')}
 for key in ('PAIR_CONTINUITY','OPEN_SIDE','IDENTITY_LATCH','ROTATION_SEARCH','FULL_RATE','FILTER_GUARD','OUTPUT_RECOVERY','JOINT_IDENTITY','RIG_WHOLE_POSE','RIG_FINGER_CORRESPONDENCE','RIG_CORRESPONDENCE_CONTINUITY','RIG_CORRESPONDENCE_RENEWAL'):envbase['FRAME_HANDS_'+key]='1'
 envbase.update(FRAME_HANDS_PALM_THRESHOLD='0.2',FRAME_JOB_DEVICE_ROOT=str(source/'device'),FRAME_HANDS_RIG_DIR=str(source/'profiles'))
 for number,(label,session,heads,refpath) in enumerate([('first',a.first_session,a.first_heads,a.frozen),('second',a.second_session,a.second_heads,a.second_reference)]):
  if '143912Z' in str(session):raise ValueError('Accepted holdout excluded')
  assert sha(session/'frame/sets.bin')==RECORDINGS[number];head=verify_head_evidence(heads,RECORDINGS[number]);old=json.loads((refpath/'result.json').read_text())
  for name in ('device_config.json','xrservice.json'):assert sha(session/'frame/snapshot/device/persist'/name)==sha(source/'device/persist'/name)
  folder=a.output/label;folder.mkdir();results=[]
  for index,variant in enumerate(('baseline','candidate','candidate','baseline')):
   name=str(index)+'-'+variant;dest=(folder/name).resolve();dest.mkdir();env=envbase.copy()
   env.update(FRAME_HANDS_RIG_REPLAY_HEADS=str((heads/'heads.txt').resolve()),FRAME_HANDS_AUDIT=str(dest/'views.txt'),FRAME_HANDS_RIG_TRACE=str(dest/'rig.txt'),FRAME_HANDS_SIDE_DIAGNOSTIC=str(dest/'side.txt'))
   if variant=='candidate':
    env['FRAME_HANDS_PROVISIONAL_SIDE']='1'
    if a.stereo_first:env['FRAME_HANDS_PROVISIONAL_STEREO_FIRST']='1'
   command=[str(exe),str((session/'frame').resolve()),'--threads','3','--models',str(source/'models/ncnn'),'--to','60','--all-sets','--poses',str(dest/'poses.txt'),'--step-times',str(dest/'times.txt')]
   start=time.monotonic();run=subprocess.run(command,env=env,capture_output=True,text=True,timeout=300)
   for stream in ('stdout','stderr'):(dest/(stream+'.txt')).write_text(getattr(run,stream))
   assert run.returncode==0,run.stderr;times=[float(x.split()[1]) for x in (dest/'times.txt').read_text().splitlines()];assert len(times)==1800
   r=dict(name=name,variant=variant,processed_frames=len(times),step_ms=stats(times),wall_seconds=time.monotonic()-start,**{n+'_sha256':sha(dest/(n+'.txt')) for n in ('views','poses','rig','side')},**recorded_trace_stats(dest/'rig.txt'));results.append(r)
   assert r['views_sha256']==old['results'][1]['views_sha256'],label+' neural/identity state changed'
   if variant=='baseline':
    for n in ('views','poses','rig'):assert r[n+'_sha256']==old['results'][1][n+'_sha256'],label+' baseline '+n
   print(label,json.dumps(r),flush=True)
  repeat=all(results[0][n+'_sha256']==results[3][n+'_sha256'] and results[1][n+'_sha256']==results[2][n+'_sha256'] for n in ('views','poses','rig','side'));assert repeat
  report=dict(scope=__doc__,comparison='provisional_side_hysteresis',side_policy='stereo_first' if a.stereo_first else 'score_margin',seconds=60,results=results,rig_outputs_repeat_exactly=repeat,baseline_exact=True,neural_views_exact=True,provenance=dict(recording_sha256=RECORDINGS[number],head_evidence=head,reference_sha256=sha(refpath/'result.json'),executable_sha256=sha(exe),runner_sha256=sha(Path(__file__)),build=build),environment=dict(machine=platform.machine(),kernel=platform.release()),limitations=['Only publication/rig side changes. Original joint recovery uses raw score sides; complete view audit remains exact.','Host CPU, no native Frame timing, XR, camera or wearer acceptance.','No change to raw model scores, thresholds, geometry, associations, presence or output holds by this component.','Correct initial/correction handedness requires physical evidence; fewer switches alone is insufficient.'])
  (folder/'result.json').write_text(json.dumps(report,indent=2)+'\n')
 (a.output/'manifest.json').write_text(json.dumps(dict(source_sha256={str(f.relative_to(source)):sha(f) for f in source.rglob('*') if f.is_file()},build=build,executable_sha256=sha(exe),runner_sha256=sha(Path(__file__))),indent=2)+'\n');print('Two-recording provisional ABBA passed',flush=True)
if __name__=='__main__':main()
