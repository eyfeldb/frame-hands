"""Causal early-hypothesis shadow; never consult skipped output to gate exit."""
import argparse,collections,io,json,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE.parent/'exploratory'))
from frame_data import load_calibration,read_audit
from temporal_association import TemporalAssociation
from replay_mesh_trial import packets
from run_mesh_trial import pose_matrix
from shadow_stereo_identity import shape
from early_hypothesis import early_score,first_side,MAX_REPROJECTION_PX,MAX_RAW_SIGMA,MIN_DETECTOR_CONFIDENCE,MIN_PRESENCE
from distill_encoder import sha
from stereo_exit import needs_completion


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('session','evidence','upstream','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--require-stereo-exit',action='store_true');a=p.parse_args()
    if '143912Z' in str(a.session):raise ValueError('Latest holdout excluded')
    a.output.mkdir(exist_ok=False);sys.path.insert(0,str(a.upstream))
    refs={r['capture_raw_ns']:r['capture_pose'] for line in (a.session/'predictions.jsonl').open() if (r:=json.loads(line))}
    rows={r['stamp']:r for line in a.evidence.open() if (r:=json.loads(line))}
    cameras=load_calibration(a.session/'calibration');candidate=TemporalAssociation(cameras);control=TemporalAssociation(cameras)
    frames=views_seen=exits=original=estimated=opposite=abstained=strong_opposite=late_alternative=control_diffs=0
    changed=[];availability=[];fit=[];completed_views=completed_frames=0
    with (a.output/'trace.jsonl').open('w') as out:
        for meta,_ in packets(a.session/'transport.bin'):
            if meta.get('waiting'):continue
            pose=refs.get(meta['stamp']);row=rows.get(meta['stamp'])
            if pose is None or row is None or not pose['valid']:candidate.reset();control.reset();continue
            observation=next(iter(read_audit(io.StringIO(meta['audit'])).values()));world=pose_matrix(pose['capture'])
            prepared={(v['camera'],v['id']) for v in meta['crops']};views=[v for v in observation['views'] if (v['camera'],v['id']) in prepared and v['presence']>=.5]
            full={(v['camera'],v['source_id']):v for v in row['views']}
            if {(v['camera'],v['id']) for v in views}!=set(full):raise ValueError('Evidence/input mismatch')
            skipped={};audit=[]
            for view in views:
                key=(view['camera'],view['id']);reference=full[key];raw=float(view['right']);side=first_side(raw)
                # This gate sees ONE model hypothesis only. The remaining fields
                # below are audit labels; they cannot change the decision.
                error=reference['reprojection_px'][side];sigma=reference['mean_raw_sigmas'][side]
                score=early_score(raw,view['presence'],error,sigma)
                if score is None:continue
                skipped[key]=score
                audit.append(dict(camera=key[0],source_id=key[1],chosen_side=side,first_error=error,first_sigma=sigma,score=score,withheld_full_score=reference['right'],withheld_agreement=reference['agreement']))
            provisional=dict(observation,views=[dict(v,right=skipped[v['camera'],v['id']] if (v['camera'],v['id']) in skipped else full[v['camera'],v['id']]['right']) for v in views])
            complete=bool(skipped) and a.require_stereo_exit and needs_completion(candidate,provisional,world)
            if complete:
                completed_frames+=1;completed_views+=len(skipped);skipped={}
            for item in audit:
                item['completed']=complete
                if complete:continue
                conflict=item['withheld_agreement'] and (item['withheld_full_score']>=.5)!=(item['score']>=.5)
                opposite+=conflict;strong_opposite+=conflict and max(item['withheld_full_score'],1-item['withheld_full_score'])>=.85
                abstained+=not item['withheld_agreement']
            exits+=len(skipped)
            control.assign(dict(observation,views=[dict(v,right=full[v['camera'],v['id']]['right']) for v in views]),world)
            candidate.assign(dict(observation,views=[dict(v,right=skipped.get((v['camera'],v['id']),full[v['camera'],v['id']]['right'])) for v in views]),world)
            old,new=shape(control.last_debug),shape(candidate.last_debug)
            if shape(control.last_debug,True)!=shape(row.get('association',{}),True):control_diffs+=1
            if old!=new:changed.append(dict(stamp=meta['stamp'],control=old,candidate=new))
            if {r[0] for r in old}!={r[0] for r in new}:availability.append(meta['stamp'])
            extra=sum((v['camera'],v['source_id']) in skipped and row['side']!=int(skipped[v['camera'],v['source_id']]>=.5) for row in candidate.last_debug.get('assignments',[]) for v in row['views'])
            # The first-hypothesis feature is cached. If final association wants
            # the opposite side, its encoder still runs: never count it as saved.
            late_alternative+=extra;original+=2*len(views);estimated+=2*len(views)-len(skipped)+extra;views_seen+=len(views);frames+=1
            out.write(json.dumps(dict(stamp=meta['stamp'],early=audit,late_alternative=extra,control=old,candidate=new))+'\n')
    result=dict(scope='First-hypothesis causal association shadow only; no changed neural execution, timing or physical truth.',policy=dict(require_stereo_exit=a.require_stereo_exit,max_reprojection_px=MAX_REPROJECTION_PX,max_raw_sigma=MAX_RAW_SIGMA,min_detector_confidence=MIN_DETECTOR_CONFIDENCE,min_presence=MIN_PRESENCE,exit_side_score=.9),session=a.session.name,frames=frames,views=views_seen,early_exit_views=exits,completed_views=completed_views,completed_frames=completed_frames,withheld_agreed_opposite=opposite,withheld_strong_opposite=strong_opposite,withheld_abstentions=abstained,late_opposite_pose_passes=late_alternative,encoder_passes_original=original,encoder_passes_estimated=estimated,estimated_pass_saving_fraction=1-estimated/original,control_mismatch_count=control_diffs,assignment_difference_count=len(changed),availability_difference_count=len(availability),assignment_differences=changed,availability_difference_stamps=availability,evidence_sha256=sha(a.evidence),source_sha256={name:sha(BASE/name) for name in ('early_hypothesis.py','shadow_early_hypothesis.py','stereo_exit.py')})
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('assignment_differences','availability_difference_stamps')},indent=2))


if __name__=='__main__':main()
