"""Offline causal stereo-sharing policy probe; no inference or runtime change."""
import argparse,collections,io,json,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE.parent/'exploratory'))
from frame_data import load_calibration,read_audit
from temporal_association import TemporalAssociation
from replay_mesh_trial import packets
from run_mesh_trial import pose_matrix
from stereo_identity_plan import StereoIdentityPlan,shared_score


def shape(debug,track=False):
    return sorted((r['side'],r['track'] if track else 0,tuple(sorted((v['camera'],v['source_id']) for v in r['views']))) for r in debug.get('assignments',[]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('session','evidence','output','upstream'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(a.upstream))
    refs={}
    for line in (a.session/'predictions.jsonl').open():
        r=json.loads(line);refs[r['capture_raw_ns']]=r['capture_pose']
    evs={r['stamp']:r for line in a.evidence.open() if (r:=json.loads(line))}
    cameras=load_calibration(a.session/'calibration');plan=StereoIdentityPlan(cameras)
    candidate=TemporalAssociation(cameras);control=TemporalAssociation(cameras)
    reasons=collections.Counter();differences=[];control_diffs=[];frames=original=scheduled=shared=omitted_conflicts=0
    with (a.output/'trace.jsonl').open('w') as out:
        for meta,blobs in packets(a.session/'transport.bin'):
            if meta.get('waiting'):continue
            pose=refs.get(meta['stamp']);ev=evs.get(meta['stamp'])
            if pose is None or ev is None or not pose['valid']:
                candidate.reset();control.reset();continue
            obs=next(iter(read_audit(io.StringIO(meta['audit'])).values()));world=pose_matrix(pose['capture'])
            evidence={(r['camera'],r['source_id']):r for r in ev['views']}
            prepared={(v['camera'],v['id']) for v in meta['crops']}
            views=[v for v in obs['views'] if (v['camera'],v['id']) in prepared and v['presence']>=.5]
            if {(v['camera'],v['id']) for v in views}!=set(evidence):raise ValueError('Evidence/input coverage mismatch')
            observation=dict(obs,views=views)
            groups,reason=plan.groups(observation,world);reasons[reason]+=1
            reused={};pairs=[]
            for primary,secondary,error in groups:
                pk=(primary['camera'],primary['id']);sk=(secondary['camera'],secondary['id'])
                q,reason=shared_score(primary,secondary,evidence[pk]);reasons[reason]+=1
                if q is None:continue
                reused[sk]=q
                reference=evidence[sk]
                conflict=reference['agreement'] and max(reference['right'],1-reference['right'])>=.85 and (q>=.5)!=(reference['right']>=.5)
                omitted_conflicts+=conflict
                pairs.append(dict(primary=pk,secondary=sk,geometric_error=error,primary_score=q,withheld_secondary_score=reference['right'],withheld_confident_conflict=bool(conflict)))
            fresh=dict(observation,views=[dict(v,right=evidence[v['camera'],v['id']]['right']) for v in views])
            selected=dict(observation,views=[dict(v,right=reused.get((v['camera'],v['id']),evidence[v['camera'],v['id']]['right'])) for v in views])
            control.assign(fresh,world);candidate.assign(selected,world)
            baseline=shape(control.last_debug);actual=shape(candidate.last_debug)
            if shape(control.last_debug,True)!=shape(ev.get('association',{}),True):control_diffs.append(meta['stamp'])
            if actual!=baseline:differences.append(dict(stamp=meta['stamp'],candidate=actual,baseline=baseline))
            # A skipped secondary evidence pass still needs one encoder pass if
            # selected by pose, rather than the two hypotheses used by control.
            used={(v['camera'],v['source_id']) for row in candidate.last_debug.get('assignments',[]) for v in row['views']}
            original+=2*len(views);scheduled+=2*(len(views)-len(reused))+len(set(reused)&used);shared+=len(reused);frames+=1
            out.write(json.dumps(dict(stamp=meta['stamp'],pairs=pairs,candidate=actual,baseline=baseline))+'\n')
    result=dict(scope='Same-frame sharing with causal association recomputed. Encoder pass counts include missing selected features; no neural speed or pose accuracy measured.',
        session=a.session.name,frames=frames,reasons=dict(reasons),shared_secondary_views=shared,withheld_confident_conflicts=omitted_conflicts,
        encoder_passes_original=original,encoder_passes_estimated=scheduled,estimated_pass_saving_fraction=1-scheduled/original if original else 0,
        control_assignment_mismatch_count=len(control_diffs),control_mismatch_stamps=control_diffs,candidate_assignment_difference_count=len(differences),candidate_assignment_differences=differences,
        policy=dict(max_geometric_error=.015,unique_pair_margin=.03,min_presence=.8,min_primary_image_confidence=.85,min_detector_confidence=.8,expanded_overlap_pixels=8))
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('candidate_assignment_differences','control_mismatch_stamps')},indent=2))


if __name__=='__main__':main()
