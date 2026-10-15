"""Causal shadow of target-conditioned stereo identity shortcuts; no runtime edits."""
import argparse,collections,io,json,sys
from pathlib import Path
import numpy as np,torch
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE.parent/'exploratory'))
from frame_data import load_calibration,read_audit
from temporal_association import TemporalAssociation
from replay_mesh_trial import packets
from run_mesh_trial import pose_matrix
from train_chirality import local_images,predict
from target_chirality import condition,load_target
from stereo_identity_plan import StereoIdentityPlan
from shadow_stereo_identity import shape
from distill_encoder import sha


def shortcut(left,right,scores):
    qs=[]
    for v in (left,right):
        q,m=scores[v['camera'],v['id']]
        raw=float(v['right'])
        if not np.isfinite([q,m,raw]).all() or not all(0<=v<=1 for v in (q,m,raw)):return None,'invalid'
        if min(max(q,1-q),max(m,1-m))<.98 or abs(q+m-1)>.05:return None,'weak_or_mirror_conflict'
        if max(raw,1-raw)<.8 or (raw>=.5)!=(q>=.5):return None,'detector_conflict'
        qs.append(q)
    if (qs[0]>=.5)!=(qs[1]>=.5):return None,'camera_conflict'
    return .98 if qs[0]>=.5 else .02,'shortcut'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('session','evidence','data','model','upstream','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if '143912Z' in str(a.session):raise ValueError('Latest local holdout excluded')
    a.output.mkdir(exist_ok=False);torch.set_num_threads(1);sys.path.insert(0,str(a.upstream))
    arrays=np.load(a.data/'development.npz',allow_pickle=False)
    manifest=json.loads((a.data/'manifest.json').read_text());spec=manifest['datasets']['development']
    if sha(a.data/'development.npz')!=spec['archive_sha256'] or sha(a.session/'transport.bin')!=spec['transport_sha256'] or sha(a.evidence)!=spec['evidence_sha256']:raise ValueError('Development data changed')
    x,_,_,keys,_=local_images(a.session,a.evidence)
    if not np.array_equal(x,arrays['images']):raise ValueError('Conditioned crop order mismatch')
    model=load_target(a.model,device='cpu');value=condition(torch.from_numpy(x),torch.from_numpy(arrays['points']))
    q=predict(model,value);mirrored=predict(model,value.flip(-1));scores={(k['stamp'],k['camera'],k['source_id']):(float(q[i]),float(mirrored[i])) for i,k in enumerate(keys)}
    evidence={r['stamp']:r for line in a.evidence.open() if (r:=json.loads(line))}
    refs={r['capture_raw_ns']:r['capture_pose'] for line in (a.session/'predictions.jsonl').open() if (r:=json.loads(line))}
    cameras=load_calibration(a.session/'calibration');plan=StereoIdentityPlan(cameras);control=TemporalAssociation(cameras);candidate=TemporalAssociation(cameras)
    reasons=collections.Counter();differences=[];control_differences=[];frames=original=estimated=skipped=opposite=0
    with (a.output/'trace.jsonl').open('w') as out:
        for meta,_ in packets(a.session/'transport.bin'):
            if meta.get('waiting'):continue
            pose=refs.get(meta['stamp']);ev=evidence.get(meta['stamp'])
            if pose is None or ev is None or not pose['valid']:control.reset();candidate.reset();continue
            obs=next(iter(read_audit(io.StringIO(meta['audit'])).values()));world=pose_matrix(pose['capture'])
            prepared={(v['camera'],v['id']) for v in meta['crops']};views=[v for v in obs['views'] if (v['camera'],v['id']) in prepared and v['presence']>=.5]
            full={(v['camera'],v['source_id']):v for v in ev['views']}
            if {(v['camera'],v['id']) for v in views}!=set(full):raise ValueError('Evidence coverage differs')
            groups,reason=plan.groups(dict(obs,views=views),world);reasons[reason]+=1
            current={(v['camera'],v['id']):scores[meta['stamp'],v['camera'],v['id']] for v in views};replace={}
            for left,right,_ in groups:
                value,reason=shortcut(left,right,current);reasons[reason]+=1
                if value is None:continue
                for view in (left,right):
                    key=(view['camera'],view['id']);replace[key]=value;reference=full[key]
                    opposite+=bool(reference['agreement'] and max(reference['right'],1-reference['right'])>=.85 and (reference['right']>=.5)!=(value>=.5))
            control.assign(dict(obs,views=[dict(v,right=full[v['camera'],v['id']]['right']) for v in views]),world)
            candidate.assign(dict(obs,views=[dict(v,right=replace.get((v['camera'],v['id']),full[v['camera'],v['id']]['right'])) for v in views]),world)
            old=shape(control.last_debug);new=shape(candidate.last_debug)
            if shape(control.last_debug,True)!=shape(ev.get('association',{}),True):control_differences.append(meta['stamp'])
            if new!=old:differences.append(dict(stamp=meta['stamp'],control=old,candidate=new))
            used={(v['camera'],v['source_id']) for row in candidate.last_debug.get('assignments',[]) for v in row['views']}
            original+=2*len(views);estimated+=2*(len(views)-len(replace))+len(set(replace)&used);skipped+=len(replace);frames+=1
            out.write(json.dumps(dict(stamp=meta['stamp'],shortcut_views=[list(k) for k in replace],control=old,candidate=new))+'\n')
    result=dict(scope='Causal association shadow only; no native timing or neural pose replay. Counts include full encoder once for selected shortcut views; tiny classifier/maps overhead not timed.',frames=frames,shortcut_views=skipped,withheld_strong_opposite_labels=opposite,encoder_passes_original=original,encoder_passes_estimated=estimated,estimated_pass_saving_fraction=1-estimated/original,control_mismatch_count=len(control_differences),candidate_assignment_difference_count=len(differences),candidate_assignment_differences=differences,reasons=dict(reasons),checkpoint_sha256=sha(a.model),source_sha256=sha(Path(__file__)))
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='candidate_assignment_differences'},indent=2))


if __name__=='__main__':main()
