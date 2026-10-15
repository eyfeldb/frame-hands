"""Score predicted tips against sparse, uncertainty-marked raw-image regions.

Labels are evaluation-only and may be assistant visual estimates. Optimistic
injective matching handles explicit anatomical ambiguity; it is not a dense
accuracy benchmark, occlusion inference or a physical 3D ground-truth test.
"""
import argparse,collections,hashlib,json,sys
from pathlib import Path
import cv2,numpy as np
from scipy.optimize import linear_sum_assignment
from frame_data import load_calibration
from fixed_scale_backend import ScaledBackend
from probe_wrist_refinement import landmarks
from run_mesh_trial import pose_matrix


def validate_annotations(data, manifest, root, labels_sha256, review=None):
    """Bind visual claims to immutable raw assets before reading predictions.

    A later review may reject a claim, but never move it to fit model output.
    Rejections are separate, hash-bound records retained in the scored report.
    """
    if not data.get('frozen_before_candidate_projection'):
        raise ValueError('Freeze visual labels before scoring')
    labels = data['labels']
    ids = [row['label_id'] for row in labels]
    if any(type(i) is not int for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Annotation IDs must be globally unique integers')
    crops = {}
    for frame in manifest['frames']:
        for crop in frame['crops']:
            key = (frame['stamp'], crop['camera'], crop['source_id'])
            if key in crops:
                raise ValueError('Duplicate capture/camera/crop identity')
            crops[key] = crop
    for row in labels:
        if type(row['physical_side']) is not int or row['physical_side'] not in (0, 1):
            raise ValueError('Invalid physical side')
        key = (row['stamp'], row['camera'], row['crop_source_id'])
        if key not in crops:
            raise ValueError('Annotation has no exact raw crop')
        crop = crops[key]
        if row['raw_sha256'] != crop['raw_sha256']:
            raise ValueError('Annotation raw hash differs from manifest')
        raw = root / Path(crop['path']).parent / (crop['camera'] + '.png')
        if hashlib.sha256(raw.read_bytes()).hexdigest() != row['raw_sha256']:
            raise ValueError('Annotated raw image changed')
        # Validate anatomical and coordinate claims even when a review rejects them.
        match_regions([row], np.full((5, 2), np.nan))
    rejected = {}
    if review is not None:
        if review['labels_sha256'] != labels_sha256:
            raise ValueError('Review belongs to a different annotation file')
        for row in review['rejected_labels']:
            i = row['label_id']
            if i not in ids or i in rejected or not row.get('reason', '').strip():
                raise ValueError('Invalid or duplicate review rejection')
            rejected[i] = row['reason']
    return rejected


def match_regions(labels,tips):
    tips=np.asarray(tips,float)
    if tips.shape!=(5,2):raise ValueError('Expected five projected tips')
    if len(labels)>5 or len({r['label_id'] for r in labels})!=len(labels):raise ValueError('Duplicate or excessive labels')
    distances=np.full((len(labels),5),1e12);cost=distances.copy()
    for i,row in enumerate(labels):
        xy=np.asarray(row['xy'],float);radius=row['radius_px'];allowed=row['possible_native_tips']
        if xy.shape!=(2,) or not np.isfinite(xy).all() or not np.isfinite(radius) or radius<=0 or row['visibility']!='visible distal region':raise ValueError('Invalid or nonvisible annotation')
        if not allowed or len(set(allowed))!=len(allowed) or any(not isinstance(n,int) or not 0<=n<5 for n in allowed):raise ValueError('Invalid tip identity set')
        for tip in allowed:
            if np.isfinite(tips[tip]).all():
                d=float(np.linalg.norm(tips[tip]-xy));distances[i,tip]=d;cost[i,tip]=max(0,d-radius)+d*1e-8
    # Dummy columns represent missing projections and prevent duplicate use.
    padded=np.column_stack((cost,np.full((len(labels),len(labels)),1e9)))
    ids,slots=linear_sum_assignment(padded);result=[]
    for i,slot in zip(ids,slots):
        row=labels[i];valid=slot<5 and distances[i,slot]<1e9
        d=float(distances[i,slot]) if valid else None
        result.append(dict(label_id=row['label_id'],status=('within_region' if d<=row['radius_px'] else 'outside_region') if valid else 'unmatched_tip',matched_tip=int(slot) if valid else None,projected_xy=tips[slot].tolist() if valid else None,distance_px=d,outside_radius_px=max(0,d-row['radius_px']) if valid else None,ambiguous_anatomy=len(row['possible_native_tips'])>1))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('session','predictions','labels','upstream','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--review',type=Path,help='Separate hash-bound annotation rejection audit')
    p.add_argument('--scale',type=float,default=1.);a=p.parse_args();data=json.loads(a.labels.read_text());labels=data['labels']
    manifest=json.loads((a.labels.parent/'manifest.json').read_text());review=json.loads(a.review.read_text()) if a.review else None
    label_hash=hashlib.sha256(a.labels.read_bytes()).hexdigest()
    rejected=validate_annotations(data,manifest,a.labels.parent,label_hash,review)
    a.output.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(a.upstream));cal=a.session/'calibration';cameras=load_calibration(cal if cal.exists() else a.session/'frame/snapshot/device/persist');backend=ScaledBackend(a.upstream,cameras,hand_scale=a.scale)
    wanted={r['stamp'] for r in labels};predictions={}
    with a.predictions.open() as f:
        for line in f:
            row=json.loads(line)
            if row['capture_raw_ns'] in wanted:
                if row['capture_raw_ns'] in predictions:raise ValueError('Duplicate exact model frame')
                predictions[row['capture_raw_ns']]=row
    groups=collections.defaultdict(list)
    for r in labels:
        if r['label_id'] not in rejected:groups[(r['stamp'],r['camera'],r['physical_side'])].append(r)
    results=[dict(label_id=r['label_id'],stamp=r['stamp'],camera=r['camera'],side=r['physical_side'],status='untrusted_annotation',reason=rejected[r['label_id']]) for r in labels if r['label_id'] in rejected]
    for (stamp,camera,side),group in groups.items():
        row=predictions.get(stamp);status=None
        if row is None:status='no_exact_model_frame'
        elif not row['capture_pose']['valid']:status='invalid_capture_context'
        else:
            hands=[h for h in row['hands'] if h['side']==side]
            if len(hands)>1:raise ValueError('Duplicate anatomical slot')
            if not hands:status='missing_hand'
        if status:scored=[dict(label_id=r['label_id'],status=status) for r in group]
        else:
            xyz=landmarks(backend,hands[0])[:5];world=pose_matrix(row['capture_pose']['capture']);cam=cameras[camera].copy(camera_to_world_xf=world@cameras[camera].camera_to_world_xf);eye=cam.world_to_eye(xyz);uv=cam.eye_to_window(eye);uv[eye[:,2]<=0]=np.nan
            scored=match_regions(group,uv)
        results.extend(dict(stamp=stamp,camera=camera,side=side,**r) for r in scored)
    byid={r['label_id']:r for r in results}
    for frame in manifest['frames']:
        cards=[]
        for crop in frame['crops']:
            card=cv2.imread(str(a.labels.parent/crop['path']))
            raw=a.labels.parent/Path(crop['path']).parent/(crop['camera']+'.png')
            if hashlib.sha256(raw.read_bytes()).hexdigest()!=crop['raw_sha256']:raise ValueError('Raw image changed')
            for label in labels:
                if label['stamp']!=frame['stamp'] or label['camera']!=crop['camera'] or label['crop_source_id']!=crop['source_id']:continue
                def canvas(p):return tuple(np.rint((np.asarray(p)-crop['crop_origin'])*crop['scale']+crop['canvas_origin']).astype(int))
                center=canvas(label['xy']);cv2.circle(card,center,int(round(label['radius_px']*crop['scale'])),(255,255,0),1);result=byid[label['label_id']]
                if result.get('projected_xy') is not None:
                    point=canvas(result['projected_xy']);cv2.drawMarker(card,point,(0,0,255),cv2.MARKER_CROSS,15,2);cv2.line(card,center,point,(0,150,255),1)
                cv2.putText(card,f'{label["label_id"]}: {result["status"]}',(5,495-label['label_id']%2*15),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
            cards.append(card)
        if len(cards)%2:cards.append(np.zeros_like(cards[0]))
        cv2.imwrite(str(a.output/f'{frame["seconds"]:06.3f}.png'),np.vstack([np.hstack(cards[i:i+2]) for i in range(0,len(cards),2)]))
    report=dict(scope=__doc__,annotation_provenance=data['provenance'],annotation_review=review,labels_sha256=label_hash,predictions_sha256=hashlib.sha256(a.predictions.read_bytes()).hexdigest(),scale=a.scale,counts=dict(collections.Counter(r['status'] for r in results)),results=results,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(counts=report['counts'],results=results),indent=2))

if __name__=='__main__':main()
