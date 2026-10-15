"""Export the pinned generic hand as a versioned native rig; no personal calibration claim."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
MODEL_SHA='96649ddf7d036be3357dadd7c13db325a18638c6713583fbbbdd71446ce07564'
def export(model,output):
    model=Path(model);output=Path(output)
    if hashlib.sha256(model.read_bytes()).hexdigest()!=MODEL_SHA:raise ValueError('Unpinned generic anatomy')
    d=json.loads(model.read_text());axes=np.asarray(d['joint_rotation_axes'],float);pivots=np.asarray(d['joint_rest_positions'],float);limits=np.asarray(d['joint_limits'],float)
    rest=np.asarray(d['landmark_rest_positions'],float);indices=np.asarray(d['landmark_rest_bone_indices'],int);sparse=np.asarray(d['landmark_rest_bone_weights'],float)
    if axes.shape!=(22,3) or pivots.shape!=(22,3) or limits.shape!=(22,2) or rest.shape!=(21,3) or indices.shape!=sparse.shape or sparse.shape!=(21,3):raise ValueError('Changed model schema')
    weights=np.zeros((21,17))
    for row in range(21):
        seen=set()
        for index,w in zip(indices[row],sparse[row]):
            if not w:continue
            if index<0 or index>=17 or index in seen:raise ValueError('Invalid/duplicate weight index')
            weights[row,index]=w;seen.add(index)
    if not np.isfinite(np.r_[axes.ravel(),pivots.ravel(),limits.ravel(),rest.ravel(),weights.ravel()]).all() or not np.allclose(weights.sum(1),1,atol=1e-6):raise ValueError('Invalid model numbers')
    output.mkdir(parents=True,exist_ok=False)
    with (output/'generic.fhrig').open('x') as f:
        f.write(f'FHRIG 1 mm left generic -1 {MODEL_SHA} - 20 21 17\n')
        for row in np.column_stack((axes[:20],pivots[:20],limits[:20])):f.write(' '.join(format(v,'.17g') for v in row)+'\n')
        for row in np.column_stack((rest,weights)):f.write(' '.join(format(v,'.17g') for v in row)+'\n')
    manifest=dict(format='FHRIG',version=1,units='millimeters',canonical_side='left',status='generic; not personalized',model_sha256=MODEL_SHA,profile_sha256=hashlib.sha256((output/'generic.fhrig').read_bytes()).hexdigest(),articulated_dof=20,wrist_dof=6,frames=17,landmarks=21,upstream_unused_angle_slots=[20,21],notes=['Upstream skin_landmarks uses only first20angles; wrist transform supplies root pose.','Right hand mirrors canonical x before applying a proper wrist transform.','MediaPipe thumb CMC has no direct equivalent; mapping deliberately marks it absent.','Rest geometry/weights are private upstream assets; existing license restrictions remain.'],export_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(export(a.model,a.output),indent=2))
