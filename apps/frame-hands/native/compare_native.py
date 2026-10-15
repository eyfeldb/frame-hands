"""Compare native replays by capture stamp and selected side/source views."""
import argparse,itertools,json
from pathlib import Path
import numpy as np
from compare_student import hand_key,quantiles


def compare(reference,candidate):
    counts=[0,0];frames=changed=availability=0;vertices=[];wrists=[];angles=[];stamps=[]
    with reference.open() as left,candidate.open() as right:
        for a,b in itertools.zip_longest(left,right):
            if a is None or b is None:raise ValueError('Replay lengths differ')
            a,b=json.loads(a),json.loads(b)
            if a['stamp']!=b['stamp']:raise ValueError('Capture stamps differ')
            maps=[{hand_key(h):h for h in item['hands']} for item in (a,b)]
            if any(len(m)!=len(item['hands']) for m,item in zip(maps,(a,b))):raise ValueError('Duplicate hand source keys')
            different=maps[0].keys()!=maps[1].keys();changed+=different
            if different:stamps.append(a['stamp'])
            availability+={h['side'] for h in a['hands']}!={h['side'] for h in b['hands']}
            counts[0]+=len(a['hands']);counts[1]+=len(b['hands'])
            for key in maps[0].keys()&maps[1].keys():
                old,new=maps[0][key],maps[1][key]
                for field,values in (('vertices_world_mm',vertices),('wrist_world_mm',wrists),('joint_angles',angles)):
                    x,y=np.asarray(old[field]),np.asarray(new[field])
                    if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Invalid output array')
                    values.append(float(np.abs(x-y).max()))
            frames+=1
    return dict(scope='Numerical saved replay parity by side and source views; no physical accuracy or acceptance claim.',frames=frames,hand_counts=counts,assignment_changed_frames=changed,availability_changed_frames=availability,changed_stamps=stamps,max_vertex_coordinate_difference_mm_p50_p95_max=quantiles(vertices),max_wrist_matrix_component_difference_p50_p95_max=quantiles(wrists),max_joint_angle_difference_radians_p50_p95_max=quantiles(angles))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','candidate','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();result=compare(a.reference,a.candidate)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
