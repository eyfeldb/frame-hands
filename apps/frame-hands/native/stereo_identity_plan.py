"""Same-capture stereo identity sharing probe; never enabled by a live launcher."""
import numpy as np
from identity_schedule import IdentitySchedule


def unique_pairs(costs, maximum=.015, margin=.03):
    """Mutual unique geometric matches; ambiguity in either camera rejects reuse."""
    costs=np.asarray(costs,dtype=float)
    if costs.ndim!=2:raise ValueError('Expected pair cost matrix')
    pairs=[]
    for i in range(costs.shape[0]):
        for j in range(costs.shape[1]):
            value=costs[i,j]
            if not np.isfinite(value) or value<0 or value>maximum:continue
            row=np.delete(costs[i],j);column=np.delete(costs[:,j],i)
            if np.any(row<value+margin) or np.any(column<value+margin):continue
            pairs.append((i,j))
    return pairs


def shared_score(primary,secondary,evidence):
    """Only primary image evidence and both current detector labels are inputs."""
    q=float(evidence['right'])
    if not evidence['agreement'] or not np.isfinite(q) or not 0<=q<=1 or max(q,1-q)<.85:
        return None,'weak_primary_image'
    for view in (primary,secondary):
        raw=float(view['right'])
        if not np.isfinite(raw) or not 0<=raw<=1 or max(raw,1-raw)<.8 or (raw>=.5)!=(q>=.5):
            return None,'detector_conflict'
    return q,'share'


class StereoIdentityPlan:
    def __init__(self,cameras):
        from temporal_association import TemporalAssociation
        self.geometry=TemporalAssociation(cameras)
        self.cameras=cameras

    def groups(self,observation,world):
        from frame_data import NAMES
        views=observation['views']
        if world is None or np.shape(world)!=(4,4) or not np.isfinite(world).all():return [],'invalid_pose'
        if not IdentitySchedule.eligible_geometry(views):return [],'overlap_or_invalid_points'
        if len({(v['camera'],v['id']) for v in views})!=len(views):return [],'duplicate_view'
        by_camera={name:[v for v in views if v['camera']==name and .5<=v['presence']<=1] for name in NAMES}
        if any(not any(v['presence']>=.8 for v in rows) for rows in by_camera.values()):return [],'missing_high_quality_view'
        if any(len(rows)>4 for rows in by_camera.values()):raise ValueError('Unbounded views')
        cameras={name:camera.copy(camera_to_world_xf=world@camera.camera_to_world_xf) for name,camera in self.cameras.items()}
        features={name:[self.geometry._feature(v,cameras[name]) for v in by_camera[name]] for name in NAMES}
        costs=np.full((len(features[NAMES[0]]),len(features[NAMES[1]])),np.inf)
        for i,a in enumerate(features[NAMES[0]]):
            for j,b in enumerate(features[NAMES[1]]):
                pair=self.geometry._pair(a,b)
                if pair is not None:costs[i,j]=pair['error']
        groups=[]
        for i,j in unique_pairs(costs):
            left,right=by_camera[NAMES[0]][i],by_camera[NAMES[1]][j]
            if min(left['presence'],right['presence'])<.8:continue
            primary,secondary=(left,right) if left['presence']>=right['presence'] else (right,left)
            groups.append((primary,secondary,float(costs[i,j])))
        return groups,'unique_pairs' if groups else 'ambiguous_or_poor_stereo'
