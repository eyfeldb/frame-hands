"""Unused current-camera correspondences for offline acquisition hypotheses.

Geometry corroborates an observation, not anatomical identity or physical truth.
No state mutation, detector thresholds or live publication.
"""
import numpy as np
from frame_data import NAMES
from temporal_association import TemporalAssociation


def unused_matches(hands,observation,cameras,world,eligible):
    if world is None or np.shape(world)!=(4,4) or not np.isfinite(world).all():return {}
    if len({h['side'] for h in hands})!=len(hands):raise ValueError('Duplicate slots')
    policy=TemporalAssociation(cameras)
    cams={n:c.copy(camera_to_world_xf=world@c.camera_to_world_xf) for n,c in cameras.items()}
    views={}
    for v in observation['views']:
        key=(v['camera'],v['id']);points=np.asarray(v['points'])
        if key[0] not in NAMES or points.shape!=(21,2) or not np.isfinite(points).all():continue
        if not np.isfinite(v['presence']) or not .5<=v['presence']<=1:continue
        if key in views:raise ValueError('Duplicate current view identity')
        views[key]=v
    result={}
    for h in hands:
        if h['source_id'] not in eligible or len(h['source_views'])!=1:continue
        source=h['source_views'][0];key=(source['camera'],source['source_id'])
        if key not in views:continue
        occupied={(v['camera'],v['source_id']) for other in hands if other['side']!=h['side'] for v in other['source_views']}
        first=policy._feature(views[key],cams[key[0]]);matches=[]
        for k,v in views.items():
            if k[0]==key[0] or k in occupied:continue
            pair=policy._pair(first,policy._feature(v,cams[k[0]]))
            if pair is not None:matches.append(dict(view=v,error=pair['error']))
        if len(matches)==1:result[h['side']]=dict(source=views[key],other=matches[0]['view'],error=matches[0]['error'])
    # Joint ownership: two hypotheses cannot borrow the same unused observation.
    keys=[(m['other']['camera'],m['other']['id']) for m in result.values()]
    return {s:m for s,m in result.items() if keys.count((m['other']['camera'],m['other']['id']))==1}
