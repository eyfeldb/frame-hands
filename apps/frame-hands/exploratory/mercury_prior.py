"""Causal two-frame pose history plus an independent stereo-estimation helper.

EarlierPosePrior stores caller-supplied points. The original prior ablation uses
external stereo estimates; the isolated native-feedback experiment instead supplies
its own earlier accepted fits. Neither source may include current/future points.
"""
import numpy as np
from frame_data import unproject


def stereo_world(cameras,views):
    rays=[];origins=[]
    for name in ['slam_left','slam_right']:
        cam=cameras[name];points=np.asarray(views[name],float)
        if points.shape!=(21,2) or not np.isfinite(points).all():return None,dict(reason='Invalid stereo landmarks')
        rays.append(unproject(cam,points)@cam.camera_to_world_xf[:3,:3].T);origins.append(cam.camera_to_world_xf[:3,3])
    r,s=rays;delta=origins[1]-origins[0];dot=np.sum(r*s,axis=1);den=1-dot*dot
    if np.any(den<np.sin(np.deg2rad(5))**2):return None,dict(reason='Weak stereo angle')
    ra=r@delta;sb=s@delta;ta=(ra-dot*sb)/den;tb=(dot*ra-sb)/den
    if np.any(np.r_[ta,tb]<100) or np.any(np.r_[ta,tb]>1000):return None,dict(reason='Prior range outside100-1000mm')
    pa=origins[0]+r*ta[:,None];pb=origins[1]+s*tb[:,None]
    return (pa+pb)/2,dict(reason='External model hypothesis, not verified pose',max_ray_miss_mm=float(np.linalg.norm(pa-pb,axis=1).max()))


class EarlierPosePrior:
    def __init__(self):self.rows=[];self.last_stamp=None
    def predict(self,stamp,generation,context_valid=True):
        if not context_valid:self.rows=[];return None,'Invalid capture context'
        if self.last_stamp is not None and stamp<=self.last_stamp:self.rows=[];return None,'Nonincreasing capture'
        if len(self.rows)<2:return None,'Need two earlier estimates'
        (ta,ga,a),(tb,gb,b)=self.rows
        if ga!=generation or gb!=generation:self.rows=[];return None,'Generation changed'
        if stamp-tb>100000000:self.rows=[];return None,'Prior expired'
        return b+(b-a)*(.4*(stamp-tb)/(tb-ta)),'Earlier frames only'
    def observe(self,stamp,generation,points):
        if self.last_stamp is not None and stamp<=self.last_stamp:
            self.rows=[];return
        gap=self.last_stamp is not None and stamp-self.last_stamp>100000000
        self.last_stamp=stamp
        if points is None:self.rows=[];return
        points=np.asarray(points,float)
        if points.shape!=(21,3) or not np.isfinite(points).all():self.rows=[];return
        if gap or (self.rows and self.rows[-1][1]!=generation):self.rows=[]
        self.rows.append((stamp,generation,points.copy()));self.rows=self.rows[-2:]
