# Copyright 2021-2023, Collabora, Ltd. (adapted projection/heatmap decoding)
# SPDX-License-Identifier: BSL-1.0
"""Offline Mercury stereographic crop and keypoint probe.

Adapted from pinned Monado22d5c936 hg_image_distorter.cpp/hg_model.cpp.
Input ROI is external hypothesis geometry, not ground truth or Mercury detection.
"""
import numpy as np
from frame_data import unproject
from mercury_probe import normalize

FLIP=np.array([1.,-1.,-1.])


def align_forward(direction):
    d=np.asarray(direction,float);d=d/np.linalg.norm(d)
    a=np.array([0.,0.,-1.]);v=np.cross(a,d);c=float(a@d)
    if not np.isfinite(d).all() or c<=-1+1e-8:raise ValueError('Degenerate camera direction')
    x,y,z=v;k=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+k+k@k/(1+c)


class StereoCrop:
    def __init__(self,camera,center,size,side):
        center=np.asarray(center,float)
        if center.shape!=(2,) or not np.isfinite(center).all() or not 4<=size<=4096 or side not in (0,1):raise ValueError('Invalid ROI')
        self.camera=camera;self.side=side
        corners=center+np.array([[1,1],[-1,1],[-1,-1],[1,-1]])*size/2
        rays=unproject(camera,np.vstack((center,corners)))*FLIP
        angle=float(np.max(np.arccos(np.clip(rays[1:]@rays[0],-1,1))))
        self.radius=float(np.tan(angle/2)*.65) # Pinned after_detection_fac.
        if not 0<self.radius<1:raise ValueError('Invalid stereographic radius')
        self.rotation=align_forward(rays[0]);self.center=center;self.size=float(size)

    @classmethod
    def from_geometry(cls,camera,rotation,radius,side):
        """Restore an exact saved crop for paired offline inference."""
        rotation=np.asarray(rotation,float)
        if side not in (0,1) or not np.isfinite(radius) or not 0<radius<1:raise ValueError('Invalid saved crop')
        if rotation.shape!=(3,3) or not np.isfinite(rotation).all() or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-7) or abs(np.linalg.det(rotation)-1)>1e-7:raise ValueError('Invalid saved crop rotation')
        result=cls.__new__(cls);result.camera=camera;result.side=side;result.rotation=rotation.copy();result.radius=float(radius);result.center=None;result.size=None
        return result

    @classmethod
    def from_predicted_points(cls,camera,points_eye,side):
        """Pinned native predicted ROI algorithm, zero twist and expansion1.7.

        Uses float64 projection here, not byte-identical native float32 arithmetic.
        All points must come from strictly earlier poses in the caller.
        """
        return cls.from_hand_geometry(camera,points_eye,side)

    @classmethod
    def from_hand_geometry(cls,camera,points_eye,side):
        """Pure native ROI geometry; caller declares current seed versus prediction.

        This method neither selects identity nor creates a temporal prediction.
        The21slots follow Mercury indices. If an external skeleton lacks a joint,
        callers may duplicate an existing point for bounds only, explicitly noting
        the missing joint; such a duplicate must never become a pose observation.
        """
        eye=np.asarray(points_eye,float)
        if eye.shape!=(21,3) or not np.isfinite(eye).all() or np.any(eye[:,2]<=0):raise ValueError('Invalid predicted camera points')
        joints=eye*FLIP;directions=joints/np.linalg.norm(joints,axis=1)[:,None]
        old_direction=directions[9];rotation=align_forward(old_direction)
        def unscaled(rot):
            local=directions@rot;denominator=1-local[:,2]
            if np.any(denominator<1e-8):raise ValueError('Predicted point behind stereographic crop')
            return local[:,:2]/denominator[:,None]
        for iteration in range(8):
            xy=unscaled(rotation);low=xy.min(0);high=xy.max(0);center=(low+high)*.5;radius=float(np.max(center-low))
            x,y=center;squared=x*x+y*y;new_direction=rotation@np.array([2*x,2*y,squared-1])/(1+squared)
            rotation=align_forward(new_direction)
            if np.linalg.norm(old_direction-new_direction)<.0001:break
            old_direction=new_direction
        xy=unscaled(rotation);palm=max(np.linalg.norm(xy[0]-xy[9]),np.linalg.norm(xy[5]-xy[17]));floor=float(palm*.5*(2.2/1.65)*1.7)
        expanded=radius*1.7;result=cls.from_geometry(camera,rotation,max(expanded,floor),side)
        return result,dict(iterations=iteration+1,expansion=1.7,unexpanded_radius=radius,palm_radius_floor=floor,palm_floor_applied=floor>expanded)

    def rays(self,points):
        points=np.asarray(points,float)
        x=(points[...,0]/64-1)*self.radius*(-1 if self.side else 1)
        y=(1-points[...,1]/64)*self.radius
        local=np.stack((2*x,2*y,x*x+y*y-1),axis=-1)
        return (local@self.rotation.T)*FLIP

    def pixels(self,points):
        rays=self.rays(points)
        if not np.isfinite(rays).all() or np.any(rays[...,2]<=0):raise ValueError('Crop crosses camera forward hemisphere')
        return self.camera.eye_to_window(rays)

    def normalized_coordinates(self,points_eye):
        eye=np.asarray(points_eye,float)
        if eye.shape!=(21,3) or not np.isfinite(eye).all() or np.any(eye[:,2]<=0):raise ValueError('Invalid prior camera points')
        local=(eye*FLIP)@self.rotation;local/=np.linalg.norm(local,axis=1)[:,None]
        denominator=1-local[:,2]
        if np.any(denominator<1e-8):raise ValueError('Prior behind crop')
        xy=local[:,:2]/denominator[:,None]/self.radius
        xy[:,1]*=-1
        if self.side:xy[:,0]*=-1
        return xy

    def sample(self,image):
        yy,xx=np.mgrid[:128,:128];uv=self.pixels(np.stack((xx,yy),-1))
        # Pinned native remap truncates float to integer, not bilinear/round.
        indices=np.trunc(uv).astype(np.int64);x,y=indices[...,0],indices[...,1]
        valid=(x>=0)&(x<image.shape[1])&(y>=0)&(y<image.shape[0]);out=np.zeros((128,128),np.uint8)
        out[valid]=image[y[valid],x[valid]]
        return out,float(valid.mean())


def heatmap_centers(heatmaps):
    heatmaps=np.asarray(heatmaps,float)
    if heatmaps.shape!=(21,22,22) or not np.isfinite(heatmaps).all():raise ValueError('Invalid XYheatmaps')
    points=[];confidence=[]
    for heat in heatmaps:
        row,col=np.unravel_index(np.argmax(heat),heat.shape)
        wx=max(0,min(col,10,abs(col-22)-1));wy=max(0,min(row,10,abs(row-22)-1))
        patch=heat[row-wy:row+wy+1,col-wx:col+wx+1];total=patch.sum()
        if total==0:xy=np.array([col,row],float)
        else:
            yy,xx=np.mgrid[row-wy:row+wy+1,col-wx:col+wx+1]
            xy=np.array([np.sum(patch*(xx+.5)),np.sum(patch*(yy+.5))])/total
        points.append(xy*128/22);confidence.append(float(heat[row,col]))
    return np.array(points),confidence


def estimate(model,crop,image,prior=None,pixels_override=None):
    pixels,valid_fraction=crop.sample(image)
    if pixels_override is not None:
        if pixels_override.shape!=(128,128) or pixels_override.dtype!=np.uint8:raise ValueError("Invalid crop control")
        pixels=pixels_override.copy()
    values=normalize(pixels)
    if values is None:return dict(reason='No crop variance',active=False,valid_fraction=valid_fraction),pixels
    last=np.zeros((1,42),np.float32);use=np.zeros(1,np.float32)
    if prior is not None:
        prior=np.asarray(prior,float)
        if prior.shape!=(21,2) or not np.isfinite(prior).all():raise ValueError('Invalid normalized prior')
        last[:]=prior.reshape(1,42);use[0]=1
    outputs=model.keypoints.run(['heatmap_xy','heatmap_depth','scalar_extras','curls'],dict(inputImg=values[None,None],lastKeypoints=last,useLastKeypoints=use))
    if not all(np.isfinite(x).all() for x in outputs):raise ValueError('Nonfinite keypoint result')
    points,confidence=heatmap_centers(outputs[0][0]);logit=float(outputs[2][0,0]);presence=float(1/(1+np.exp(-np.clip(logit,-700,700))))
    return dict(reason=('Current-image hypothesis with earlier-frame prior, not verified visibility' if prior is not None else 'Current-image stateless hypothesis, not verified visibility'),active=presence>=.97,presence=presence,confidence_xy=confidence,points_crop=points.tolist(),points_raw=crop.pixels(points).tolist(),valid_fraction=valid_fraction,stereographic_radius=crop.radius,rotation=crop.rotation.tolist(),depth_heatmap=outputs[1][0].tolist(),scalar_extras=outputs[2][0].tolist(),curls=outputs[3][0].tolist()),pixels
