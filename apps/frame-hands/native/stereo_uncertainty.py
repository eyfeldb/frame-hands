"""Current stereo geometry sensitivity, not calibrated ML confidence.
Units: millimeters. Covariance assumes independent isotropic one-pixel errors.
"""
import numpy as np
from frame_data import unproject

def triangulate(origins,rays):
    origins=np.asarray(origins,float);rays=np.asarray(rays,float)
    if origins.shape!=(2,3) or rays.ndim!=3 or rays.shape[0]!=2 or rays.shape[2]!=3 or not np.isfinite(origins).all() or not np.isfinite(rays).all():raise ValueError('Expected two finite cameras and Nx3 rays')
    n=np.linalg.norm(rays,axis=2,keepdims=True)
    if np.any(n<1e-10):raise ValueError('Degenerate ray')
    rays=rays/n
    projectors=np.eye(3)-rays[...,None]*rays[...,None,:]
    information=projectors.sum(axis=0)
    eigen=np.linalg.eigvalsh(information)
    if np.any(eigen[:,0]<1e-8):raise ValueError('Degenerate stereo intersection')
    rhs=np.einsum('vnij,vj->ni',projectors,origins)
    points=np.linalg.solve(information,rhs[...,None])[...,0]
    depth=np.einsum('vni,vni->vn',points[None]-origins[:,None],rays)
    if np.any(depth<=0):raise ValueError('Intersection behind a camera')
    distance=np.linalg.norm(np.einsum('vnij,vnj->vni',projectors,points[None]-origins[:,None]),axis=2)
    angle=np.arccos(np.clip(np.einsum('ni,ni->n',rays[0],rays[1]),-1,1))
    return points,dict(angle_degrees=np.rad2deg(angle),ray_distance_mm=distance,condition=eigen[:,-1]/eigen[:,0])

def projection_jacobian(camera,points,step_mm=.1):
    x=np.asarray(points,float);j=[]
    for axis in range(3):
        d=np.eye(3)[axis]*step_mm
        a=camera.eye_to_window(camera.world_to_eye(x+d));b=camera.eye_to_window(camera.world_to_eye(x-d))
        j.append((a-b)/(2*step_mm))
    return np.stack(j,axis=2)

def sensitivity(cameras,views):
    if len(views)!=2 or len({v['camera'] for v in views})!=2:raise ValueError('Two different current cameras required')
    cs=[cameras[v['camera']] for v in views]
    origins=np.array([c.camera_to_world_xf[:3,3] for c in cs])
    rays=np.array([(unproject(c,v['points']) if len(c.distort)>=4 else c.window_to_eye(v['points']))@c.camera_to_world_xf[:3,:3].T for c,v in zip(cs,views)])
    points,info=triangulate(origins,rays)
    # Pixel residual Jacobians preserve fisheye angular sensitivity at each joint.
    j=np.concatenate([projection_jacobian(c,points) for c in cs],axis=1)
    information=np.einsum('nki,nkj->nij',j,j)
    if np.any(np.linalg.eigvalsh(information)[:,0]<1e-10):raise ValueError('Singular pixel information')
    covariance=np.linalg.inv(information)
    if not np.isfinite(covariance).all():raise ValueError('Nonfinite sensitivity')
    residual=np.array([np.linalg.norm(c.eye_to_window(c.world_to_eye(points))-v['points'],axis=1) for c,v in zip(cs,views)])
    return points,covariance,dict(info,reprojection_px=residual,major_axis_mm_per_pixel=np.sqrt(np.linalg.eigvalsh(covariance)[:,-1]))
