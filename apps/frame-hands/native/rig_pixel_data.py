"""Offline serialization for the explicit two-camera KB fitter. No runtime dependencies."""
import numpy as np
from scipy.spatial.transform import Rotation
NAMES=('slam_left','slam_right')
def write_cameras(cameras,path):
 rows=[]
 for name in NAMES:
  c=cameras[name]
  if len(c.distort)!=8 or np.any(np.asarray(c.distort)[4:]!=0):raise ValueError('Only verified four-coefficient radial KB calibration is supported')
  R=c.camera_to_world_xf[:3,:3].T;t=-R@c.camera_to_world_xf[:3,3]
  rows.append([*R.reshape(-1),*t,*c.f,*c.c,*c.distort[:4]])
 with path.open('x') as f:f.write('FHCAMKB 1 2\n');np.savetxt(f,rows,fmt='%.17g')
def pixel_row(seed,points,present,sigma=3.,huber=3.):
 row=[*seed,30,huber,.01]
 for j in range(21):
  for v in range(2):row.extend([int(present[v,j]),*points[v,j],sigma])
 return row
