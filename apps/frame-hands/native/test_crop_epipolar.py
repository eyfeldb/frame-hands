import unittest
from types import SimpleNamespace
import numpy as np
from audit_crop_expansion import epi
class EpipolarControls(unittest.TestCase):
 def test_known_correspondence_and_wrong_finger(self):
  cameras={};pixels=[];points=np.array([[x,y,450.+i*8] for i,(x,y) in enumerate([(-30,-40),(-20,20),(10,45),(20,-25),(40,10)])])
  for name,origin in [('slam_left',-30.),('slam_right',30.)]:
   xf=np.eye(4);xf[0,3]=origin;cameras[name]=SimpleNamespace(camera_to_world_xf=xf,c=np.zeros(2),f=np.array([500.,500.]),distort=np.zeros(8));eye=points-np.array([origin,0,0]);radius=np.linalg.norm(eye[:,:2],axis=1);pixels.append(eye[:,:2]*(np.arctan2(radius,eye[:,2])/radius)[:,None]*500)
  self.assertLess(float(epi(cameras,pixels).max()),1e-10)
  swapped=pixels[1][[1,0,2,3,4]];self.assertGreater(float(epi(cameras,[pixels[0],swapped])[:2].min()),1)
  np.testing.assert_allclose(epi(cameras,[pixels[0],swapped])[2:],0,atol=1e-10)
 def test_zero_baseline_rejected(self):
  cam=SimpleNamespace(camera_to_world_xf=np.eye(4),c=np.zeros(2),f=np.ones(2)*500,distort=np.zeros(8))
  with self.assertRaises(AssertionError):epi({'slam_left':cam,'slam_right':cam},[np.array([[10.,20.]]),np.array([[10.,20.]])])
if __name__=='__main__':unittest.main()
