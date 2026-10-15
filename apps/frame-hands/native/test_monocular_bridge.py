import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from probe_monocular_bridge import continue_rigid,transform,PALM_RIG

class Camera:
 def __init__(self):
  self.camera_to_world_xf=np.eye(4);self.camera_to_world_xf[:3,:3]=Rotation.from_rotvec([.1,.2,-.05]).as_matrix();self.camera_to_world_xf[:3,3]=[30,-10,5]
 def world_to_eye(self,points):return transform(points,np.linalg.inv(self.camera_to_world_xf))
 def eye_to_window(self,points):return points[:,:2]/points[:,2:]*500+256

class MonocularControls(unittest.TestCase):
 def setUp(self):
  self.cam=Camera();self.eye=np.random.default_rng(12).normal(0,20,(21,3));self.eye[:,2]+=500
  self.seed=transform(self.eye,self.cam.camera_to_world_xf)
 def test_known_plane_translation_recovers_and_preserves_shape(self):
  target=self.eye+[10,-6,0];pixels=self.cam.eye_to_window(target[PALM_RIG]);old=self.seed.copy()
  actual,meta=continue_rigid(self.cam,self.seed,pixels,.067,'plane')
  self.assertTrue(meta['accepted']);self.assertLess(np.max(np.linalg.norm(self.cam.world_to_eye(actual)-target,axis=1)),.2)
  np.testing.assert_allclose(actual[:,None]-actual[None,:],old[:,None]-old[None,:],atol=1e-10)
  np.testing.assert_array_equal(self.seed,old)
 def test_rigid_update_preserves_pair_distances(self):
  center=self.eye[PALM_RIG].mean(0);target=(self.eye-center)@Rotation.from_rotvec([.04,-.02,.03]).as_matrix().T+center+[3,-2,0]
  actual,meta=continue_rigid(self.cam,self.seed,self.cam.eye_to_window(target[PALM_RIG]),.067,'rigid')
  self.assertTrue(meta['accepted']);self.assertLess(meta['observed_after_px'],meta['observed_before_px'])
  np.testing.assert_allclose(np.linalg.norm(actual[:,None]-actual[None,:],axis=2),np.linalg.norm(self.seed[:,None]-self.seed[None,:],axis=2),atol=1e-10)
 def test_old_or_invalid_inputs_cannot_become_fresh_pose(self):
  pixels=self.cam.eye_to_window(self.eye[PALM_RIG])
  for age in (0,.101):
   actual,meta=continue_rigid(self.cam,self.seed,pixels,age,'plane');self.assertFalse(meta['accepted']);np.testing.assert_array_equal(actual,self.seed)
  pixels[0,0]=np.nan
  with self.assertRaises(ValueError):continue_rigid(self.cam,self.seed,pixels,.033,'plane')

if __name__=='__main__':unittest.main()
