import unittest
import numpy as np
from mercury_keypoints import StereoCrop
from replay_mercury_feedback import encoded_prior
from test_mercury_keypoints import Fisheye

class FrozenFeedbackCropTests(unittest.TestCase):
    def test_saved_crop_roundtrip_preserves_pixels_and_rays_both_sides(self):
        image=np.random.default_rng(27).integers(0,256,(480,640),dtype=np.uint8)
        for side in (0,1):
            original=StereoCrop(Fisheye(),[387,195],117,side)
            restored=StereoCrop.from_geometry(Fisheye(),original.rotation.tolist(),original.radius,side)
            np.testing.assert_array_equal(original.sample(image)[0],restored.sample(image)[0])
            points=np.array([[0,0],[64,64],[127,127]])
            np.testing.assert_array_equal(original.rays(points),restored.rays(points))
    def test_saved_crop_rejects_nonrigid_or_reflected_rotation(self):
        for matrix in [np.diag([2,1,1]),np.diag([-1,1,1]),np.full((3,3),np.nan)]:
            with self.assertRaises(ValueError):StereoCrop.from_geometry(Fisheye(),matrix,.2,0)
        for radius in (0,-.1,float('nan'),1):
            with self.assertRaises(ValueError):StereoCrop.from_geometry(Fisheye(),np.eye(3),radius,0)
    def test_prior_encoding_rejects_outside_or_behind_and_keeps_world_input(self):
        class Camera:
            def world_to_eye(self,points):return np.asarray(points)
        crop=StereoCrop(Fisheye(),[320,240],120,0);camera=Camera()
        points=crop.rays(np.full((21,2),64))*300;before=points.copy()
        encoded,_=encoded_prior(crop,camera,points);np.testing.assert_allclose(encoded,0,atol=1e-9);np.testing.assert_array_equal(before,points)
        self.assertIsNone(encoded_prior(crop,camera,None)[0])
        self.assertIsNone(encoded_prior(crop,camera,-points)[0])
        self.assertIsNone(encoded_prior(crop,camera,np.tile([300,300,50],(21,1)))[0])

class PredictedCropTests(unittest.TestCase):
    def test_recenters_contains_points_and_preserves_scale_both_sides(self):
        points=np.array([[i%5*15-25,i//5*13-20,400+i*2] for i in range(21)],float);before=points.copy()
        for side in (0,1):
            crop,info=StereoCrop.from_predicted_points(Fisheye(),points,side)
            encoded=crop.normalized_coordinates(points)
            self.assertLess(np.max(np.abs(encoded)),.59);self.assertLessEqual(info['iterations'],8)
            rays=crop.rays((encoded+1)*64);rays/=np.linalg.norm(rays,axis=1)[:,None]
            np.testing.assert_allclose(rays,points/np.linalg.norm(points,axis=1)[:,None],atol=1e-9)
            scaled,_=StereoCrop.from_predicted_points(Fisheye(),points*2,side)
            np.testing.assert_allclose(crop.rotation,scaled.rotation,atol=1e-12);self.assertAlmostEqual(crop.radius,scaled.radius)
        np.testing.assert_array_equal(before,points)
    def test_explicit_current_seed_preserves_prediction_geometry_and_missing_joint_bounds(self):
        points=np.array([[i%5*15-25,i//5*13-20,400+i*2] for i in range(21)],float)
        points[1]=points[0] # Bounds-only duplicate for a missing external CMC.
        for side in (0,1):
            seed,info=StereoCrop.from_hand_geometry(Fisheye(),points,side)
            predicted,old=StereoCrop.from_predicted_points(Fisheye(),points,side)
            np.testing.assert_array_equal(seed.rotation,predicted.rotation)
            self.assertEqual(seed.radius,predicted.radius);self.assertEqual(info,old)
            # Replacing that duplicate by an interior wrist/middle interpolation
            # must not enlarge bounds, affect palm size or invent a new extreme.
            interior=points.copy();interior[1]=(points[0]+points[9])*.5
            alternate,_=StereoCrop.from_hand_geometry(Fisheye(),interior,side)
            np.testing.assert_array_equal(seed.rotation,alternate.rotation)
            self.assertEqual(seed.radius,alternate.radius)
    def test_ray_seed_does_not_bypass_forward_hemisphere_guard(self):
        points=np.array([[350+i%5*15,i//5*100-200,20] for i in range(21)],float)
        crop,_=StereoCrop.from_hand_geometry(Fisheye(),points,0)
        with self.assertRaisesRegex(ValueError,'forward hemisphere'):
            crop.sample(np.ones((480,640),np.uint8))
    def test_compact_fingers_activate_native_palm_size_floor(self):
        points=np.tile([0.,0.,400.],(21,1));points[0,0]=-30;points[9,0]=30
        crop,info=StereoCrop.from_predicted_points(Fisheye(),points,0)
        self.assertTrue(info['palm_floor_applied']);self.assertEqual(crop.radius,info['palm_radius_floor'])
        self.assertGreater(crop.radius,info['unexpanded_radius']*1.7)
    def test_missing_degenerate_or_behind_points_reject_crop(self):
        for points in [np.zeros((21,3)),np.full((21,3),np.nan),np.tile([0,0,-100],(21,1)),np.tile([0,0,100],(21,1)),np.ones((20,3))]:
            with self.assertRaises(ValueError):StereoCrop.from_predicted_points(Fisheye(),points,0)

if __name__=='__main__':unittest.main()
