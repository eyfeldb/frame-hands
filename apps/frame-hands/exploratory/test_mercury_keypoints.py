import unittest
import numpy as np
from mercury_keypoints import StereoCrop,align_forward,heatmap_centers

class Fisheye:
    f=np.array([200.,200.]);c=np.array([320.,240.]);distort=[0,0,0,0]
    def eye_to_window(self,rays):
        rays=np.asarray(rays);r=np.linalg.norm(rays[...,:2],axis=-1);theta=np.arctan2(r,rays[...,2])
        return rays[...,:2]*np.divide(theta,r,out=np.ones_like(r),where=r>1e-10)[...,None]*self.f+self.c

class MercuryCropTests(unittest.TestCase):
    def test_ray_alignment_and_crop_center(self):
        for center in [[320,240],[400,190],[210,280]]:
            for side in [0,1]:
                crop=StereoCrop(Fisheye(),center,110,side)
                np.testing.assert_allclose(crop.pixels([[64,64]])[0],center,atol=1e-7)
                np.testing.assert_allclose(crop.rotation.T@crop.rotation,np.eye(3),atol=1e-9)
                self.assertAlmostEqual(np.linalg.det(crop.rotation),1.)
    def test_side_mirror_uses128_coordinate_extent(self):
        left=StereoCrop(Fisheye(),[330,220],120,0);right=StereoCrop(Fisheye(),[330,220],120,1)
        points=np.array([[5,17],[40,62],[111,98]],float)
        mirrored=points.copy();mirrored[:,0]=128-points[:,0]
        np.testing.assert_allclose(left.pixels(points),right.pixels(mirrored),atol=1e-8)
    def test_heatmap_peak_centres_and_empty_fallback(self):
        heat=np.zeros((21,22,22));heat[:,8,13]=.75
        points,confidence=heatmap_centers(heat)
        np.testing.assert_allclose(points,np.tile(np.array([13.5,8.5])*128/22,(21,1)))
        self.assertEqual(confidence,[.75]*21)
        empty,_=heatmap_centers(np.zeros((21,22,22)));np.testing.assert_array_equal(empty,0)
    def test_sampling_is_truncated_and_blank_remains_blank(self):
        crop=StereoCrop(Fisheye(),[330.7,220.7],120,0)
        image=np.tile(np.arange(640,dtype=np.int64)%256,(480,1)).astype(np.uint8)
        sampled,fraction=crop.sample(image)
        self.assertEqual(sampled[64,64],image[220,330]);self.assertEqual(fraction,1.)
        blank,_=crop.sample(np.zeros_like(image));self.assertFalse(blank.any())
if __name__=='__main__':unittest.main()
