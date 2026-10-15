import unittest
import numpy as np
import cv2
from mercury_probe import normalize,blackbar

class MercuryPreprocessTests(unittest.TestCase):
    def test_flat_image_is_not_hand_evidence(self):
        self.assertIsNone(normalize(np.full((160,160),33,np.uint8)))
    def test_normalization_keeps_contrast_and_expected_moments(self):
        source=np.arange(256,dtype=np.uint8).reshape(16,16)
        out=normalize(source)
        self.assertAlmostEqual(float(out.mean()),.5,places=6)
        self.assertAlmostEqual(float(out.std()),.25,places=6)
        self.assertTrue(np.all(np.diff(out.ravel())>0))
    def test_cardinal_orientation_and_inverse_on_rectangular_images(self):
        for shape in [(1024,1056),(720,1280)]:
            for rotation in [0,90,180,270]:
                image=np.zeros(shape,np.uint8);small,back,forward=blackbar(image,rotation)
                pts=np.array([[200,200],[300,200],[200,300]],float)
                mapped=np.c_[pts,np.ones(3)]@forward.T
                returned=np.c_[mapped,np.ones(3)]@back.T
                np.testing.assert_allclose(returned,pts,atol=.0002)
                dx=mapped[1]-mapped[0];dy=mapped[2]-mapped[0]
                expected={0:([1,0],[0,1]),90:([0,-1],[1,0]),180:([-1,0],[0,-1]),270:([0,1],[-1,0])}[rotation]
                np.testing.assert_allclose(dx/np.linalg.norm(dx),expected[0],atol=1e-6)
                np.testing.assert_allclose(dy/np.linalg.norm(dy),expected[1],atol=1e-6)
                self.assertEqual(small.shape,(160,160))
if __name__=='__main__':unittest.main()
