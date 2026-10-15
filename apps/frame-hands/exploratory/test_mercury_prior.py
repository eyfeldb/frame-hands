import unittest
import numpy as np
from mercury_prior import EarlierPosePrior
from mercury_keypoints import StereoCrop
from test_mercury_keypoints import Fisheye

class CausalMercuryPriorTests(unittest.TestCase):
    def test_predict_uses_two_earlier_estimates_with_native_gain(self):
        p=EarlierPosePrior();a=np.zeros((21,3));b=np.ones((21,3))*10
        p.observe(100000000,7,a);self.assertIsNone(p.predict(133000000,7)[0])
        p.observe(133000000,7,b);prediction,_=p.predict(166000000,7)
        np.testing.assert_allclose(prediction,14)
        np.testing.assert_array_equal(a,0);np.testing.assert_array_equal(b,10)
    def test_invalid_expired_generation_and_duplicate_context_reset(self):
        for mode in ['invalid','expired','generation','duplicate']:
            p=EarlierPosePrior();p.observe(100000000,7,np.zeros((21,3)));p.observe(133000000,7,np.ones((21,3)))
            args=dict(stamp=166000000,generation=7)
            if mode=='invalid':args['context_valid']=False
            if mode=='expired':args['stamp']=234000000
            if mode=='generation':args['generation']=8
            if mode=='duplicate':args['stamp']=133000000
            self.assertIsNone(p.predict(**args)[0]);self.assertEqual(p.rows,[])
    def test_missing_measurement_and_long_history_gap_need_reseed(self):
        p=EarlierPosePrior();p.observe(100000000,7,np.zeros((21,3)));p.observe(133000000,7,None)
        p.observe(166000000,7,np.ones((21,3)));self.assertIsNone(p.predict(199000000,7)[0])
        p.observe(333000000,7,np.ones((21,3)));self.assertEqual(len(p.rows),1)
    def test_prior_encoding_is_inverse_crop_mapping_for_both_sides(self):
        points=np.column_stack((np.linspace(10,118,21),np.linspace(113,7,21)))
        for side in [0,1]:
            crop=StereoCrop(Fisheye(),[330,220],120,side);rays=crop.rays(points)*300
            np.testing.assert_allclose(crop.normalized_coordinates(rays),points/64-1,atol=1e-9)
if __name__=='__main__':unittest.main()
