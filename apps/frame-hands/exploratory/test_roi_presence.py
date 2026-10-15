"""Crop geometry invariance and an actual empty-image rejection control."""
import importlib.util,unittest
from pathlib import Path
import numpy as np
from probe_roi_presence import crop_matrix,PresenceModel,disagreement_details
from score_roi_presence import supported_views

class RoiPresenceTests(unittest.TestCase):
    def setUp(self):
        self.points=np.array([[100+20*(i%4),100+10*(i//4)] for i in range(21)],float)
        self.points[0]=[130,200];self.points[5]=[110,150];self.points[9]=[130,145];self.points[13]=[150,150]

    def test_upright_wrist_is_below_knuckles_and_translation_is_covariant(self):
        matrix,size=crop_matrix(self.points)
        local=(self.points-matrix[:,2])@np.linalg.inv(matrix[:,:2]).T
        self.assertGreater(local[0,1],local[9,1])
        shifted,shifted_size=crop_matrix(self.points+[37,-19])
        np.testing.assert_allclose(shifted[:,:2],matrix[:,:2],atol=1e-6)
        np.testing.assert_allclose(shifted[:,2],matrix[:,2]+[37,-19],atol=1e-5)
        self.assertAlmostEqual(size,shifted_size)

    def test_rotation_scale_covariance(self):
        angle=.7;rotate=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        matrix,size=crop_matrix(self.points)
        transformed,new_size=crop_matrix(self.points@rotate.T*1.3)
        np.testing.assert_allclose(transformed,1.3*rotate@matrix,atol=2e-5)
        self.assertAlmostEqual(new_size,size*1.3)

    def test_support_must_come_from_same_camera_and_finite_evidence(self):
        self.assertFalse(supported_views([dict(presence=.99,normalized_disagreement=.3),dict(presence=.1,normalized_disagreement=.01)]))
        self.assertFalse(supported_views([dict(presence=float('nan'),normalized_disagreement=.01)]))
        self.assertFalse(supported_views([dict(presence=.9,normalized_disagreement=float('nan'))]))
        self.assertEqual(len(supported_views([dict(presence=.9,normalized_disagreement=.05)])),1)

    def test_median_can_hide_every_wrong_fingertip(self):
        projected=np.zeros((21,2));observed=projected.copy()
        projected[1]=np.nan  # Nonhomologous thumb CMC remains excluded.
        observed[[4,8,12,16,20],0]=20.
        report=disagreement_details(observed,projected,100.)
        self.assertEqual(report['normalized_disagreement'],0.)
        self.assertEqual(report['median_palm_disagreement_fraction'],0.)
        self.assertEqual(report['tip_disagreement_fraction'],[.2]*5)
        self.assertEqual(report['max_tip_disagreement_fraction'],.2)

    @unittest.skipUnless(importlib.util.find_spec('ncnn'),'Optional ncnn research wheel absent')
    def test_real_model_blank_control(self):
        models=Path(__file__).resolve().parents[3]/'.local/hand-alternatives/results/live-mesh-20261005T050930Z/frame/snapshot/models/ncnn'
        if not (models/'hand.ncnn.bin').exists():self.skipTest('Private saved model absent')
        model=PresenceModel(models)
        points,presence,right=model.infer(np.zeros((224,224),np.uint8))
        self.assertEqual(points.shape,(21,2));self.assertTrue(np.isfinite(points).all())
        self.assertLess(presence,.5)

if __name__=='__main__':unittest.main()
