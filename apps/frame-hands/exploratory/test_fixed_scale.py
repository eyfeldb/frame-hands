import unittest
from pathlib import Path
import numpy as np,torch
from fixed_scale_backend import ScaledBackend
from live_backend import Backend

ROOT=Path(__file__).resolve().parents[3]
UPSTREAM=ROOT/'.local/hand-alternatives/UmeTrack'
SCALE=1.0407845934232076

class FixedScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original=Backend(UPSTREAM,{})
        cls.control=ScaledBackend(UPSTREAM,{},hand_scale=1.)
        cls.scaled=ScaledBackend(UPSTREAM,{},hand_scale=SCALE)
    def test_original_scale_is_exact(self):
        for key,value in self.original.hand._asdict().items():
            other=getattr(self.control.hand,key)
            if isinstance(value,torch.Tensor):self.assertTrue(torch.equal(value,other),key)
            else:self.assertEqual(value,other)
        self.assertTrue(torch.equal(self.original.skeleton.joint_rest_positions,self.control.skeleton.joint_rest_positions))
    def test_encoder_and_mesh_use_same_fixed_scale(self):
        np.testing.assert_allclose(self.scaled.skeleton.joint_rest_positions.numpy(),self.original.skeleton.joint_rest_positions.numpy()*SCALE,rtol=2e-7,atol=1e-8)
        before=self.scaled.hand.joint_rest_positions.clone()
        for side in (0,1):
            for curl in (0.,.2,.7):
                wrist=np.eye(4);wrist[:3,3]=[100,-20,400]
                angles=np.ones(22)*curl
                a=self.original.mesh(angles,wrist,side);b=self.scaled.mesh(angles,wrist,side)
                np.testing.assert_allclose(b-wrist[:3,3],(a-wrist[:3,3])*SCALE,atol=.00015)
        self.assertTrue(torch.equal(before,self.scaled.hand.joint_rest_positions))
    def test_invalid_fixed_scale_rejected_before_model_loading(self):
        for scale in (np.nan,np.inf,0.,-1.,2.):
            with self.assertRaises(ValueError):ScaledBackend('no-file',{},hand_scale=scale)

if __name__=='__main__':unittest.main()
