import unittest
import numpy as np
from skeleton_regularizer import fit
from audit_native_lengths import PARENTS
class RegularizerTests(unittest.TestCase):
    def fixture(self):
        p=np.zeros((21,3));p[:,2]=500
        for base in (1,5,9,13,17):
            p[base]=[base*3,0,500]
            for j in range(base+1,base+4):p[j]=p[j-1]+[0,20,0]
        t=np.linalg.norm(p-p[PARENTS],axis=1);c=np.tile(np.eye(3),(21,1,1));return p,c,t
    def test_exact_anatomy_and_no_calibration_are_identity(self):
        p,c,t=self.fixture();q,_=fit(p,c,t);np.testing.assert_allclose(q,p,atol=1e-9)
        q,_=fit(p,c,np.full(21,np.nan));np.testing.assert_array_equal(q,p)
    def test_directional_uncertainty_controls_correction(self):
        p,c,t=self.fixture();observed=p.copy();observed[8,1]+=12
        tight=c*.01;loose=tight.copy();loose[8,1,1]=100
        a,_=fit(observed,tight,t);b,_=fit(observed,loose,t)
        self.assertLess(np.linalg.norm(b[8]-p[8]),np.linalg.norm(a[8]-p[8]))
        self.assertLess(np.linalg.norm(b[0]-p[0]),.001)
    def test_rigid_frame_equivariance(self):
        p,c,t=self.fixture();p[8,1]+=12;c[8,1,1]=100
        a,_=fit(p,c,t);R=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
        b,_=fit(p@R.T+[10,20,30],R@c@R.T,t)
        np.testing.assert_allclose(b,a@R.T+[10,20,30],atol=1e-7)
    def test_invalid_covariance_rejected(self):
        p,c,t=self.fixture();c[0,0,0]=-1
        with self.assertRaises(ValueError):fit(p,c,t)
if __name__=='__main__':unittest.main()
