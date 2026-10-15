import sys,unittest
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
sys.path[:0]=[str(BASE.parent/'exploratory'),str(BASE.parents[2]/'.local/hand-alternatives/UmeTrack')]
from stereo_uncertainty import triangulate,projection_jacobian,sensitivity
from lib.common.camera import PinholePlaneCameraModel
class StereoSensitivityTests(unittest.TestCase):
    def setup_pair(self):
        cams={}
        for i,name in enumerate(('left','right')):
            xf=np.eye(4);xf[0,3]=(-1 if i==0 else 1)*30
            cams[name]=PinholePlaneCameraModel(1000,1000,500,(500,500),[],xf)
        return cams
    def test_exact_stereo_and_depth_sensitivity(self):
        cams=self.setup_pair();points=np.array([[0.,0.,500.],[0.,0.,1000.]])
        views=[dict(camera=k,points=c.eye_to_window(c.world_to_eye(points))) for k,c in cams.items()]
        xyz,cov,info=sensitivity(cams,views)
        np.testing.assert_allclose(xyz,points,atol=1e-8)
        # Analytic rectified disparity propagation: sigmaZ=sqrt(2)*Z^2/(f*baseline).
        expected=np.sqrt(2)*points[:,2]**2/(500*60)
        np.testing.assert_allclose(np.sqrt(cov[:,2,2]),expected,rtol=1e-6)
        self.assertGreater(info['major_axis_mm_per_pixel'][1],3.9*info['major_axis_mm_per_pixel'][0])
    def test_projection_jacobian_independent_analytic(self):
        c=self.setup_pair()['left'];x=np.array([[10.,20.,500.]])
        eye=c.world_to_eye(x)[0]
        expected=np.array([[500/eye[2],0,-500*eye[0]/eye[2]**2],[0,500/eye[2],-500*eye[1]/eye[2]**2]])
        np.testing.assert_allclose(projection_jacobian(c,x)[0],expected,rtol=1e-6,atol=1e-8)
    def test_degenerate_and_behind_rejected(self):
        with self.assertRaises(ValueError):triangulate(np.zeros((2,3)),np.array([[[0,0,1]],[[0,0,1]]]))
        with self.assertRaises(ValueError):triangulate(np.array([[-30,0,0],[30,0,0]]),np.array([[[-.1,0,1]],[[.1,0,1]]]))
        with self.assertRaises(ValueError):triangulate(np.zeros((2,3)),np.zeros((2,1,3)))
if __name__=='__main__':unittest.main()
