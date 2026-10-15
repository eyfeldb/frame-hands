from copy import deepcopy
from pathlib import Path
import ctypes
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from mercury_forward import NativeHandGenerator
from mercury_native_solver import NativeSolver,ML_TO_XR
from mercury_stereo_init import initialize_from_stereo
from test_mercury_native_solver import POSE,observations_from_joints

DLL=Path(__file__).resolve().parents[3]/'.local/hand-alternatives/mercury-solver-stereo-init-v1/mercury_solver.dll'


@unittest.skipUnless(DLL.exists(),'Optional stereo initializer unavailable')
class StereoInitializationTests(unittest.TestCase):
    def test_calibrated_pose_and_handed_reflection_from_observations(self):
        gen=NativeHandGenerator(DLL);angles=np.tile([-.4,-.5,-.3],(4,1))
        for side in (0,1):
            for euler in ([80,10,-15],[110,-15,20]):
                wrist=np.r_[Rotation.from_euler('xyz',euler,degrees=True).as_quat(),[.04,-.03,-.5]]
                points=gen.generate(wrist,angles,side)[ML_TO_XR,:3];rotation=Rotation.from_euler('y',8,degrees=True);pose=np.r_[rotation.as_quat(),rotation.apply([-.06,0,0])]
                views=observations_from_joints(points,side)
                right=rotation.apply(points)+pose[4:];rays=right/np.linalg.norm(right,axis=1)[:,None];xy=rays[:,:2]/(1-rays[:,2,None])/.4;xy[:,1]*=-1
                if side:xy[:,0]*=-1
                views['slam_right']['points_crop']=((xy+1)*64).tolist();original=deepcopy(views)
                with NativeSolver(DLL,pose,side) as solver:
                    result=initialize_from_stereo(solver,views);self.assertTrue(result['accepted']);estimated=np.asarray(result['wrist_internal_xyzw_xyz'])
                    np.testing.assert_allclose(estimated[4:],wrist[4:],atol=1e-6,rtol=0)
                    self.assertLess((Rotation.from_quat(estimated[:4]).inv()*Rotation.from_quat(wrist[:4])).magnitude(),1e-5)
                    with self.assertRaises(ValueError):initialize_from_stereo(solver,views)
                self.assertEqual(views,original)
    def test_rejections_preserve_default_fit_and_reset(self):
        gen=NativeHandGenerator(DLL);wrist=[.70710678,0,0,.70710678,.04,-.03,-.5];angles=np.tile([-.1,-.05,-.05],(4,1));views=observations_from_joints(gen.generate(wrist,angles,0)[ML_TO_XR,:3],0)
        variants=[]
        missing=deepcopy(views);missing['slam_right']['active']=False;variants.append((missing,'missing_view'))
        weak=deepcopy(views);weak['slam_left']['confidence_xy'][0]=.1;variants.append((weak,'low_palm_confidence'))
        parallel=deepcopy(views);parallel['slam_right']['points_crop']=deepcopy(parallel['slam_left']['points_crop']);variants.append((parallel,'parallel_rays'))
        behind=deepcopy(views);behind['slam_left']['points_crop'],behind['slam_right']['points_crop']=behind['slam_right']['points_crop'],behind['slam_left']['points_crop'];variants.append((behind,'depth_or_cheirality'))
        for values,reason in variants:
            with NativeSolver(DLL,POSE,0) as solver,NativeSolver(DLL,POSE,0) as control:
                result=initialize_from_stereo(solver,values);self.assertFalse(result['accepted']);self.assertEqual(result['reason'],reason);self.assertTrue(solver.first)
                self.assertEqual(solver.run(values,depth_weight=0),control.run(values,depth_weight=0))
        with NativeSolver(DLL,POSE,0) as solver,NativeSolver(DLL,POSE,0) as control:
            self.assertTrue(initialize_from_stereo(solver,views)['accepted']);self.assertEqual(solver.run(views,reset=True,depth_weight=0),control.run(views,depth_weight=0))
    def test_malformed_closed_and_truth_seed_absent(self):
        self.assertFalse(hasattr(ctypes.CDLL(str(DLL)),'fh_solver_seed_synthetic'))
        views=observations_from_joints(NativeHandGenerator(DLL).generate([0,0,0,1,0,0,-.5],np.zeros((4,3)),0)[ML_TO_XR,:3],0)
        with NativeSolver(DLL,POSE,0) as solver:
            with self.assertRaises(ValueError):initialize_from_stereo(solver,views,size=.5)
            self.assertTrue(solver.first)
            invalid=deepcopy(views);invalid['slam_left']['points_crop'][0][0]=float('nan')
            with self.assertRaises(ValueError):initialize_from_stereo(solver,invalid)
        with self.assertRaises(ValueError):initialize_from_stereo(solver,views)


if __name__=='__main__':unittest.main()
