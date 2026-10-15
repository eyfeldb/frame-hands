from copy import deepcopy
from pathlib import Path
import ctypes
import unittest
import numpy as np
from mercury_forward import NativeHandGenerator
from mercury_native_solver import NativeSolver,ML_TO_XR
from mercury_curl_init import initialize_observations
from test_mercury_native_solver import POSE,observations_from_joints

DLL=Path(__file__).resolve().parents[3]/'.local/hand-alternatives/mercury-solver-curl-init-v1/mercury_solver.dll'


@unittest.skipUnless(DLL.exists(),'Optional curl initializer unavailable')
class CurlInitializationTests(unittest.TestCase):
    def views(self,side=0):
        points=NativeHandGenerator(DLL).generate([.70710678,0,0,.70710678,.04,-.03,-.5],np.tile([-.4,-.5,-.3],(4,1)),side)[ML_TO_XR,:3]
        return observations_from_joints(points,side)
    def test_native_weighted_sum_and_bounded_minimum_change(self):
        views=self.views();views['slam_left']['curls'][1:5]=[-3,-100,100,-.5];views['slam_right']['curls'][1:5]=[-1,-100,100,-.5];views['slam_right']['curls'][6:10]=[.03]*4;original=deepcopy(views)
        with NativeSolver(DLL,POSE,0) as solver:
            result=initialize_observations(solver,views,curls=True);rows=np.asarray(result['fingers_observed_target_angles']);lower=np.deg2rad([-100,-90,-90]);upper=np.deg2rad([20,0,0]);defaults=np.deg2rad([15,-5,-5])
            np.testing.assert_allclose(rows[:,0],[-2.6,-100,100,-.5],atol=1e-6)
            np.testing.assert_allclose(rows[:,2:].sum(1),rows[:,1],atol=1e-6)
            self.assertTrue((rows[:,2:]>=lower-1e-6).all());self.assertTrue((rows[:,2:]<=upper+1e-6).all())
            np.testing.assert_allclose(rows[1,2:],lower,atol=1e-6);np.testing.assert_allclose(rows[2,2:],upper,atol=1e-6)
            for index in (0,3):
                delta=rows[index,2:]-defaults;np.testing.assert_allclose(delta,delta[0],atol=1e-6)
        self.assertEqual(original,views)
    def test_noop_matches_native_and_thumb_curl_does_not_seed(self):
        for side in (0,1):
            views=self.views(side)
            with NativeSolver(DLL,POSE,side) as control,NativeSolver(DLL,POSE,side) as manual:
                initialize_observations(manual,views)
                self.assertEqual(manual.run(views,depth_weight=0),control.run(views,depth_weight=0))
            altered=deepcopy(views)
            for v in altered.values():v['curls'][0]=9
            with NativeSolver(DLL,POSE,side) as first,NativeSolver(DLL,POSE,side) as second:
                a=initialize_observations(first,views,stereo=True,curls=True);b=initialize_observations(second,altered,stereo=True,curls=True);self.assertEqual(a,b)
                self.assertEqual(first.run(views,depth_weight=0),second.run(altered,depth_weight=0))
    def test_lifecycle_invalid_input_and_truth_api_absence(self):
        self.assertFalse(hasattr(ctypes.CDLL(str(DLL)),'fh_solver_seed_synthetic'));views=self.views()
        with NativeSolver(DLL,POSE,0) as solver:
            with self.assertRaises(ValueError):initialize_observations(solver,views,size=.5)
            self.assertTrue(solver.first)
            absent=deepcopy(views)
            for v in absent.values():v['active']=False
            with self.assertRaises(ValueError):initialize_observations(solver,absent,curls=True)
            initialize_observations(solver,views,curls=True)
            with self.assertRaises(ValueError):initialize_observations(solver,views,curls=True)
        with self.assertRaises(ValueError):initialize_observations(solver,views)


if __name__=='__main__':unittest.main()
