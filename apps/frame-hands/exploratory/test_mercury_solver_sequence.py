"""Causal replay reset and optional native history-transport checks."""
from pathlib import Path
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from mercury_native_solver import NativeSolver,ML_TO_XR
from replay_mercury_solver import reset_reason,to_world
from test_mercury_native_solver import POSE,synthetic_views

DLL=Path(__file__).resolve().parents[3]/'.local/hand-alternatives/mercury-solver-head-transport-v3/mercury_solver.dll'


class ReplayHistoryTests(unittest.TestCase):
    def test_invalid_noncausal_gapped_and_changed_generation_reset(self):
        previous=(100000000,(0,1))
        self.assertIsNone(reset_reason(previous,133000000,(0,1)))
        for stamp,generation,valid in [(100000000,(0,1),True),(99999999,(0,1),True),(200000001,(0,1),True),(133000000,(0,2),True),(133000000,(0,1),False)]:
            self.assertIsNotNone(reset_reason(previous,stamp,generation,valid))
        self.assertIsNotNone(reset_reason(None,133000000,(0,1)))
    def test_world_joint_conversion_keeps_metre_and_millimetre_origins(self):
        points=np.arange(78).reshape(26,3)/1000;head=np.eye(4);head[:3,3]=[.02,.03,.04]
        world=np.eye(4);world[:3,:3]=Rotation.from_euler('z',90,degrees=True).as_matrix();world[:3,3]=[100,200,300]
        result=to_world(dict(joints_left_camera_m=points.tolist()),head,world)
        expected=((points[ML_TO_XR]+[.02,.03,.04])*1000)@world[:3,:3].T+world[:3,3]
        np.testing.assert_allclose(result,expected)


@unittest.skipUnless(DLL.exists(),'Optional transported-history native DLL is not built')
class NativeTransportTests(unittest.TestCase):
    def test_identity_is_exact_noop_and_inverse_preserves_both_sides(self):
        for side in (0,1):
            views=synthetic_views(side)
            with NativeSolver(DLL,POSE,side) as control,NativeSolver(DLL,POSE,side) as candidate:
                for index in range(12):
                    if index:candidate.transport(np.eye(4))
                    a=control.run(views,depth_weight=0);b=candidate.run(views,depth_weight=0)
                    np.testing.assert_array_equal(a['joints_left_camera_m'],b['joints_left_camera_m'])
                transform=np.eye(4);transform[:3,:3]=Rotation.from_euler('xyz',[13,-27,7],degrees=True).as_matrix();transform[:3,3]=[.02,-.03,.04]
                before=candidate.history().copy();candidate.transport(transform);after=candidate.history()
                np.testing.assert_allclose(after[:,:3],before[:,:3]@transform[:3,:3].T+transform[:3,3],atol=1e-7)
                mirror=np.diag([-1,1,1]) if side else np.eye(3)
                expected=Rotation.from_matrix(mirror@transform[:3,:3]@mirror)*Rotation.from_quat(before[:,3:])
                np.testing.assert_allclose(Rotation.from_quat(after[:,3:]).as_matrix(),expected.as_matrix(),atol=2e-7)
                candidate.transport(np.linalg.inv(transform));returned=candidate.history()
                np.testing.assert_allclose(returned[:,:3],before[:,:3],atol=1e-7)
                np.testing.assert_allclose(Rotation.from_quat(returned[:,3:]).as_matrix(),Rotation.from_quat(before[:,3:]).as_matrix(),atol=2e-7)
                # Do not claim post-fit bitwise parity from a float roundtrip:
                # native iteration can amplify tiny initialization differences.
    def test_transport_requires_history_and_proper_rigid_transform(self):
        with NativeSolver(DLL,POSE,0) as solver:
            with self.assertRaises(ValueError):solver.transport(np.eye(4))
            solver.run(synthetic_views(0))
            for bad in [np.full((4,4),np.nan),np.diag([2,1,1,1]),np.diag([-1,1,1,1])]:
                with self.assertRaises(ValueError):solver.transport(bad)
        with self.assertRaises(ValueError):solver.transport(np.eye(4))

if __name__=='__main__':unittest.main()
