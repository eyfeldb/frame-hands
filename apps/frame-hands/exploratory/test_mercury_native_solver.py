"""Coordinate and native-solver contract checks; synthetic inputs contain no images."""
from pathlib import Path
import unittest
import numpy as np
from mercury_native_solver import NativeSolver, camera_transforms, decode_depth, pack_views, ML_TO_XR, FLIP

DLL=Path(__file__).resolve().parents[3]/'.local/hand-alternatives/mercury-solver/mercury_solver.dll'
POSE=np.array([0,0,0,1,-.06,0,0],np.float32)


def synthetic_views(side):
    points=[[0,0,-.45]]
    for finger in range(5):
        x=(finger-2)*.02
        for joint in range(4):points.append([x,.035+joint*.022,-.45])
    return observations_from_joints(np.asarray(points),side)


def observations_from_joints(joints,side):
    views={}
    for name,shift in [('slam_left',np.zeros(3)),('slam_right',POSE[4:])]:
        rays=joints+shift;rays=rays/np.linalg.norm(rays,axis=1)[:,None]
        xy=rays[:,:2]/(1-rays[:,2,None])/.4;xy[:,1]*=-1
        if side:xy[:,0]*=-1
        views[name]=dict(active=True,rotation=np.eye(3).tolist(),stereographic_radius=.4,points_crop=((xy+1)*64).tolist(),confidence_xy=[1.]*21,depth_heatmap=np.zeros((21,22)).tolist(),curls=[0.]*10)
    return views


class NativeCoordinateTests(unittest.TestCase):
    def test_stereo_translation_and_metre_conversion(self):
        class Camera:pass
        cams={name:Camera() for name in ['slam_left','slam_right']}
        for cam in cams.values():cam.camera_to_world_xf=np.eye(4)
        cams['slam_right'].camera_to_world_xf[0,3]=60
        pose,head=camera_transforms(cams)
        np.testing.assert_allclose(pose,POSE,atol=1e-7);np.testing.assert_allclose(head,FLIP)
    def test_packet_unmirrors_both_anatomical_sides(self):
        np.testing.assert_allclose(pack_views(synthetic_views(0),0),pack_views(synthetic_views(1),1),atol=1e-6)
    def test_depth_invalid_and_zero_evidence(self):
        depth,confidence=decode_depth(np.zeros((21,22)));np.testing.assert_array_equal(depth,0);np.testing.assert_array_equal(confidence,0)
        heat=np.zeros((21,22));heat[:,11]=.8;depth,confidence=decode_depth(heat)
        np.testing.assert_allclose(depth,(11.5/22-.5)*3,atol=1e-6);np.testing.assert_allclose(confidence,.8)
        with self.assertRaises(ValueError):decode_depth(np.full((21,22),np.nan))


@unittest.skipUnless(DLL.exists(),'Optional offline native solver is not built')
class NativeContractTests(unittest.TestCase):
    def test_finite_fixed_geometry_reset_and_projection_roundtrip(self):
        for side in (0,1):
            views=synthetic_views(side)
            with NativeSolver(DLL,POSE,side) as solver:
                first=solver.run(views,depth_weight=0)
                joints=np.asarray(first['joints_left_camera_m'])
                self.assertTrue(np.isfinite(joints).all())
                np.testing.assert_allclose(np.linalg.norm(first['orientations_xyzw'],axis=1),1,atol=2e-5)
                self.assertAlmostEqual(first['hand_size_m'],.09,places=6)
                solver.run(views,depth_weight=0)
                reset=solver.run(views,reset=True,depth_weight=0)
                np.testing.assert_array_equal(reset['joints_left_camera_m'],joints)
                # This tests geometry consistency, not independent physical accuracy.
                consistent=observations_from_joints(joints[ML_TO_XR],side)
                # Native has a bounded per-call iteration budget; cold initialization
                # can miss the exact synthetic solution on its first call.
                for index in range(8):
                    fresh=solver.run(consistent,reset=index==0,smoothing=0,depth_weight=0)
                self.assertLess(fresh['reprojection_error'],.001)
                again=np.asarray(fresh['joints_left_camera_m'])
                # Every articulated bone preserves its length across different fits.
                edges=[(a,a+1) for lo,hi in [(2,5),(6,10),(11,15),(16,20),(21,25)] for a in range(lo,hi)]
                lengths=lambda points:np.array([np.linalg.norm(points[a]-points[b]) for a,b in edges])
                np.testing.assert_allclose(lengths(joints),lengths(again),atol=2e-7)
    def test_reject_absent_negative_and_closed_inputs(self):
        for kind in ['absent','confidence','radius']:
            views=synthetic_views(0)
            if kind=='absent':
                for view in views.values():view['active']=False
            if kind=='confidence':views['slam_left']['confidence_xy'][0]=-1
            if kind=='radius':views['slam_left']['stereographic_radius']=0
            with NativeSolver(DLL,POSE,0) as solver:
                with self.assertRaises(ValueError):solver.run(views)
        solver=NativeSolver(DLL,POSE,0);solver.close()
        with self.assertRaises(ValueError):solver.run(synthetic_views(0))

if __name__=='__main__':unittest.main()
