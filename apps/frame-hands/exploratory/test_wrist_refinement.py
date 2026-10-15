import unittest
import numpy as np
from wrist_refinement import fit_wrist,apply_correction,transform,PALM_NATIVE,PALM_DETECTOR

class Camera:
    def __init__(self,x):self.x=x
    def world_to_eye(self,p):return np.asarray(p)-np.array([self.x,0,0])
    def eye_to_window(self,p):return np.asarray(p)[...,:2]/np.asarray(p)[...,2:]*500+512

class WristRefinementTests(unittest.TestCase):
    def setUp(self):
        self.cameras={'left':Camera(-35),'right':Camera(35)}
        self.wrist=np.eye(4);self.wrist[:3,3]=[0,0,450]
        self.xyz=np.tile([0.,0.,450.],(21,1))
        self.xyz[PALM_NATIVE]=np.array([[0,0,450],[-30,60,445],[-10,65,449],[12,60,453],[30,48,452]])
    def observations(self,xyz):
        views=[]
        for name,cam in self.cameras.items():
            points=np.zeros((21,2));points[PALM_DETECTOR]=cam.eye_to_window(cam.world_to_eye(xyz[PALM_NATIVE]))
            views.append(dict(camera=name,points=points,presence=1.))
        return views
    def test_small_known_shift_recovers_without_changing_shape(self):
        expected=np.array([.6,-.4,.3,.1,-.1,.1])
        target=transform(self.xyz,self.wrist[:3,3],expected)
        x,debug=fit_wrist(self.xyz,self.wrist,self.cameras,self.observations(target))
        self.assertIsNotNone(x,debug)
        corrected=transform(self.xyz,self.wrist[:3,3],x)
        self.assertLess(np.linalg.norm(corrected-target,axis=1).mean(),1.)
        np.testing.assert_allclose(np.linalg.norm(corrected[:,None]-corrected[None,:],axis=2),np.linalg.norm(self.xyz[:,None]-self.xyz[None,:],axis=2),atol=1e-10)
    def test_exact_pose_is_unchanged(self):
        x,debug=fit_wrist(self.xyz,self.wrist,self.cameras,self.observations(self.xyz))
        self.assertIsNone(x)
    def test_monocular_duplicate_and_invalid_observations_abstain(self):
        views=self.observations(self.xyz)
        for v in (views[:1],[views[0],views[0]]):
            self.assertIsNone(fit_wrist(self.xyz,self.wrist,self.cameras,v)[0])
        views[1]['points'][0,0]=np.nan
        self.assertIsNone(fit_wrist(self.xyz,self.wrist,self.cameras,views)[0])
    def test_behind_camera_rejected(self):
        xyz=self.xyz.copy();xyz[:,2]=-100
        self.assertIsNone(fit_wrist(xyz,self.wrist,self.cameras,self.observations(self.xyz))[0])
    def test_large_requested_displacement_abstains(self):
        target=self.xyz+np.array([100.,0.,0.])
        x,debug=fit_wrist(self.xyz,self.wrist,self.cameras,self.observations(target))
        self.assertIsNone(x,debug)

    def test_actual_native_mesh_matches_reskinning_for_both_sides(self):
        import json,sys,torch
        from pathlib import Path
        from scipy.spatial.transform import Rotation
        root=Path(__file__).resolve().parents[3]/'.local/hand-alternatives/UmeTrack'
        sys.path.insert(0,str(root))
        from lib.common.hand import HandModel
        from lib.common.hand_skinning import _skin_points
        hand=HandModel(**{k:torch.tensor(v) if isinstance(v,list) else v for k,v in json.loads((root/'dataset/generic_hand_model.json').read_text()).items()})
        angles=torch.linspace(-.1,.2,22);wrist=self.wrist.copy();wrist[:3,:3]=Rotation.from_rotvec([.3,-.2,.1]).as_matrix()
        def skin(xf,side):
            x=torch.tensor(xf,dtype=torch.float32)
            if side:x[:,0]*=-1
            return _skin_points(hand.joint_rest_positions,hand.joint_rotation_axes,hand.dense_bone_weights.unsqueeze(0),angles,hand.mesh_vertices,x).numpy()
        for side in (0,1):
            mesh=skin(wrist,side)
            h=dict(side=side,source_id=1,wrist_world_mm=wrist.tolist(),joint_angles=angles.tolist(),vertices_world_mm=mesh.tolist())
            corrected=apply_correction(h,[.3,-.4,.6,-.2,.5,.7])
            np.testing.assert_allclose(corrected['vertices_world_mm'],skin(corrected['wrist_world_mm'],side),atol=.0002)

    def test_apply_preserves_joint_angles_side_and_reflection(self):
        for side in (0,1):
            h=dict(side=side,source_id=42,joint_angles=[.1]*22,wrist_world_mm=self.wrist.tolist(),vertices_world_mm=self.xyz.tolist())
            p=np.array([.2,.3,-.1,.1,.05,.1]);out=apply_correction(h,p)
            self.assertEqual(out['joint_angles'],h['joint_angles']);self.assertEqual(out['side'],side);self.assertEqual(out['source_id'],42)
            np.testing.assert_allclose(np.linalg.det(np.array(out['wrist_world_mm'])[:3,:3]),1.,atol=1e-12)
            np.testing.assert_allclose(out['vertices_world_mm'],transform(self.xyz,self.wrist[:3,3],p))

if __name__=='__main__':unittest.main()
