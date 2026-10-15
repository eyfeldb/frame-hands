import json,sys,unittest
from pathlib import Path
import numpy as np
import torch
from finger_refinement import LandmarkBatch,fit_fingers
from replay_umetrack import COMMON


class Camera:
    def __init__(self,x):self.x=x
    def world_to_eye(self,xyz):return np.asarray(xyz)-[self.x,0,0]
    def eye_to_window(self,xyz):return np.asarray(xyz)[...,:2]/np.asarray(xyz)[...,2:]*500+320


class FingerRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        upstream=Path(__file__).resolve().parents[3]/'.local/hand-alternatives/UmeTrack'
        sys.path.insert(0,str(upstream))
        from lib.common.hand import HandModel
        raw=json.loads((upstream/'dataset/generic_hand_model.json').read_text())
        cls.hand=HandModel(**{k:torch.tensor(v) if isinstance(v,list) else v for k,v in raw.items()})
        torch.set_num_threads(4)

    def fixture(self,side=0):
        angles=np.asarray(self.hand.joint_limits).mean(1);angles[20:]=0
        wrist=np.eye(4);wrist[:3,3]=[0,-30,450]
        return dict(side=side,source_id=11,joint_angles=angles.tolist(),wrist_world_mm=wrist.tolist())

    def observations(self,h,angles):
        xyz=LandmarkBatch(self.hand,h['wrist_world_mm'],h['side'])(np.asarray(angles)[None])[0]
        cameras={'left':Camera(-30),'right':Camera(30)};views=[]
        for name,cam in cameras.items():
            points=np.zeros((21,2));uv=cam.eye_to_window(cam.world_to_eye(xyz))
            for native,mp in COMMON:points[mp]=uv[native]
            views.append(dict(camera=name,presence=.99,model_points=points.tolist()))
        return cameras,views

    def test_batched_double_skinning_matches_native_for_both_sides(self):
        from lib.common.hand_skinning import skin_landmarks
        for side in [0,1]:
            h=self.fixture(side);angles=np.asarray(h['joint_angles']);wrist=torch.tensor(h['wrist_world_mm'],dtype=torch.float32)
            if side:wrist[:,0]*=-1
            expected=skin_landmarks(self.hand,torch.tensor(angles,dtype=torch.float32),wrist).numpy()
            result=LandmarkBatch(self.hand,h['wrist_world_mm'],side)(np.stack([angles,angles]))
            np.testing.assert_allclose(result,[expected,expected],atol=.00015,rtol=1e-6)

    def test_known_finger_error_improves_without_moving_palm_or_tail_angles(self):
        for side in [0,1]:
            h=self.fixture(side);angles=np.asarray(h['joint_angles']);target=angles.copy();target[[6,10,14]]+=.15
            cameras,views=self.observations(h,target)
            updated,debug=fit_fingers(self.hand,h,cameras,views)
            self.assertIsNotNone(updated,debug)
            self.assertLess(max(debug['after_mean_px']),max(debug['before_mean_px']))
            self.assertLessEqual(debug['max_angle_change_rad'],.35000001)
            np.testing.assert_array_equal(updated[20:],angles[20:])
            limits=np.asarray(self.hand.joint_limits)
            self.assertTrue(np.all(updated>=limits[:,0]) and np.all(updated<=limits[:,1]))
            xyz=LandmarkBatch(self.hand,h['wrist_world_mm'],side)(np.stack([angles,updated]))
            np.testing.assert_allclose(xyz[0,[5,8,11,14,17]],xyz[1,[5,8,11,14,17]],atol=1e-8)

    def test_matching_observations_do_not_change_pose(self):
        h=self.fixture();cameras,views=self.observations(h,h['joint_angles'])
        result,debug=fit_fingers(self.hand,h,cameras,views)
        self.assertIsNone(result,debug)

    def test_missing_or_weak_stereo_is_not_fit(self):
        h=self.fixture();cameras,views=self.observations(h,h['joint_angles'])
        self.assertIsNone(fit_fingers(self.hand,h,cameras,views[:1])[0])
        views[0]['presence']=.1
        self.assertIsNone(fit_fingers(self.hand,h,cameras,views)[0])


if __name__=='__main__':unittest.main()
