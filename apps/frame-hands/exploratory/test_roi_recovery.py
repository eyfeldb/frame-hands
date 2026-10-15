"""Recovery lifecycle invariants and competing-hand ownership."""
import unittest
import numpy as np
from roi_recovery import RecoveryState,exclusive_support


def hand(side=0,identity=1):return dict(side=side,source_id=identity,source_views=[dict(camera="slam_left"),dict(camera="slam_right")],joint_angles=[0.]*22,wrist_world_mm=np.eye(4).tolist())

class RecoveryStateTests(unittest.TestCase):
    def test_current_anchors_override_and_reset_memory(self):
        s=RecoveryState();h=hand();self.assertFalse(s.begin(1,True,[h]))
        c=s.begin(30_000_001,True,[]);self.assertIn(0,c)
        s.finish([h],{0});self.assertGreater(s.slots[0].prediction_ns,0)
        s.begin(60_000_001,True,[hand(identity=2)])
        self.assertEqual(s.slots[0].prediction_ns,0);self.assertEqual(s.slots[0].hand['source_id'],2)
        h['source_id']=99;self.assertEqual(s.slots[0].hand['source_id'],2)

    def test_unsupported_proposals_cannot_renew_expiry(self):
        s=RecoveryState();h=hand();s.begin(1,True,[h])
        for i in range(1,23):
            c=s.begin(1+i*30_000_000,True,[])
            if c:s.finish([h],set())
        self.assertFalse(s.slots)

    def test_supported_proposals_still_need_periodic_detector_anchor(self):
        s=RecoveryState();h=hand();s.begin(1,True,[h])
        for i in range(1,69):
            c=s.begin(1+i*30_000_000,True,[])
            if c:s.finish([h],{0})
        self.assertFalse(s.slots)

    def test_invalid_context_gap_and_fresh_reseed(self):
        s=RecoveryState();h=hand();s.begin(1,True,[h]);s.begin(30_000_001,False,[])
        self.assertFalse(s.begin(60_000_001,True,[]))
        s.begin(90_000_001,True,[h]);self.assertIn(0,s.slots)
        self.assertFalse(s.begin(200_000_002,True,[]))
        s.begin(230_000_002,True,[h]);self.assertIn(0,s.slots)
        with self.assertRaises(ValueError):s.begin(230_000_002,True,[])

    def test_proposals_cannot_birth_or_change_generation(self):
        s=RecoveryState();s.begin(1,True,[])
        with self.assertRaises(ValueError):s.finish([hand()],{0})
        s.begin(2,True,[hand()])
        with self.assertRaises(ValueError):s.finish([hand(identity=2)],{0})

    def test_monocular_birth_cannot_establish_or_replace_recovery_lineage(self):
        s=RecoveryState();mono=hand();mono['source_views']=mono['source_views'][:1]
        s.begin(1,True,[mono]);self.assertFalse(s.slots)
        s.begin(2,True,[hand()]);s.begin(3,True,[mono]);self.assertIn(0,s.slots)
        mono['source_id']=2;s.begin(4,True,[mono]);self.assertFalse(s.slots)

    def test_other_hand_and_tie_block_ownership(self):
        points=np.ones((21,2))*20
        row=dict(camera='left',presence=.9,normalized_disagreement=.03,model_points=points.tolist())
        own={'left':points+2};far={'left':points+30};near={'left':points+1}
        self.assertTrue(exclusive_support([row],own,[far]))
        self.assertFalse(exclusive_support([row],own,[near]))
        self.assertFalse(exclusive_support([row],own,[own]))

class PoseGeometryTests(unittest.TestCase):
    def test_same_capture_reuse_preserves_camera_and_pixels(self):
        import torch
        import test_chirality as fixtures
        from probe_roi_continuation import pose_crop
        from frame_data import warp
        fixtures.ChiralityTests.setUpClass();f=fixtures.ChiralityTests;b=f.model.backend
        from lib.tracker.perspective_crop import _get_crop_points_from_hand_pose
        from lib.tracker.tracking_result import SingleHandPose
        output=b.predict(f.observation,{},f.world,f.prepared)
        self.assertEqual(len(output['hands']),2)
        generator=np.random.default_rng(721)
        checked=0
        for h in output['hands']:
            for turn in range(4):
                modified=dict(h,joint_angles=(np.asarray(h['joint_angles'])+generator.normal(0,.1,22)).tolist())
                prior=SingleHandPose(np.array(modified['joint_angles'],np.float32),np.array(h['wrist_world_mm'],np.float32),hand_confidence=0.)
                with torch.inference_mode():points=_get_crop_points_from_hand_pose(b.hand,prior,h['side'],63)
                for base in b.cameras.values():
                    cam=base.copy(camera_to_world_xf=f.world@base.camera_to_world_xf)
                    a=pose_crop(b,modified,cam,f.world);c=pose_crop(b,modified,cam,f.world,points)
                    self.assertEqual(a is None,c is None)
                    if a is None:continue
                    checked+=1
                    np.testing.assert_array_equal(a.camera_to_world_xf,c.camera_to_world_xf)
                    np.testing.assert_array_equal(a.uv_to_window_matrix(),c.uv_to_window_matrix())
                    pixels=generator.integers(0,256,(cam.height,cam.width),dtype=np.uint8)
                    np.testing.assert_array_equal(warp(cam,a,pixels),warp(cam,c,pixels))
        self.assertGreaterEqual(checked,8)
        with self.assertRaises(ValueError):pose_crop(b,h,cam,f.world,np.zeros((21,3)))


if __name__=='__main__':unittest.main()
