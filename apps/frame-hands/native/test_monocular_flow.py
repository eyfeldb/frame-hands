import unittest
import numpy as np
from probe_monocular_flow import flow_targets
from probe_monocular_bridge import continue_rigid, PALM_RIG, transform


class Camera:
    camera_to_world_xf = np.eye(4)
    def world_to_eye(self, points): return points
    def eye_to_window(self, points): return points[:, :2]/points[:, 2:]*500+256


class MotionTargets(unittest.TestCase):
    def test_head_motion_is_not_applied_twice(self):
        camera = Camera()
        prior = np.random.default_rng(17).normal(0, 20, (21, 3))
        prior[:, 2] = 500
        # Camera translates +10mm X in the world: the stationary hand moves -10px.
        current_head_from_world = np.eye(4)
        current_head_from_world[0, 3] = -10
        compensated = transform(prior, current_head_from_world)
        affine = np.array([[1, 0, -10], [0, 1, 0]])
        target = flow_targets(camera, prior, affine)
        np.testing.assert_allclose(target, camera.eye_to_window(compensated[PALM_RIG]))
        output, detail = continue_rigid(camera, compensated, target, .033, 'plane')
        self.assertTrue(detail['accepted'])
        np.testing.assert_allclose(output, compensated, atol=1e-8)

    def test_hand_motion_and_head_motion_are_separated(self):
        camera = Camera()
        prior = np.random.default_rng(18).normal(0, 20, (21, 3))
        prior[:, 2] = 500
        compensated = prior+[-10, 0, 0]
        # Net image motion +5px: -10px camera motion plus +15px hand motion.
        target = flow_targets(camera, prior, [[1, 0, 5], [0, 1, 0]])
        output, detail = continue_rigid(camera, compensated, target, .067, 'plane')
        self.assertTrue(detail['accepted'])
        np.testing.assert_allclose(output, prior+[5, 0, 0], atol=.2)

    def test_invalid_transform_rejected(self):
        for matrix in (np.eye(3), np.full((2, 3), np.nan)):
            with self.assertRaises(ValueError):
                flow_targets(Camera(), np.zeros((21, 3)), matrix)


if __name__ == '__main__': unittest.main()
