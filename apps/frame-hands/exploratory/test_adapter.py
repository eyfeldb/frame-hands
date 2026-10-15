"""Geometry, time alignment and identity regressions for the offline adapter."""
import json
import tempfile
from pathlib import Path
import struct
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from frame_data import Recording, SET, CAM, read_audit, Ownership, load_calibration, unproject, crop_camera, capture_poses
from rotations import so3_exp_map

UPSTREAM = Path(__file__).resolve().parents[3]/'.local/hand-alternatives/UmeTrack'
sys.path.insert(0, str(UPSTREAM))


class GeometryTests(unittest.TestCase):
    def test_rodrigues_against_independent_scipy(self):
        vectors = np.vstack((np.zeros((1, 3)), [[1e-12, -1e-12, 0], [np.pi, 0, 0]],
                             np.random.default_rng(15).normal(size=(200, 3))))
        result = so3_exp_map(torch.tensor(vectors)).numpy()
        np.testing.assert_allclose(result, Rotation.from_rotvec(vectors).as_matrix(), atol=1e-12)

    def make_camera(self):
        from lib.common.camera import Fisheye62CameraModel
        return Fisheye62CameraModel(1056, 1024, (329., 327.), (528., 512.),
                                    (.035, -.0096, -.003, -.00017, 0, 0, 0, 0), np.eye(4))

    def test_fisheye_inverse_and_opencv_independent_projection(self):
        import cv2
        camera = self.make_camera()
        pixels = np.array([[528, 512], [120, 240], [900, 760], [520, 950]])
        rays = unproject(camera, pixels)
        np.testing.assert_allclose(camera.eye_to_window(rays), pixels, atol=1e-8)
        k = np.array([[329., 0, 528.], [0, 327., 512.], [0, 0, 1.]])
        result, _ = cv2.fisheye.projectPoints(rays.reshape(1, -1, 3), np.zeros(3), np.zeros(3), k,
                                            np.array(camera.distort[:4]))
        np.testing.assert_allclose(result[0], pixels, atol=1e-8)

    def test_crop_mirror_and_roundtrip(self):
        camera = self.make_camera()
        points = np.array([[480., 380.], [500, 430], [530, 470], [570, 390]])
        world = camera.eye_to_world(unproject(camera, points)*500)
        for side in (0, 1):
            crop = crop_camera(camera, points, side, np.eye(4))
            self.assertAlmostEqual(np.linalg.det(crop.camera_to_world_xf[:3, :3]), -1 if side else 1)
            pixels = crop.eye_to_window(crop.world_to_eye(world))
            self.assertTrue(np.all(pixels >= 0) and np.all(pixels <= 95))
            back = crop.eye_to_world(crop.window_to_eye(pixels))
            reprojection = camera.eye_to_window(camera.world_to_eye(back))
            np.testing.assert_allclose(reprojection, points, atol=1e-8)

    def test_factory_units_and_axes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = dict(plus_x=[1, 0, 0], plus_z=[0, 0, 1], position=[0, 0, 0])
            device = dict(head=dict(identity, position=[.01, 0, 0]), cv=dict(cad_from_cal=dict(identity, position=[.1, 0, 0])))
            kb = dict(cameraModel='kb', fx=300, fy=300, cx=528, cy=512, k1=0, k2=0, k3=0, k4=0)
            cams = [dict(sourceCamera=n, width=1056, height=1024, intrinsics=[kb],
                         extrinsics=dict(identity, position=[i*60, 0, 0])) for i, n in enumerate(('slam_left', 'slam_right'))]
            (root/'device_config.json').write_text(json.dumps(device))
            (root/'xrservice.json').write_text(json.dumps(dict(cameras=cams)))
            cameras = load_calibration(root)
            self.assertAlmostEqual(cameras['slam_left'].camera_to_world_xf[0, 3], 90)
            self.assertAlmostEqual(cameras['slam_right'].camera_to_world_xf[0, 3], 150)


class RecordTests(unittest.TestCase):
    def test_recording_truncation_and_exact_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'sets.bin'
            data = SET.pack(b'FHSET01\0', 1, SET.size+CAM.size+4)+CAM.pack(b'slam_left', 2, 2, 100, 110)+bytes([1, 2, 3, 4])
            path.write_bytes(data)
            rows = list(Recording(path).frames(0, 1))
            np.testing.assert_array_equal(rows[0][2]['slam_left'][0], [[1, 2], [3, 4]])
            for damaged in (data[:-1], data+data, data[:12]):
                path.write_bytes(damaged)
                with self.assertRaises(ValueError):
                    Recording(path)

    def test_audit_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'audit.txt'
            path.write_text('set 1 2 0\n')
            with self.assertRaises(ValueError):
                read_audit(path)

    def test_clock_domains_join_through_exact_packet(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            relay = bytearray(3280)
            struct.pack_into('<8sII', relay, 0, b'FHRELAY1', 1, 3280)
            relay[40:48] = b'FHREAD1\0'
            relay[2712:2720] = b'FHPCVR1\0'
            struct.pack_into('<Q', relay, 80, 1234567890123)
            struct.pack_into('<Q', relay, 2728, 51)
            (path/'relay-trace.bin').write_bytes(relay)
            trace = bytearray(3136)
            struct.pack_into('<8sII', trace, 0, b'FHTRACE1', 1, 3136)
            struct.pack_into('<I', trace, 32, 3)
            struct.pack_into('<Q', trace, 720, 51)
            xf = np.eye(4)[:3].copy()
            xf[:, 3] = [1, 2, 3]
            struct.pack_into('<12f', trace, 88, *xf.flatten())
            (path/'sampler-trace.bin').write_bytes(trace)
            poses = capture_poses(path)
            np.testing.assert_array_equal(poses[1234567890123][:3, 3], [1000, 2000, 3000])
            struct.pack_into('<Q', trace, 720, 52)
            (path/'sampler-trace.bin').write_bytes(trace)
            self.assertEqual(capture_poses(path), {})


class IdentityTests(unittest.TestCase):
    def observation(self, a=.1, b=.9, stamp=1):
        return dict(raw_ns=stamp, hands={12:dict(right=.374), 13:dict(right=.344)},
                    views=[dict(id=i, camera=c, presence=.93, right=q, points=np.zeros((21, 2)))
                           for i, q in ((12, a), (13, b)) for c in ('slam_left', 'slam_right')])

    def test_fresh_stereo_overrides_duplicate_history(self):
        owner = Ownership()
        assigned, _, _ = owner.assign(self.observation())
        self.assertEqual(assigned, {0:12, 1:13})
        self.assertEqual(owner.assign(self.observation(), 'baseline')[0], {})

    def test_never_force_two_left_detections_to_two_sides(self):
        assigned, _, _ = Ownership().assign(self.observation(.02, .01))[0:3]
        self.assertEqual(set(assigned), {0})

    def test_bounded_history_cannot_latch_wrong_side(self):
        owner = Ownership()
        owner.assign(self.observation(.9, .1))
        self.assertEqual(owner.assign(self.observation(.1, .9, 33_000_001))[0], {0:12, 1:13})
        empty = dict(raw_ns=70_000_000, hands={}, views=[])
        self.assertEqual(owner.assign(empty)[0], {})
        self.assertEqual(owner.previous, {})

    def test_ambiguous_new_hands_remain_unknown(self):
        self.assertEqual(Ownership().assign(self.observation(.50, .51))[0], {})

    def test_permuting_rows_preserves_assignment(self):
        obs = self.observation()
        first = Ownership().assign(obs)[0]
        obs['views'].reverse()
        self.assertEqual(Ownership().assign(obs)[0], first)


if __name__ == '__main__':
    unittest.main()
