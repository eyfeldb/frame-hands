"""Recorded crop geometry and local-pixel parity; cross-platform PNG rounding differs."""
import unittest
from pathlib import Path
import numpy as np
import test_chirality as fixtures
from frame_data import Recording,warp
from replay_raw_chirality import prepare

class RecordedCropsTests(unittest.TestCase):
    def test_geometry_matches_export_and_local_pixels(self):
        fixtures.ChiralityTests.setUpClass();f=fixtures.ChiralityTests
        from lib.common.camera import PinholePlaneCameraModel
        session=Path(__file__).resolve().parents[3]/'.local/hand-alternatives/results/live-mesh-20261005T050930Z'
        rec=Recording(session/'frame/sets.bin');seconds=(f.observation['raw_ns']-rec.index[0][0])/1e9
        images=next(images for _,stamp,images in rec.frames(seconds-.00001,seconds+.00001) if stamp==f.observation['raw_ns'])
        candidate=prepare(f.model.backend.cameras,images,f.observation)
        self.assertEqual(candidate.keys(),f.prepared.keys())
        for key,(image,meta) in candidate.items():
            saved=f.prepared[key][1];k=np.array(saved['intrinsics'])
            reference=PinholePlaneCameraModel(96,96,(k[0,0],k[1,1]),(k[0,2],k[1,2]),[],np.array(saved['camera_to_head_mm']))
            expected=warp(f.model.backend.cameras[key[0]],reference,images[key[0]][0])
            np.testing.assert_array_equal(image,expected)
            np.testing.assert_allclose(meta['intrinsics'],f.prepared[key][1]['intrinsics'],rtol=0,atol=1e-9)
            np.testing.assert_allclose(meta['camera_to_head_mm'],f.prepared[key][1]['camera_to_head_mm'],rtol=0,atol=1e-9)

if __name__=='__main__':unittest.main()
