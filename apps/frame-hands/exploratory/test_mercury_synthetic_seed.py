from pathlib import Path
import unittest
import numpy as np
from mercury_forward import NativeHandGenerator
from mercury_native_solver import NativeSolver,ML_TO_XR
from mercury_synthetic_seed import seed_known_synthetic_hand
from test_mercury_native_solver import POSE,observations_from_joints

ROOT=Path(__file__).resolve().parents[3]/'.local/hand-alternatives'
DLL=ROOT/'mercury-solver-initialization-v1/mercury_solver.dll'
OLD=ROOT/'mercury-solver-forward-v1/mercury_solver.dll'
WRIST=[.70710678,0,0,.70710678,.04,-.03,-.5]


@unittest.skipUnless(DLL.exists(),'Optional synthetic initialization diagnostic is unavailable')
class SyntheticSeedTests(unittest.TestCase):
    def test_truth_roundtrip_and_first_step_stability_parity(self):
        generator=NativeHandGenerator(DLL)
        for side in (0,1):
            for values in ([ -.1,-.05,-.05],[-1.2,-1.2,-.8]):
                angles=np.tile(values,(4,1));truth=generator.generate(WRIST,angles,side)[ML_TO_XR,:3]
                views=observations_from_joints(truth,side);outputs=[]
                for smoothing in (0,2):
                    with NativeSolver(DLL,POSE,side) as solver:
                        seed_known_synthetic_hand(solver,WRIST,angles)
                        result=solver.run(views,smoothing=smoothing,depth_weight=0)
                        joints=np.asarray(result['joints_left_camera_m'])[ML_TO_XR]
                        np.testing.assert_allclose(joints,truth,rtol=0,atol=1e-7)
                        outputs.append(result)
                        with self.assertRaises(ValueError):seed_known_synthetic_hand(solver,WRIST,angles)
                self.assertEqual(outputs[0],outputs[1])
    def test_invalid_seed_preserves_cold_behavior_and_reset_restores_it(self):
        angles=np.tile([-.1,-.05,-.05],(4,1));truth=NativeHandGenerator(DLL).generate(WRIST,angles,0)[ML_TO_XR,:3];views=observations_from_joints(truth,0)
        with NativeSolver(DLL,POSE,0) as baseline:expected=baseline.run(views,depth_weight=0)
        for wrist,values,size in [([0]*7,angles,.09),(WRIST,np.ones((4,3)),.09),(WRIST,angles,.5),(WRIST,np.full((4,3),np.nan),.09),(WRIST,np.zeros((3,3)),.09)]:
            with NativeSolver(DLL,POSE,0) as solver:
                with self.assertRaises(ValueError):seed_known_synthetic_hand(solver,wrist,values,size)
                self.assertTrue(solver.first);self.assertEqual(solver.run(views,depth_weight=0),expected)
        with NativeSolver(DLL,POSE,0) as solver:
            seed_known_synthetic_hand(solver,WRIST,angles)
            self.assertEqual(solver.run(views,reset=True,depth_weight=0),expected)
        with NativeSolver(DLL,POSE,0) as solver:pass
        with self.assertRaises(ValueError):seed_known_synthetic_hand(solver,WRIST,angles)
    @unittest.skipUnless(OLD.exists(),'Previous control DLL unavailable')
    def test_uncalled_diagnostic_preserves_outputs_and_is_absent_from_old_dll(self):
        import ctypes
        self.assertFalse(hasattr(ctypes.CDLL(str(OLD)),'fh_solver_seed_synthetic'))
        for side in (0,1):
            with NativeSolver(OLD,POSE,side) as before,NativeSolver(DLL,POSE,side) as after:
                for i in range(6):
                    angles=np.tile([-.1-i*.1,-.05-i*.1,-.05-i*.1],(4,1))
                    old=NativeHandGenerator(OLD).generate(WRIST,angles,side);new=NativeHandGenerator(DLL).generate(WRIST,angles,side)
                    np.testing.assert_array_equal(old,new);views=observations_from_joints(new[ML_TO_XR,:3],side)
                    self.assertEqual(before.run(views,depth_weight=0),after.run(views,depth_weight=0))


if __name__=='__main__':unittest.main()
