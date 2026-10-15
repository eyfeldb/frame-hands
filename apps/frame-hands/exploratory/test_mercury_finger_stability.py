from copy import deepcopy
from pathlib import Path
import unittest
import numpy as np
from mercury_native_solver import NativeSolver, pack_views
from replay_mercury_crop_fallback import constant_finger_stability
from test_mercury_native_solver import synthetic_views, POSE

DLL=Path(__file__).resolve().parents[3]/'.local/hand-alternatives/mercury-solver-curl-diag-v1/mercury_solver.dll'
OLD=Path(__file__).resolve().parents[3]/'.local/hand-alternatives/mercury-solver-size-v1/mercury_solver.dll'


@unittest.skipUnless(DLL.exists() and OLD.exists(),'Optional native finger diagnostic build unavailable')
class FingerDiagnosticTests(unittest.TestCase):
    def test_diagnostic_is_read_only_and_preserves_old_fits(self):
        for side in (0,1):
            views=synthetic_views(side)
            with NativeSolver(DLL,POSE,side) as new, NativeSolver(OLD,POSE,side) as old:
                for step in range(4):
                    self.assertEqual(new.run(views),old.run(views))
                    before=new.history().copy();fingers=new.fingers().copy()
                    a=new.finger_stability(views);b=new.finger_stability(views)
                    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(a[:,:3],fingers)
                    np.testing.assert_array_equal(new.history(),before);np.testing.assert_array_equal(new.fingers(),fingers)
    def test_absolute_variance_does_not_change_native_disagreement_multiplier(self):
        with NativeSolver(DLL,POSE,0) as solver:
            solver.run(synthetic_views(0));sums=solver.fingers().sum(1);outputs=[]
            for variance in (.1,10.):
                views=synthetic_views(0)
                for v in views.values():
                    v['curls'][1:5]=(sums+.6).tolist();v['curls'][6:10]=[variance-.01]*4
                outputs.append(solver.finger_stability(views))
            np.testing.assert_allclose(outputs[0][:,5],.6,atol=1e-6)
            np.testing.assert_allclose(outputs[0][:,5],outputs[1][:,5],atol=1e-6)
            self.assertGreater(outputs[1][0,7],outputs[0][0,7]*99)
    def test_constant_ablation_changes_only_non_thumb_curl_values(self):
        for side in (0,1):
            views=synthetic_views(side);saved=deepcopy(views)
            with NativeSolver(DLL,POSE,side) as solver:
                solver.run(views);before=pack_views(views,side);after=pack_views(constant_finger_stability(solver,views),side)
                allowed={121*view+111+2*finger for view in range(2) for finger in range(1,5)}
                self.assertTrue(set(np.flatnonzero(before!=after)).issubset(allowed))
                protected=[i for i in range(242) if i not in allowed]
                np.testing.assert_array_equal(before[protected],after[protected]);self.assertEqual(views,saved)
    def test_native_zero_weight_and_invalid_lifecycle(self):
        views=synthetic_views(0)
        with NativeSolver(DLL,POSE,0) as solver:
            with self.assertRaises(ValueError):solver.fingers()
            with self.assertRaises(ValueError):solver.finger_stability(views)
            solver.run(views);sums=solver.fingers().sum(1)
            for v in views.values():v['curls'][1:5]=(sums+1.3).tolist()
            np.testing.assert_array_equal(solver.finger_stability(views)[:,5],np.zeros(4))
            absent=deepcopy(views)
            for v in absent.values():v['active']=False
            with self.assertRaises(ValueError):solver.finger_stability(absent)
        with self.assertRaises(ValueError):solver.fingers()


if __name__=='__main__':unittest.main()
