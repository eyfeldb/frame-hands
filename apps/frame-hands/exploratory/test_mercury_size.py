from pathlib import Path
import unittest
import numpy as np
from mercury_size import NativeSizeRefinement
DLL=Path(__file__).resolve().parents[3]/'.local/hand-alternatives/mercury-solver-size-v1/mercury_solver.dll'

@unittest.skipUnless(DLL.exists(),'Optional native size-refinement DLL is not built')
class NativeSizeTests(unittest.TestCase):
    def test_confidence_progress_target_and_weight(self):
        found=[[True,False],[True,False]]
        with NativeSizeRefinement(DLL) as scale:
            before=scale.pre(found,[False,False]);self.assertTrue(before['optimize_size']);self.assertEqual(before['size_error'],0)
            scale.post(.1,0,np.ones((2,21)));after=scale.finish()
            self.assertEqual(after['schedule_x'],1);self.assertAlmostEqual(after['target_size'],.1,places=7)
            before=scale.pre(found,[True,False]);self.assertAlmostEqual(before['size_error'],.0001,places=9)
            scale.post(.095,.1,np.ones((2,21))*.5);after=scale.finish();self.assertAlmostEqual(after['schedule_x'],1.25,places=6)
    def test_freeze_and_new_hand_schedule_order(self):
        with NativeSizeRefinement(DLL) as scale:
            for i in range(101):
                scale.pre([[1,0],[1,0]],[i>0,False]);scale.post(.09,0,np.ones((2,21)));scale.finish()
            frozen=scale.pre([[1,0],[1,0]],[True,False]);self.assertFalse(frozen['optimize_size']);self.assertEqual(frozen['size_error'],1);scale.finish()
            # Native computes this frame's flag/weight before the newly seen hand
            # reduces accumulated progress; refinement resumes on the next frame.
            new=scale.pre([[1,1],[1,1]],[True,False]);self.assertFalse(new['optimize_size']);self.assertEqual(new['schedule_x'],50);scale.finish()
            resumed=scale.pre([[1,1],[1,1]],[True,True]);self.assertTrue(resumed['optimize_size']);self.assertEqual(resumed['size_error'],.25);scale.finish()
    def test_monocular_inhibits_progress_and_two_hands_average_target(self):
        with NativeSizeRefinement(DLL) as scale:
            pre=scale.pre([[1,0],[0,0]],[True,False]);self.assertFalse(pre['optimize_size']);scale.post(.1,0,np.ones((2,21)));after=scale.finish();self.assertEqual(after['schedule_x'],0)
            scale.pre([[1,1],[1,1]],[True,True]);scale.post(.08,0,np.ones((2,21)));scale.post(.1,0,np.ones((2,21)));after=scale.finish();self.assertAlmostEqual(after['target_size'],.09,places=7);self.assertEqual(after['schedule_x'],2)
    def test_invalid_lifecycle_and_confidence_reject(self):
        with NativeSizeRefinement(DLL) as scale:
            with self.assertRaises(ValueError):scale.post(.1,0,np.ones((2,21)))
            scale.pre([[1,0],[1,0]],[False,False])
            with self.assertRaises(ValueError):scale.pre([[1,0],[1,0]],[True,False])
            with self.assertRaises(ValueError):scale.post(.1,0,np.full((2,21),np.nan))
            scale.finish()
        with self.assertRaises(ValueError):scale.finish()

if __name__=='__main__':unittest.main()
