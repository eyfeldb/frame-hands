"""Causal seed and metadata boundaries of the offline ROI experiment."""
import unittest
from probe_roi_continuation import read_reference


def row(t,valid=True,hands=None):
    return dict(seconds=t,capture_raw_ns=int(t*1e9),capture_mono_ns=int(t*1e9),capture_pose=dict(valid=valid),current_pose={},hands=[dict(side=0,source_id=1)] if hands is None else hands)


class RoiReferenceTests(unittest.TestCase):
    def test_future_hands_do_not_seed_or_enter_metadata(self):
        seed=row(1.)
        future=row(1.03,hands=[dict(side=1,source_id=999)])
        actual,metadata=read_reference([seed,future],1.,.1)
        future['hands']=[dict(side=0,source_id=777)]
        actual2,metadata2=read_reference([seed,future],1.,.1)
        self.assertEqual(actual,seed);self.assertEqual(actual2,seed)
        self.assertEqual(metadata,metadata2)
        self.assertNotIn('hands',metadata[future['capture_raw_ns']])

    def test_invalid_or_empty_latest_seed_does_not_fall_back_to_old_hand(self):
        for invalid in (row(1.,valid=False),row(1.,hands=[])):
            with self.assertRaises(ValueError):read_reference([row(.97),invalid,row(1.03)],1.,.1)

    def test_stale_future_or_duplicate_side_seed_rejected(self):
        for rows in ([row(.8)],[row(1.01)],[row(1.,hands=[dict(side=0),dict(side=0)])]):
            with self.assertRaises(ValueError):read_reference(rows,1.,.1)

    def test_duplicate_out_of_order_and_nonfinite_captures_rejected(self):
        bad=row(1.03);bad['seconds']=float('nan')
        for rows in ([row(1.),row(1.)],[row(1.),row(.99)],[row(1.),bad]):
            with self.assertRaises(ValueError):read_reference(rows,1.,.1)


if __name__=='__main__':unittest.main()
