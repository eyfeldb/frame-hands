"""Negative controls for evaluation leakage in the saved size-replay audit."""
from copy import deepcopy
import unittest
import numpy as np
from audit_mercury_size_long import audit


class FreezeAuditTests(unittest.TestCase):
    def fixture(self):
        size = float(np.float32(.09))
        post = dict(target_size=size, schedule_x=0)
        common = dict(observations=[], fits=[], scale_post=post)
        return [dict(common, seconds=0, stamp=1, phase='calibration',
                     scale_pre=dict(target_size=size, schedule_x=0, size_error=0, optimize_size=True)),
                dict(common, seconds=18, stamp=18000000001, phase='evaluation',
                     scale_pre=dict(target_size=size, schedule_x=0, size_error=1, optimize_size=False))]

    def test_empty_prefix_does_not_manufacture_calibration(self):
        result = audit(self.fixture(), dict(freeze_seconds=18))
        self.assertEqual(result['freeze']['schedule_x'], 0)
        self.assertEqual(result['counts']['verified_scale_posts'], 0)

    def test_evaluation_cannot_resume_size_updates(self):
        for field, value in [('optimize_size', True), ('target_size', .1), ('schedule_x', 1)]:
            rows = deepcopy(self.fixture())
            rows[1]['scale_pre'][field] = value
            with self.subTest(field=field), self.assertRaises(AssertionError):
                audit(rows, dict(freeze_seconds=18))

    def test_evaluation_cannot_mutate_shared_state_or_boundary(self):
        rows = deepcopy(self.fixture())
        rows[1]['scale_post'] = dict(target_size=.1, schedule_x=0)
        with self.assertRaises(AssertionError):
            audit(rows, dict(freeze_seconds=18))
        rows = self.fixture()
        rows[1]['phase'] = 'calibration'
        with self.assertRaises(AssertionError):
            audit(rows, dict(freeze_seconds=18))


if __name__ == '__main__':
    unittest.main()
