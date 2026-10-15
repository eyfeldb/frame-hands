import tempfile
import struct
import unittest
from pathlib import Path
import numpy as np
from pose_trace_audit import classify, records, valid_matrix


class PoseTraceAuditTests(unittest.TestCase):
    def sample(self, **overrides):
        return dict(dict(sequence=7,flags=3,head_flags=3,capture_flags=3,matrix_valid=True),**overrides)

    def test_missing_evidence_is_not_a_headset_failure(self):
        self.assertEqual(classify([],[],set()),'not_exported')
        self.assertEqual(classify([{}],[],{7}),'not_sampled')
        self.assertEqual(classify([{}],[self.sample(flags=1,head_flags=0,capture_flags=0)],{7}),'mailbox_stale')

    def test_observed_current_and_capture_pose_failures_are_distinct(self):
        self.assertEqual(classify([{}],[self.sample(head_flags=1,capture_flags=0)],{7}),'current_head_invalid')
        self.assertEqual(classify([{}],[self.sample(capture_flags=1)],{7}),'capture_head_invalid')

    def test_pose_from_rejected_transport_does_not_establish_capture_geometry(self):
        self.assertEqual(classify([{}],[self.sample()],set()),'unusable_transport_pose')
        self.assertEqual(classify([{}],[self.sample()],{7}),'exact_join_valid')

    def test_bad_matrix_cannot_pass_valid_flags(self):
        self.assertEqual(classify([{}],[self.sample(matrix_valid=False)],{7}),'matrix_rejected')
        transform=np.eye(4)[:3]
        self.assertTrue(valid_matrix(transform))
        for bad in [transform*2,np.full((3,4),np.nan),transform*np.array([[-1,1,1,1]])]:
            self.assertFalse(valid_matrix(bad))

    def test_rejected_duplicate_sample_does_not_hide_valid_exact_sample(self):
        samples=[self.sample(capture_flags=0),self.sample()]
        self.assertEqual(classify([{}],samples,{7}),'exact_join_valid')

    def test_truncated_or_wrong_protocol_records_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'trace.bin'
            record=struct.pack('<8sII',b'FHRELAY1',1,32)+bytes(16)
            path.write_bytes(record)
            self.assertEqual(list(records(path,32,b'FHRELAY1')),[record])
            path.write_bytes(record+b'partial')
            with self.assertRaises(ValueError):list(records(path,32,b'FHRELAY1'))
            path.write_bytes(record)
            with self.assertRaises(ValueError):list(records(path,32,b'FHTRACE1'))


if __name__=='__main__':unittest.main()
