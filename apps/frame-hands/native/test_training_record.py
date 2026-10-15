import unittest
from unittest.mock import patch
import numpy as np
from fetch_training_record import parse_index,range_bytes


class BoundedDatasetTests(unittest.TestCase):
    def test_valid_index_and_inconsistent_extents(self):
        # Two differently sized uint8 arrays: (2,3), (3,3).
        values=[0x584449544E54,1,1,1,2,4,0,2,4,0,6,15,2,3,3,3]
        result=parse_index(np.asarray(values,dtype='<i8').tobytes())
        self.assertEqual(result['shapes'],[[2,3],[3,3]])
        self.assertEqual(result['offsets'],[0,6,15])
        values[11]=14
        with self.assertRaisesRegex(ValueError,'extent'):parse_index(np.asarray(values,dtype='<i8').tobytes())

    def test_ignored_range_never_reads_full_payload(self):
        class Response:
            status=200
            headers={}
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self,*args):raise AssertionError('Must not read unsupported range')
        with patch('urllib.request.urlopen',return_value=Response()):
            with self.assertRaisesRegex(ValueError,'refusing full-file'):range_bytes('https://example.invalid',0,5,100)

    def test_range_identity_and_length_are_both_required(self):
        class Response:
            status=206
            headers={'Content-Range':'bytes 0-5/100'}
            data=b'abcdef'
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self,n):return self.data[:n]
        response=Response()
        with patch('urllib.request.urlopen',return_value=response):
            self.assertEqual(range_bytes('https://example.invalid',0,5,100),b'abcdef')
            response.headers={'Content-Range':'bytes 10-15/100'}
            with self.assertRaisesRegex(ValueError,'Content-Range'):range_bytes('https://example.invalid',0,5,100)
            response.headers={'Content-Range':'bytes 0-5/100'};response.data=b'abc'
            with self.assertRaisesRegex(ValueError,'Truncated'):range_bytes('https://example.invalid',0,5,100)

    def test_oversized_record_rejected_before_network(self):
        with patch('urllib.request.urlopen',side_effect=AssertionError('No network allowed')):
            with self.assertRaisesRegex(ValueError,'cap'):range_bytes('https://example.invalid',0,8000000,9000000)


if __name__=='__main__':unittest.main()
