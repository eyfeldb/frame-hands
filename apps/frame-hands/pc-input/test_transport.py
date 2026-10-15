import struct
import unittest
from unittest.mock import Mock,patch
from relay import convert,publish_atomic,RAW_SIZE,HEADER,HAND


def response(**changes):
    values=dict(magic=b'FHHANDS1',version=1,size=RAW_SIZE,seq=2,capture=980_000_000,publish=990_000_000,count=2,capsules=0)
    values.update(changes);raw=bytearray(RAW_SIZE)
    struct.pack_into('<8sIIQQQII',raw,0,*values.values())
    for i in range(min(values['count'],2)):
        struct.pack_into('<IIff63fI',raw,64+i*272,i+1,2+i,.9,0,*([.1]*63),0)
    return struct.pack('<8sQ',b'FHREAD1\0',1_000_000_000)+raw


class TransportTests(unittest.TestCase):
    def test_windows_replace_contention_retries(self):
        temporary=Mock()
        with patch('relay.os.replace',side_effect=[PermissionError(),None]) as replace, patch('relay.time.sleep'):
            self.assertTrue(publish_atomic(temporary,'destination',b'packet'))
            self.assertEqual(replace.call_count,2)
        temporary.write_bytes.assert_called_once_with(b'packet')
    def test_persistent_contention_is_bounded(self):
        with patch('relay.os.replace',side_effect=PermissionError()), patch('relay.time.monotonic',side_effect=[0,.1]):
            self.assertFalse(publish_atomic(Mock(),'destination',b'packet'))
    def test_layout_and_side_order(self):
        data,report=convert(response(),1000,1010,4)
        self.assertEqual(len(data),568);self.assertEqual(HEADER.unpack_from(data)[5],985)
        self.assertEqual(HAND.unpack_from(data,48)[:2],(1,3))
        self.assertEqual(HAND.unpack_from(data,308)[:2],(2,3));self.assertEqual(report['hands'],2)
    def test_stale_source_future_capture_and_slow_transport(self):
        for reply,start,end in [(response(capture=800_000_000),1000,1010),(response(capture=1_100_000_000),1000,1010),(response(),1000,1200),(response(),1000,1090)]:
            data,r=convert(reply,start,end,5);self.assertFalse(r['valid']);self.assertEqual(data[48:],bytes(520))
    def test_duplicate_handedness_rejected(self):
        r=bytearray(response());struct.pack_into('<I',r,16+64+272+4,2)
        data,report=convert(r,1000,1001,1);self.assertTrue(report['valid']);self.assertEqual(data[48:],bytes(520))
    def test_nan_coordinates_rejected(self):
        r=bytearray(response());struct.pack_into('<f',r,16+80,float('nan'))
        data,report=convert(r,1000,1001,1)
        self.assertTrue(report['valid']);self.assertEqual(data[48:308],bytes(260))
        self.assertEqual(HAND.unpack_from(data,308)[:2],(2,3));self.assertEqual(report['rejected_hands'],1)
    def test_new_track_age_is_not_low_detection_probability(self):
        r=bytearray(response());struct.pack_into('<f',r,16+64+8,.2)
        data,report=convert(r,1000,1001,1)
        self.assertEqual(report['hands'],2);self.assertEqual(HAND.unpack_from(data,48)[:2],(1,3))
    def test_invalid_age_score_does_not_remove_opposite_hand(self):
        for confidence in [-.1,1.1,float('nan')]:
            r=bytearray(response());struct.pack_into('<f',r,16+64+8,confidence)
            data,report=convert(r,1000,1001,1)
            self.assertEqual(report['hands'],1);self.assertEqual(data[48:308],bytes(260));self.assertEqual(HAND.unpack_from(data,308)[:2],(2,3))
    def test_seqlock_header_and_count_rejected(self):
        for args in [dict(seq=3),dict(version=7),dict(size=10),dict(count=3),dict(magic=b'BADMAGIC')]:
            self.assertFalse(convert(response(**args),1000,1001,1)[1]['valid'])
    def test_empty_source_releases(self):
        data,r=convert(response(count=0),1000,1001,1);self.assertTrue(r['valid']);self.assertEqual(data[48:],bytes(520))
    def test_short_or_invalid_export_rejected(self):
        for b in [b'',response()[:-1],bytes(len(response()))]:
            with self.assertRaises(ValueError):convert(b,1000,1001,1)
    def test_controls_explicit_opt_in(self):
        self.assertEqual(HEADER.unpack_from(convert(response(),1000,1001,1)[0])[6],0)
        self.assertEqual(HEADER.unpack_from(convert(response(),1000,1001,1,True)[0])[6],1)


if __name__=='__main__':unittest.main()
