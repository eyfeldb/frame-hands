import math
import struct
import unittest
from measure import decode, percentile, Observations, HEADER, HAND, SIZE


def packet(**changes):
    h = dict(magic=b'FHHANDS1',version=1,size=SIZE,seq=2,capture=1000000000,
             publish=1020000000,count=1,capsules=0)
    h.update(changes)
    out = bytearray(SIZE)
    HEADER.pack_into(out,0,*h.values())
    HAND.pack_into(out,64,1,2,1.,0.,*([.1]*63),0)
    HAND.pack_into(out,64+272,2,3,1.,0.,*([.1]*63),0)
    return out


class MeasurementContract(unittest.TestCase):
    def test_two_real_hands_and_latency(self):
        p=decode(packet(count=2),1030000000)
        self.assertEqual((p['latency_ms'],p['stereo'],p['sides']),(20,2,['left','right']))

    def test_stale_is_not_live(self):
        with self.assertRaisesRegex(ValueError,'stale'):
            decode(packet(),1200000000)

    def test_clock_errors(self):
        for fields in [dict(publish=999999999),dict(capture=0),dict(publish=1040000000)]:
            with self.subTest(fields=fields),self.assertRaisesRegex(ValueError,'timestamps'):
                decode(packet(**fields),1030000000)

    def test_partial_packet_and_writer(self):
        for data in [packet()[:-1],packet(seq=3),packet(seq=0),packet(version=2),packet(count=3)]:
            with self.assertRaises(ValueError):decode(data,1030000000)

    def test_nonfinite_and_extreme_positions(self):
        for value in [math.nan, math.inf, 100.]:
            p=packet();struct.pack_into('<f',p,64+16,value)
            with self.assertRaisesRegex(ValueError,'geometry'):decode(p,1030000000)

    def test_duplicate_side_rejected(self):
        p=packet(count=2);struct.pack_into('<I',p,64+272+4,2)
        with self.assertRaisesRegex(ValueError,'handedness'):decode(p,1030000000)

    def test_empty_is_valid_but_not_hands(self):
        self.assertEqual(decode(packet(count=0),1030000000)['count'],0)

    def test_percentile(self):
        self.assertIsNone(percentile([],.95))
        self.assertEqual(percentile(list(range(1,101)),.95),95)

    def test_slow_publication_stays_in_latency_tail(self):
        observations=Observations()
        for n in range(10):
            capture=1000000000+n*200000000
            delay=150000000 if n==9 else 20000000
            observations.add(packet(seq=2*(n+1),capture=capture,publish=capture+delay),capture+delay+1000000)
        result=observations.finish(2)
        self.assertEqual(result['hand_capture_to_publish_ms']['p95'],150)
        self.assertEqual(result['stale_publications'],1)
        self.assertEqual(result['samples'],10)

    def test_repeated_packet_expires_without_adding_samples(self):
        observations=Observations()
        observations.add(packet(),1030000000)
        observations.add(packet(),1040000000)
        observations.add(packet(),1200000000)
        result=observations.finish(.2)
        self.assertEqual((result['samples'],result['duplicate_polls'],result['stale_polls']),(1,2,1))
        self.assertEqual(result['fresh_hand_polls'],2)

    def test_old_hands_and_fresh_empty_packets_cannot_pass(self):
        observations=Observations()
        observations.add(packet(publish=1150000000),1160000000)
        observations.add(packet(seq=4,capture=1200000000,publish=1220000000,count=0),1230000000)
        result=observations.finish(.3)
        self.assertFalse(result['has_live_hand_evidence'])
        self.assertFalse(result['acceptance_verified'])
        self.assertEqual(result['hand_capture_to_publish_ms']['p50'],150)

    def test_duplicate_identity_rejected(self):
        p=packet(count=2);struct.pack_into('<I',p,64+272,1)
        with self.assertRaisesRegex(ValueError,'identity'):decode(p,1030000000)


if __name__=='__main__':unittest.main()
