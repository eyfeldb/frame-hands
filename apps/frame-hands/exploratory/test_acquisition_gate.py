import unittest
from acquisition_gate import AcquisitionGate


def hand(side=0,identity=1,stereo=False):
    return dict(side=side,source_id=identity,source_views=[dict(camera=c) for c in (('slam_left','slam_right') if stereo else ('slam_left',))])

def evidence(stamp,side=0,identity=1):
    return [dict(stamp=stamp,side=side,source_id=identity,camera=c,presence=.9,normalized_disagreement=.02) for c in ('slam_left','slam_right')]

class AcquisitionGateTests(unittest.TestCase):
    def test_persistent_single_camera_never_confirms(self):
        g=AcquisitionGate()
        for t in range(1,201):
            self.assertFalse(g.step(t*30_000_000,True,[hand()],evidence(t*30_000_000)[:1])[0])

    def test_independent_other_camera_confirms_and_monocular_continues(self):
        g=AcquisitionGate();h=hand();self.assertEqual(g.step(1,True,[h],evidence(1))[0],[h])
        self.assertEqual(g.step(30_000_001,True,[h],[])[0],[h])
        self.assertFalse(g.step(60_000_001,True,[hand(identity=2)],[])[0])

    def test_stereo_anchor_continuity_and_expiry(self):
        g=AcquisitionGate();h=hand(stereo=True);self.assertEqual(g.step(1,True,[h],[])[0],[h])
        self.assertFalse(g.step(650_000_002,True,[hand()],[])[0])
        self.assertEqual(g.step(650_000_003,True,[h],[])[0],[h])
        g.step(650_000_004,False,[],[]);self.assertFalse(g.step(650_000_005,True,[hand()],[])[0])

    def test_evidence_must_match_capture_side_identity_and_distinct_views(self):
        for rows in (evidence(1,identity=2),evidence(1,side=1),[evidence(1)[0]]*2):
            self.assertFalse(AcquisitionGate().step(1,True,[hand()],rows)[0])
        with self.assertRaises(ValueError):AcquisitionGate().step(1,True,[hand()],evidence(2))
        with self.assertRaises(ValueError):AcquisitionGate().step(1,True,[hand(),hand()],[])

    def test_one_failed_view_and_out_of_order_stamp_rejected(self):
        g=AcquisitionGate();rows=evidence(1);rows[1]['presence']=.49
        self.assertFalse(g.step(1,True,[hand()],rows)[0])
        with self.assertRaises(ValueError):g.step(1,True,[],[])

if __name__=='__main__':unittest.main()
