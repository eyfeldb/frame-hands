import unittest
from probe_actual_monocular_flow import select_cases


def row(stereo, side=0, source=1, camera='slam_left'):
    return dict(hands={source:dict(right=side,stereo=stereo)},views=[dict(id=source,camera=n) for n in (['slam_left','slam_right'] if stereo==2 else [camera])])


class EpisodeSelection(unittest.TestCase):
    def select(self, audit):
        poses={(stamp,1):dict(side=int(r['hands'][1]['right'])) for stamp,r in audit.items() if 1 in r['hands'] and r['hands'][1]['stereo']==2}
        return select_cases(audit,poses,set(audit))[0]
    def test_mono_does_not_refresh_original_seed(self):
        audit={0:row(2),33_000_000:row(1),66_000_000:row(1),99_000_000:row(1),132_000_000:row(1),165_000_000:row(2)}
        cases=self.select(audit)
        self.assertEqual(len(cases),3)
        self.assertTrue(all(c['prior_raw_ns']==0 for c in cases))
    def test_missing_source_side_change_and_ambiguity_clear_history(self):
        missing=dict(hands={},views=[])
        ambiguous=row(1);ambiguous['hands'][2]=dict(right=0,stereo=1);ambiguous['views'].append(dict(id=2,camera='slam_right'))
        for bad in (missing,row(1,side=1),ambiguous):
            self.assertEqual(self.select({0:row(2),33_000_000:bad,66_000_000:row(1)}),[])
    def test_reacquisition_uses_last_mono_camera_without_future_fit(self):
        cases=self.select({0:row(2),33_000_000:row(1,camera='slam_right'),66_000_000:row(2)})
        self.assertEqual([c['kind'] for c in cases],['mono','reacquired'])
        self.assertEqual(cases[-1]['view'],'slam_right')
        self.assertEqual(cases[-1]['prior_raw_ns'],0)
    def test_missing_hmd_context_resets_seed(self):
        audit={0:row(2),33_000_000:row(1),66_000_000:row(1)}
        cases,_=select_cases(audit,{(0,1):dict(side=0)},{0,66_000_000})
        self.assertEqual(cases,[])


if __name__=='__main__': unittest.main()
