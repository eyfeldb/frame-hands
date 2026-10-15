import unittest
from replay_capture_join import exact_capture_pairs


class CaptureJoinTests(unittest.TestCase):
    def raw(self, stamps):return [(i/30,s,{'pixels':s}) for i,s in enumerate(stamps)]
    def rows(self, stamps):return [dict(capture_raw_ns=s,payload=['unchanged',s]) for s in stamps]

    def test_sparse_delivery_preserves_exact_payload_and_cadence(self):
        rows=self.rows([2,4]);stats={}
        result=list(exact_capture_pairs(self.raw([1,2,3,4,5]),rows,True,stats))
        self.assertEqual([raw[1] for raw,row in result],[2,4])
        self.assertIs(result[0][1],rows[0]);self.assertIs(result[1][1],rows[1])
        self.assertEqual(stats,dict(raw_frames_available=5,raw_frames_without_baseline=3,
            raw_unmatched_inside_span=1,raw_before_first_baseline=1,raw_after_last_baseline=1,matched_captures=2))

    def test_default_does_not_silently_change_replay_cadence(self):
        with self.assertRaisesRegex(ValueError,'lacks exact'):
            list(exact_capture_pairs(self.raw([1,2,3]),self.rows([1,3])))

    def test_prediction_without_pixels_is_not_retimed(self):
        with self.assertRaisesRegex(ValueError,'no exact raw'):
            list(exact_capture_pairs(self.raw([1,3]),self.rows([1,2,3]),True))

    def test_reversed_or_duplicate_baselines_rejected(self):
        for stamps in [[1,1,2],[1,3,2]]:
            with self.assertRaisesRegex(ValueError,'strictly increasing'):
                list(exact_capture_pairs(self.raw([1,2,3,4]),self.rows(stamps),True))

    def test_interval_may_exclude_earlier_or_later_predictions(self):
        rows=self.rows([1,2,3,4]);stats={}
        result=list(exact_capture_pairs(self.raw([2,3]),rows,False,stats))
        self.assertEqual([row['capture_raw_ns'] for _,row in result],[2,3])
        self.assertEqual(stats['raw_frames_without_baseline'],0)


if __name__=='__main__':unittest.main()
