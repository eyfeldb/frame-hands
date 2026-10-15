import unittest
import cv2
import numpy as np
from palm_flow import PalmFlow


class PalmFlowTests(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(1)
        rng=np.random.default_rng(7421)
        self.image=cv2.GaussianBlur(rng.integers(0,256,(220,240),dtype=np.uint8),(3,3),.6)
        self.palm=np.float32([[60,160],[60,60],[95,50],[135,60],[155,95]])

    def test_known_translation_tracks_pixels_without_replenishing(self):
        for region in [None,np.vstack((self.palm,np.repeat(self.palm[:1],16,axis=0)))]:
            tracker=PalmFlow(self.image,self.palm,0,region_points=region)
            for i in range(1,11):
                moved=cv2.warpAffine(self.image,np.float32([[1,0,i*2],[0,1,i]]),(240,220))
                result=tracker.advance(moved,i*33_333_333)
                self.assertTrue(result['supported'],result)
                np.testing.assert_allclose(tracker.palm,self.palm+[i*2,i],atol=.4)
                self.assertLessEqual(len(tracker.points),tracker.initial_count)

    def test_blank_after_seed_cannot_continue_and_cannot_restart(self):
        tracker=PalmFlow(self.image,self.palm,0)
        self.assertFalse(tracker.advance(np.zeros_like(self.image),33_333_333)['supported'])
        self.assertFalse(tracker.advance(self.image,66_666_666)['supported'])

    def test_no_texture_cannot_seed(self):
        tracker=PalmFlow(np.zeros_like(self.image),self.palm,0)
        self.assertFalse(tracker.active)

    def test_timestamp_gaps_expire_and_do_not_resurrect(self):
        for stamp in [0,-1,100_000_001]:
            tracker=PalmFlow(self.image,self.palm,0)
            self.assertFalse(tracker.advance(self.image,stamp)['supported'])
        tracker=PalmFlow(self.image,self.palm,0)
        for stamp in range(100_000_000,700_000_000,100_000_000):
            self.assertTrue(tracker.advance(self.image,stamp)['supported'])
        self.assertEqual(tracker.advance(self.image,666_666_666)['reason'],'seed_expired')

    def test_background_texture_can_pass_and_is_not_hand_presence(self):
        # Deliberately no hand in this synthetic image. The diagnostic must not
        # be described as a hand-presence validator merely because this passes.
        tracker=PalmFlow(self.image,self.palm,0)
        result=tracker.advance(self.image,33_333_333)
        self.assertTrue(result['supported'])
        self.assertTrue(result['hypothesis_only'])

    def test_invalid_geometry_and_image_types_rejected(self):
        for palm in [np.zeros((4,2)),np.full((5,2),np.nan)]:
            with self.assertRaises(ValueError):PalmFlow(self.image,palm,0)
        with self.assertRaises(ValueError):PalmFlow(self.image.astype(float),self.palm,0)


if __name__=='__main__':unittest.main()
