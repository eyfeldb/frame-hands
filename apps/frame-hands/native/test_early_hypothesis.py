import unittest
import torch
from early_hypothesis import early_score,first_side
from early_evidence import CompletingFeatures

class EarlyTests(unittest.TestCase):
    def test_gate_needs_all_evidence(self):
        self.assertEqual(early_score(.9,.9,5,.01),.9)
        self.assertEqual(early_score(.1,.9,5,.01),.1)
        for args in ((.7,.9,5,.01),(.9,.7,5,.01),(.9,.9,11,.01),(.9,.9,5,.02),(.9,.9,-1,.01),(.9,.9,5,-1),(.9,2,5,.01)):
            self.assertIsNone(early_score(*args))
        for i in range(4):
            for bad in (float('nan'),float('inf')):
                args=[.9,.9,5,.01];args[i]=bad
                self.assertIsNone(early_score(*args))
    def test_side_prior_validation(self):
        self.assertEqual(first_side(.5),1)
        self.assertEqual(first_side(.49),0)
        for bad in (-.1,1.1,float('nan')):
            with self.assertRaises(ValueError):first_side(bad)
    def cache(self):
        class Extractor:
            calls=0
            def extract(self,x):self.calls+=len(x);return x[:,None]*2
        images=torch.arange(8,dtype=torch.float32).reshape(2,2,2)
        return CompletingFeatures(10,[('a',1,0)],images[:1],images[:1,None]*2,Extractor()),images
    def test_missing_opposite_computed_only_when_selected(self):
        cache,images=self.cache()
        keys=[('a',1,1),('a',1,0)];selected=images[[1,0]]
        self.assertTrue(torch.equal(cache.select(10,keys,selected),selected[:,None]*2))
        cache.select(10,keys,selected)
        self.assertEqual(cache.regressor.calls,1);self.assertEqual(cache.late_passes,1)
    def test_stale_or_changed_pixels_fail_before_inference(self):
        cache,images=self.cache()
        with self.assertRaises(ValueError):cache.select(11,[('a',1,1)],images[1:])
        with self.assertRaises(ValueError):cache.select(10,[('a',1,1),('a',1,0)],images[[1,1]])
        self.assertEqual(cache.regressor.calls,0)

if __name__=='__main__':unittest.main()
