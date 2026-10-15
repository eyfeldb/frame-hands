import copy,unittest
from stereo_exit import needs_completion

class Probe:
    def __init__(self,counts):
        self.counts=counts;self.tracks={0:{'time':4,'views':{'a':{'point':[1,2]}}}};self.last_debug={'old':[1]};self.last_stamp=4;self.generation=10;self.ambiguous=True;self.resolve_streak=1
    def assign(self,observation,world):
        self.tracks[0]['views']['a']['point'][0]=99
        self.last_debug['old'].append(2)
        self.last_stamp=5;self.generation+=1;self.ambiguous=False;self.resolve_streak+=1
        self.last_debug={'assignments':[{'views':[{}]*n} for n in self.counts]}

class StereoExitTests(unittest.TestCase):
    def test_single_view_or_empty_completes(self):
        for counts in ([],[1],[2,1]):self.assertTrue(needs_completion(Probe(counts),{},None))
    def test_all_stereo_can_exit(self):
        for counts in ([2],[2,2]):self.assertFalse(needs_completion(Probe(counts),{},None))
    def test_probe_cannot_advance_real_state(self):
        p=Probe([2,1]);old=copy.deepcopy(p.__dict__)
        needs_completion(p,{},None)
        self.assertEqual(p.__dict__,old)

if __name__=='__main__':unittest.main()
