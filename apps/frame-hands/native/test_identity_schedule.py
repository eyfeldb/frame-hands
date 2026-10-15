import unittest
import numpy as np
from identity_schedule import IdentitySchedule

def observation(stamp=1_000_000_000,shift=0,raw=.95,other=False):
 p=np.stack([np.linspace(100,120,21),np.linspace(100,140,21)],axis=1)+shift
 views=[{'camera':'slam_left','id':1,'presence':.9,'right':raw,'points':p}]
 if other:views.append(dict(views[0],id=2,right=.05))
 return {'raw_ns':stamp,'views':views}

def anchor(s,o):s.observe(o,[{'camera':'slam_left','source_id':1,'right':.95,'agreement':True}])

class SchedulingTests(unittest.TestCase):
 def test_reuse_cannot_extend_evidence_lifetime(self):
  s=IdentitySchedule();o=observation();s.decide(o);anchor(s,o)
  for delta in (33_000_000,66_000_000):self.assertEqual(s.decide(observation(o['raw_ns']+delta))[1],'reuse')
  self.assertEqual(s.decide(observation(o['raw_ns']+99_000_000))[1],'expired_or_missing_evidence')
 def test_overlap_conflict_motion_and_new_detection_require_fresh(self):
  for change in ({'other':True},{'raw':.05},{'shift':20}):
   s=IdentitySchedule();o=observation();s.decide(o);anchor(s,o)
   self.assertIsNone(s.decide(observation(o['raw_ns']+33_000_000,**change))[0])
  s=IdentitySchedule();o=observation();s.decide(o);anchor(s,o);n=observation(o['raw_ns']+33_000_000);n['views'][0]['id']=9
  self.assertIsNone(s.decide(n)[0])
 def test_stale_duplicate_and_reordered_timestamps_erase_anchor(self):
  for delta in (0,-1,101_000_000):
   s=IdentitySchedule();o=observation();s.decide(o);anchor(s,o)
   self.assertIsNone(s.decide(observation(o['raw_ns']+delta))[0]);self.assertIsNone(s.anchor)
 def test_saved_points_are_copied(self):
  s=IdentitySchedule();o=observation();s.decide(o);anchor(s,o);o['views'][0]['points']+=100
  self.assertEqual(s.decide(observation(o['raw_ns']+33_000_000))[1],'reuse')
if __name__=='__main__':unittest.main()
