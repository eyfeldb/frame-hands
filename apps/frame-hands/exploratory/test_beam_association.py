"""Causal beam invariants, state isolation and width-one control equivalence."""
import unittest
import numpy as np
from beam_association import BeamAssociation
from temporal_association import TemporalAssociation
import test_association as fixtures

class BeamTests(unittest.TestCase):
    observation=fixtures.AssociationTests.observation
    choices=fixtures.AssociationTests.choices
    def setUp(self):
        fixtures.AssociationTests.setUp(self)
        self.policy=BeamAssociation(self.cameras)

    def test_width_one_reproduces_single_history(self):
        beam=BeamAssociation(self.cameras,width=1);single=TemporalAssociation(self.cameras)
        for i in range(25):
            o=self.observation(1+i/30,qs=(.98,.98) if i%7==0 else (.02,.98),ids=(22,11) if i>12 else (11,22),which=() if i==18 else (0,1),offset=i*2)
            self.assertEqual(beam.profile(beam.assign(o,np.eye(4))),beam.profile(single.assign(o,np.eye(4))))

    def test_branches_have_independent_motion_state(self):
        self.choices(self.observation(1))
        tracks=[h['policy'].tracks[0] for h in self.policy.hypotheses if 0 in h['policy'].tracks]
        self.assertGreaterEqual(len(tracks),2)
        before=tracks[1]['views']['slam_left']['point'].copy() if 'slam_left' in tracks[1]['views'] else None
        tracks[0]['views']['slam_left']['point'][:]=12345
        if before is not None:self.assertTrue(np.array_equal(tracks[1]['views']['slam_left']['point'],before))
        self.assertIsNot(tracks[0],tracks[1])

    def test_single_hand_injective_and_no_empty_frame_synthesis(self):
        for i in range(12):
            o=self.observation(1+i/30,which=(0,))
            assigned,groups,_=self.policy.assign(o,np.eye(4))
            used=[(v['camera'],v['id']) for ident in assigned.values() for v in groups[ident]]
            self.assertEqual(len(set(used)),len(used))
            self.assertLessEqual(len(assigned),1)
            self.assertLessEqual(len(self.policy.hypotheses),4)
        self.assertEqual(self.choices(self.observation(1.5,which=())),{})

    def test_context_loss_resets_every_branch(self):
        self.choices(self.observation(1))
        self.assertEqual(self.policy.assign(self.observation(1.033),None),({},{},{}))
        self.assertTrue(all(not h['policy'].tracks for h in self.policy.hypotheses))
        self.assertEqual(self.choices(self.observation(1.066,qs=(.5,.5))),{})

    def test_duplicate_does_not_update_any_branch(self):
        self.choices(self.observation(1))
        count=len(self.policy.hypotheses)
        self.assertEqual(self.choices(self.observation(1,qs=(.98,.02))),{})
        self.assertEqual(len(self.policy.hypotheses),count)
        self.assertTrue(all(t['time']==1 for h in self.policy.hypotheses for t in h['policy'].tracks.values()))

if __name__=='__main__':unittest.main()
