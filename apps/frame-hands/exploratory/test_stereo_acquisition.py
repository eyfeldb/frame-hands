import unittest,copy
import numpy as np
import test_association as fixtures
from stereo_acquisition import unused_matches

class StereoAcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.AssociationTests();self.f.setUp();self.obs=self.f.observation(1,which=(0,));self.hand=dict(side=0,source_id=1001,source_views=[dict(camera='slam_left',source_id=11)])
    def match(self,obs=None,hands=None):
        return unused_matches([self.hand] if hands is None else hands,self.obs if obs is None else obs,self.f.cameras,np.eye(4),{1001})
    def test_unique_other_view_found_without_classifier_agreement(self):
        self.obs['views'][1]['right']=.99
        result=self.match();self.assertEqual(set(result),{0});self.assertLess(result[0]['error'],1e-8)
    def test_other_output_owns_view(self):
        other=dict(side=1,source_id=1002,source_views=[dict(camera='slam_right',source_id=11)])
        self.assertFalse(self.match(hands=[self.hand,other]))
    def test_ambiguity_and_bad_geometry_abstain(self):
        obs=copy.deepcopy(self.obs);extra=copy.deepcopy(obs['views'][1]);extra['id']=44;obs['views'].append(extra)
        self.assertFalse(self.match(obs))
        obs=copy.deepcopy(self.obs);obs['views'][1]['points'][:,1]+=200;self.assertFalse(self.match(obs))
    def test_invalid_pixels_geometry_presence_and_missing_source_abstain(self):
        for field in ('points','presence'):
            obs=copy.deepcopy(self.obs)
            if field=='points':obs['views'][1][field][0,0]=np.nan
            else:obs['views'][1][field]=.49
            self.assertFalse(self.match(obs))
        self.assertFalse(self.match(dict(self.obs,views=self.obs['views'][1:])))
        self.assertFalse(unused_matches([self.hand],self.obs,self.f.cameras,None,{1001}))
    def test_two_proposals_cannot_share_unused_observation(self):
        obs=copy.deepcopy(self.obs);extra=copy.deepcopy(obs['views'][0]);extra['id']=12;obs['views'].append(extra)
        other=dict(side=1,source_id=1002,source_views=[dict(camera='slam_left',source_id=12)])
        self.assertFalse(unused_matches([self.hand,other],obs,self.f.cameras,np.eye(4),{1001,1002}))

    def test_ineligible_and_duplicate_views_rejected(self):
        self.assertFalse(unused_matches([self.hand],self.obs,self.f.cameras,np.eye(4),set()))
        obs=copy.deepcopy(self.obs);obs['views'].append(copy.deepcopy(obs['views'][1]))
        with self.assertRaises(ValueError):self.match(obs)

if __name__=='__main__':unittest.main()
