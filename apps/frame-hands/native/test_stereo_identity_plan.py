import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'exploratory'))
from stereo_identity_plan import unique_pairs,shared_score,StereoIdentityPlan


class StereoSharingTests(unittest.TestCase):
    def test_mutual_uniqueness_and_ambiguity(self):
        self.assertEqual(unique_pairs([[.002,.09],[.08,.004]]),[(0,0),(1,1)])
        self.assertEqual(unique_pairs([[.002,.01],[.01,.004]]),[])
        self.assertEqual(unique_pairs([[.002],[.003]]),[])
        self.assertEqual(unique_pairs([[float('nan'),float('inf')]]),[])

    def test_weak_conflicting_and_invalid_evidence_requires_full_check(self):
        view=dict(right=.96)
        for evidence in (dict(right=.7,agreement=True),dict(right=.96,agreement=False),dict(right=1.2,agreement=True),dict(right=float('nan'),agreement=True)):
            self.assertIsNone(shared_score(view,view,evidence)[0])
        for secondary in (dict(right=.1),dict(right=.6)):
            self.assertIsNone(shared_score(view,secondary,dict(right=.96,agreement=True))[0])
        self.assertEqual(shared_score(view,view,dict(right=.96,agreement=True)),(.96,'share'))

    def test_lower_presence_candidate_still_makes_stereo_pair_ambiguous(self):
        class Camera:
            camera_to_world_xf=np.eye(4)
            def copy(self,**kw):return self
        class Geometry:
            def _feature(self,v,c):return v
            def _pair(self,a,b):return dict(error=.002)
        plan=StereoIdentityPlan({'slam_left':Camera(),'slam_right':Camera()});plan.geometry=Geometry()
        def view(camera,ident,presence,x):
            return dict(camera=camera,id=ident,presence=presence,right=.95,points=np.array([[x+i,50+i] for i in range(21)]))
        observation=dict(views=[view('slam_left',1,.99,0),view('slam_right',2,.99,0),view('slam_right',3,.6,100)])
        self.assertEqual(plan.groups(observation,np.eye(4))[0],[])

    def test_overlap_always_uses_independent_evidence(self):
        plan=StereoIdentityPlan({})
        view=dict(camera='slam_left',id=1,presence=.99,right=.95,points=np.zeros((21,2)))
        self.assertEqual(plan.groups(dict(views=[view,dict(view,id=2)]),np.eye(4)),([], 'overlap_or_invalid_points'))


if __name__=='__main__':unittest.main()
