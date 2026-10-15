"""Behavioral checks for the experimental causal association path."""
from pathlib import Path
import sys,unittest
import numpy as np
from temporal_association import TemporalAssociation
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'.local/hand-alternatives/UmeTrack'))
from lib.common.camera import Fisheye62CameraModel

class AssociationTests(unittest.TestCase):
    def setUp(self):
        self.cameras={}
        for name,x in [('slam_left',-30),('slam_right',30)]:
            xf=np.eye(4);xf[0,3]=x
            self.cameras[name]=Fisheye62CameraModel(1056,1024,(330,330),(528,512),(0,)*8,xf)
        self.policy=TemporalAssociation(self.cameras)
    def observation(self,t,qs=(.02,.98),ids=(11,22),which=(0,1),offset=0):
        views=[]
        shape=np.zeros((21,3));shape[:,2]=500
        shape[:,0]=np.tile([-30,-15,0,15,30,0,0],3)
        shape[:,1]=np.linspace(0,80,21);shape[0,1]=-40
        for side in which:
            points=shape.copy();points[:,0]+=(-140 if side==0 else 140)+offset
            for name,cam in self.cameras.items():
                views.append(dict(camera=name,id=ids[side],presence=.99,right=qs[side],points=cam.eye_to_window(cam.world_to_eye(points))))
        return dict(raw_ns=int(t*1e9),views=views,hands={})
    def choices(self,o):
        assigned,groups,_=self.policy.assign(o,np.eye(4))
        return {side:{v['id'] for v in groups[i]} for side,i in assigned.items()}
    def test_detector_id_handover_follows_pixels(self):
        self.assertEqual(self.choices(self.observation(1)),{0:{11},1:{22}})
        self.assertEqual(self.choices(self.observation(1.033,ids=(22,11))),{0:{22},1:{11}})
    def test_one_contradictory_classifier_frame_keeps_both(self):
        self.choices(self.observation(1))
        self.assertEqual(self.choices(self.observation(1.033,qs=(.98,.98))),{0:{11},1:{22}})
    def test_one_hand_is_not_duplicated(self):
        self.assertEqual(self.choices(self.observation(1,which=(0,))),{0:{11}})
    def test_ambiguous_birth_and_missing_input_do_not_invent_hands(self):
        self.assertEqual(self.choices(self.observation(1,qs=(.5,.5))),{})
        self.choices(self.observation(1.033))
        self.assertEqual(self.choices(self.observation(1.066,which=())),{})
        self.assertEqual(self.choices(self.observation(2,qs=(.5,.5))),{})
    def test_bad_stereo_pair_is_not_fused(self):
        o=self.observation(1,which=(0,))
        o['views'][1]['points'][:,1]+=200
        a,c,_=self.policy.assign(o,np.eye(4))
        self.assertTrue(all(len(c[i])<=1 for i in a.values()))
    def test_repeated_and_old_frames_do_not_refresh_or_reassign(self):
        self.choices(self.observation(1))
        ids={side:t['id'] for side,t in self.policy.tracks.items()}
        for stamp in (1,.99,1):
            self.assertEqual(self.choices(self.observation(stamp,qs=(.98,.02))),{})
        self.assertTrue(all(t['time']==1 for t in self.policy.tracks.values()))
        self.assertEqual(self.choices(self.observation(1.033)),{0:{11},1:{22}})
        self.assertEqual(ids,{side:t['id'] for side,t in self.policy.tracks.items()})

    def test_long_gap_discards_crossing_state_and_track_generation(self):
        self.choices(self.observation(1))
        old_ids={t['id'] for t in self.policy.tracks.values()}
        self.policy.ambiguous=True;self.policy.resolve_streak=2
        self.assertEqual(self.choices(self.observation(2)),{0:{11},1:{22}})
        self.assertFalse(self.policy.ambiguous)
        self.assertEqual(self.policy.resolve_streak,0)
        self.assertTrue(old_ids.isdisjoint(t['id'] for t in self.policy.tracks.values()))

    def test_context_loss_requires_fresh_identity_evidence(self):
        self.choices(self.observation(1))
        for context in (None,np.full((4,4),np.nan)):
            self.assertEqual(self.policy.assign(self.observation(1.01),context),({},{},{}))
            self.assertFalse(self.policy.tracks)
            self.assertEqual(self.choices(self.observation(1.033,qs=(.5,.5))),{})

    def test_bad_observation_cannot_poison_geometry(self):
        o=self.observation(1,which=(0,))
        o['views'][0]['points'][0,0]=np.nan
        o['views'][1]['right']=np.nan
        self.assertEqual(self.choices(o),{})
        self.assertEqual(self.choices(self.observation(1.033)),{0:{11},1:{22}})

    def test_backend_context_loss_clears_identity_and_memory(self):
        from live_backend import Backend
        b=Backend.__new__(Backend)
        b.association='temporal';b.ownership=self.policy;b.previous={0:(1,100)}
        self.choices(self.observation(1))
        result=b.predict(self.observation(1.033),{},None)
        self.assertEqual(result['hands'],[])
        self.assertEqual(b.previous,{})
        self.assertEqual(self.policy.tracks,{})

    def test_bearing_retains_range_only_with_fresh_observation(self):
        self.policy=TemporalAssociation(self.cameras,retain_depth=True)
        self.choices(self.observation(1,which=(0,)))
        old=self.policy.tracks[0]['world'].copy()
        o=self.observation(1.033,which=(0,),offset=20)
        o['views']=[v for v in o['views'] if v['camera']=='slam_left']
        assigned,groups,_=self.policy.assign(o,np.eye(4))
        self.assertEqual(set(assigned),{0})
        self.assertEqual(len(groups[assigned[0]]),1)
        self.assertIsNotNone(self.policy.tracks[0]['world'])
        self.assertGreater(self.policy.tracks[0]['world'][0],old[0]+10)
        self.assertEqual(self.policy.tracks[0]['depth_time'],1)
        self.assertEqual(self.choices(self.observation(1.066,which=())),{})

    def test_bearing_range_expires_during_continued_monocular_tracking(self):
        self.policy=TemporalAssociation(self.cameras,retain_depth=True)
        self.choices(self.observation(1,which=(0,)))
        for stamp in (1.2,1.4,1.6,1.7):
            o=self.observation(stamp,which=(0,))
            o['views']=[v for v in o['views'] if v['camera']=='slam_left']
            self.choices(o)
            if stamp<1.65:self.assertIsNotNone(self.policy.tracks[0]['world'])
        self.assertIsNone(self.policy.tracks[0]['world'])

    def test_bearing_compensates_head_translation(self):
        self.policy=TemporalAssociation(self.cameras,retain_depth=True)
        self.choices(self.observation(1,which=(0,)))
        old=self.policy.tracks[0]['world'].copy()
        o=self.observation(1.033,which=(0,),offset=-20)
        o['views']=[v for v in o['views'] if v['camera']=='slam_left']
        world=np.eye(4);world[0,3]=20
        self.policy.assign(o,world)
        self.assertLess(np.linalg.norm(self.policy.tracks[0]['world']-old),2)

    def test_evaluation_distinguishes_nonhands_from_missing_hands(self):
        from evaluate_association import verdict
        self.assertEqual(verdict(None,None),'rejected_nonhand')
        self.assertEqual(verdict(None,0),'false_positive')
        self.assertEqual(verdict(0,None),'missing')
        self.assertEqual(verdict(0,1),'wrong')
        self.assertEqual(verdict(1,1),'correct')

    def test_stereo_geometry_recovers_metric_palm(self):
        o=self.observation(1,which=(0,))
        a=self.policy._feature(o['views'][0],self.cameras['slam_left'])
        b=self.policy._feature(o['views'][1],self.cameras['slam_right'])
        pair=self.policy._pair(a,b)
        self.assertLess(pair['error'],1e-12)
        self.assertAlmostEqual(pair['point'][2],500,places=6)

if __name__=='__main__':unittest.main()
