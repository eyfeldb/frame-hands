"""Batch equivalence and memory isolation for image-based identity evidence."""
import io,json,sys,unittest
from pathlib import Path
import cv2,numpy as np
from chirality_evidence import ChiralityEvidence
from replay_mesh_trial import packets
from frame_data import read_audit,load_calibration
from run_mesh_trial import pose_matrix

class ChiralityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[3]
        upstream=root/'.local/hand-alternatives/UmeTrack';sys.path.insert(0,str(upstream))
        session=root/'.local/hand-alternatives/results/live-mesh-20261005T050930Z'
        with (session/'predictions.jsonl').open() as stream:
            records={r['capture_raw_ns']:r for l in stream if (r:=json.loads(l))}
        for meta,blobs in packets(session/'transport.bin'):
            if len(meta.get('crops',[]))<4 or meta.get('stamp') not in records:continue
            reference=records[meta['stamp']]
            if not reference['capture_pose']['valid']:continue
            cls.observation=next(iter(read_audit(io.StringIO(meta['audit'])).values()))
            cls.prepared={(v['camera'],v['id']):(cv2.imdecode(np.frombuffer(b,np.uint8),0),v) for v,b in zip(meta['crops'],blobs)}
            cls.world=pose_matrix(reference['capture_pose']['capture']);break
        else:raise RuntimeError('No complete private stereo fixture')
        cls.model=ChiralityEvidence(upstream,load_calibration(session/'calibration'))

    def test_disagreeing_image_scores_abstain(self):
        q,agreed,_,_=ChiralityEvidence.combine([5.,50.],[.04,.01])
        self.assertEqual(q,.5);self.assertFalse(agreed)
        q,agreed,_,_=ChiralityEvidence.combine([5.,50.],[.01,.04])
        self.assertLess(q,.1);self.assertTrue(agreed)

    def test_batched_scores_match_independent_views(self):
        _,batch,_=self.model.score(self.observation,self.world,self.prepared)
        for v,reference in zip(self.observation['views'],batch):
            _,single,_=self.model.score(dict(self.observation,views=[v]),self.world,self.prepared)
            self.assertTrue(np.allclose(single[0]['reprojection_px'],reference['reprojection_px'],atol=.002))
            self.assertTrue(np.allclose(single[0]['mean_raw_sigmas'],reference['mean_raw_sigmas'],atol=1e-5))
            self.assertEqual(single[0]['agreement'],reference['agreement'])

    def test_reordering_and_intervening_frames_do_not_carry_memory(self):
        _,reference,_=self.model.score(self.observation,self.world,self.prepared)
        moved=self.world.copy();moved[0,3]+=100
        reverse=dict(self.observation,views=list(reversed(self.observation['views'])))
        self.model.score(reverse,moved,self.prepared)
        _,again,_=self.model.score(self.observation,self.world,self.prepared)
        for a,b in zip(reference,again):
            self.assertTrue(np.allclose(a['reprojection_px'],b['reprojection_px'],atol=.002))
            self.assertAlmostEqual(a['right'],b['right'],places=5)

if __name__=='__main__':unittest.main()
