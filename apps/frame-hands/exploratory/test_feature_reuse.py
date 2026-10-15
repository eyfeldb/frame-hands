"""Numerical and recurrent-state controls for optional same-frame feature reuse."""
import unittest
from pathlib import Path
import numpy as np
import torch
import test_chirality as fixtures
from chirality_evidence import ChiralityEvidence
from live_backend import Backend

class FeatureReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.ChiralityTests.setUpClass()
        cls.fixture=fixtures.ChiralityTests
        upstream=Path(__file__).resolve().parents[3]/'.local/hand-alternatives/UmeTrack'
        cls.reference=cls.fixture.model
        cls.optimized=ChiralityEvidence(upstream,cls.reference.backend.cameras,optimized=True)
        cls.pose_reference=Backend(upstream,cls.reference.backend.cameras,association='temporal')
        cls.pose_optimized=Backend(upstream,cls.reference.backend.cameras,association='temporal')

    def test_stereo_mono_memory_and_absence_match_original(self):
        f=self.fixture;memory_observations=0;max_vertex_error=0.
        for step,selection in enumerate((None,None,[0,1,2],[1,3],[0,2],[],None,None,[0],None)):
            views=f.observation['views'] if selection is None else [f.observation['views'][i] for i in selection]
            obs=dict(f.observation,views=views,raw_ns=f.observation['raw_ns']+step*33_333_333)
            world=f.world.copy();world[0,3]+=step*2
            a,da,_=self.reference.score(obs,world,f.prepared)
            b,db,_=self.optimized.score(obs,world,f.prepared)
            for ra,rb in zip(da,db):
                np.testing.assert_allclose(ra['reprojection_px'],rb['reprojection_px'],atol=.003)
                np.testing.assert_allclose(ra['mean_raw_sigmas'],rb['mean_raw_sigmas'],atol=1e-5)
                self.assertEqual(ra['agreement'],rb['agreement'])
            pa=self.pose_reference.predict(a,{},world,f.prepared)
            pb=self.pose_optimized.predict(b,{},world,f.prepared,feature_cache=self.optimized.features)
            self.assertEqual(len(pa['hands']),len(pb['hands']))
            for ha,hb in zip(pa['hands'],pb['hands']):
                for key in ('side','source_id','memory','source_views'):self.assertEqual(ha[key],hb[key])
                memory_observations+=int(ha['memory'])
                error=float(np.max(np.abs(np.array(ha['vertices_world_mm'])-hb['vertices_world_mm'])))
                max_vertex_error=max(max_vertex_error,error)
                self.assertLess(error,.05)
                np.testing.assert_allclose(ha['joint_angles'],hb['joint_angles'],atol=.0002)
        self.assertGreater(memory_observations,0)
        print(f'Feature reuse recurrent comparison: max vertex error {max_vertex_error:.6f} mm, {memory_observations} memory observations')

    @unittest.skipUnless(torch.cuda.is_available(),'Separate CUDA research environment required')
    def test_cuda_recurrent_path_matches_cpu(self):
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        upstream=Path(__file__).resolve().parents[3]/'.local/hand-alternatives/UmeTrack'
        cameras=self.reference.backend.cameras
        self.reference=ChiralityEvidence(upstream,cameras)
        self.optimized=ChiralityEvidence(upstream,cameras,optimized=True,device='cuda')
        self.pose_reference=Backend(upstream,cameras,association='temporal')
        self.pose_optimized=Backend(upstream,cameras,association='temporal')
        self.optimized.regressor.warmup()
        self.assertEqual(self.optimized.backend.model._temporal._mem_features.numel(),0)
        self.assertEqual(self.optimized.backend.model._temporal._prev_extrinsics.numel(),0)
        self.test_stereo_mono_memory_and_absence_match_original()

    def test_cache_rejects_other_frame_or_changed_pixels(self):
        f=self.fixture
        self.optimized.score(f.observation,f.world,f.prepared)
        cache=self.optimized.features
        with self.assertRaisesRegex(ValueError,'another capture'):
            cache.select(cache.stamp+1,cache.keys,cache.images)
        with self.assertRaisesRegex(ValueError,'pixels changed'):
            cache.select(cache.stamp,cache.keys,cache.images+1/255)
        with self.assertRaises(ValueError):
            cache.select(cache.stamp,[('unknown',-1,0)],cache.images[:1])

if __name__=='__main__':unittest.main()
