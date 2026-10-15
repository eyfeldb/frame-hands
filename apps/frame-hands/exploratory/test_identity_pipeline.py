"""Reset/binding controls for the opt-in live identity path; no hardware access."""
from types import SimpleNamespace as NS
from unittest import TestCase, mock
import numpy as np
import torch
from identity_pipeline import ImageIdentityPipeline


def fake_regressor():
    return NS(device='cpu',model=NS(_temporal=NS(
        _mem_features=torch.ones(2),_prev_extrinsics=torch.ones(2))))


class IdentityPipelineTests(TestCase):
    def pipeline(self):
        p=ImageIdentityPipeline.__new__(ImageIdentityPipeline)
        p.ready=False
        p.backend=NS(previous={0:(1001,42)},ownership=mock.Mock(),
                     prepared_regressor=fake_regressor())
        p.evidence=NS(features=object(),regressor=fake_regressor(),backend=NS())
        return p

    def test_no_cuda_fails_before_model_loading(self):
        with mock.patch('identity_pipeline.torch.cuda.is_available',return_value=False), mock.patch('identity_pipeline.ChiralityEvidence') as load:
            with self.assertRaisesRegex(RuntimeError,'no CPU fallback'):
                ImageIdentityPipeline('unused')
            load.assert_not_called()

    def test_fresh_calibration_required_and_bound_to_every_geometry_consumer(self):
        p=self.pipeline()
        with self.assertRaisesRegex(RuntimeError,'fresh session'):
            p.predict({},np.eye(4),{})
        with self.assertRaises(ValueError):p.bind_calibration({})
        cameras={'left':object(),'right':object()}
        p.bind_calibration(cameras)
        self.assertTrue(p.ready)
        self.assertIs(p.backend.cameras,cameras)
        self.assertIs(p.backend.ownership.cameras,cameras)
        self.assertIs(p.evidence.backend.cameras,cameras)

    def test_reset_clears_pose_identity_features_and_both_neural_memories(self):
        p=self.pipeline();p.reset('Stale transport frame')
        self.assertEqual(p.backend.previous,{})
        p.backend.ownership.reset.assert_called_once()
        self.assertIsNone(p.evidence.features)
        for r in (p.evidence.regressor,p.backend.prepared_regressor):
            self.assertEqual(r.model._temporal._mem_features.numel(),0)
            self.assertEqual(r.model._temporal._prev_extrinsics.numel(),0)
        self.assertEqual(p.last_debug,{'reset_reason':'Stale transport frame'})

    def test_invalid_capture_never_runs_image_evidence(self):
        for world in (None,np.zeros((3,3)),np.full((4,4),np.nan)):
            p=self.pipeline();p.ready=True;p.evidence.score=mock.Mock()
            result=p.predict({'raw_ns':123},world,{})
            self.assertEqual(result['hands'],[])
            p.evidence.score.assert_not_called()
            self.assertIsNone(p.evidence.features)

    def test_preparation_has_no_calibration_dependency_and_warms_both_models(self):
        evidence=mock.Mock();backend=mock.Mock();regressor=mock.Mock()
        with mock.patch('identity_pipeline.torch.cuda.is_available',return_value=True), mock.patch('identity_pipeline.ChiralityEvidence',return_value=evidence) as build_evidence, mock.patch('identity_pipeline.Backend',return_value=backend) as build_backend, mock.patch('identity_pipeline.PreparedRegressor',return_value=regressor):
            p=ImageIdentityPipeline('pinned')
        build_evidence.assert_called_once_with('pinned',{},optimized=True,device='cuda')
        build_backend.assert_called_once_with('pinned',{},association='temporal')
        evidence.regressor.warmup.assert_called_once_with()
        regressor.warmup.assert_called_once_with()
        self.assertFalse(p.ready)
