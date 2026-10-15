"""Explicit CUDA image-identity candidate for supervised mesh trials and replay.

Model preparation needs no calibration, Frame access or XR session. Bind the
new recording's calibration before inference. No recovery or input injection.
"""
import time
import numpy as np
import torch
from chirality_evidence import ChiralityEvidence
from feature_reuse import PreparedRegressor
from live_backend import Backend


class ImageIdentityPipeline:
    def __init__(self, upstream):
        if not torch.cuda.is_available():
            raise RuntimeError('Image identity requires the CUDA environment; no CPU fallback')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        started = time.perf_counter()
        # Geometry is deliberately absent until the fresh recording supplies it.
        self.evidence = ChiralityEvidence(upstream, {}, optimized=True, device='cuda')
        self.backend = Backend(upstream, {}, association='temporal')
        self.backend.prepared_regressor = PreparedRegressor(
            self.backend.model, self.backend.skeleton, device='cuda')
        self.evidence.regressor.warmup()
        self.backend.prepared_regressor.warmup()
        self.ready = False
        self.last_debug = {}
        self.preparation_seconds = time.perf_counter() - started

    def bind_calibration(self, cameras):
        if not cameras:
            raise ValueError('Fresh session calibration is required')
        self.reset('Calibration binding')
        self.backend.cameras = cameras
        self.backend.ownership.cameras = cameras
        self.evidence.backend.cameras = cameras
        self.ready = True

    def reset(self, reason):
        self.backend.previous = {}
        self.backend.ownership.reset()
        self.evidence.features = None
        for regressor in (self.evidence.regressor, self.backend.prepared_regressor):
            temporal = regressor.model._temporal
            temporal._mem_features = torch.empty(0, device=regressor.device)
            temporal._prev_extrinsics = torch.empty(0, device=regressor.device)
        self.last_debug = dict(reset_reason=reason)

    def predict(self, observation, world, prepared):
        if not self.ready:
            raise RuntimeError('Cannot infer without fresh session calibration')
        started = time.perf_counter()
        valid = world is not None and np.shape(world) == (4, 4) and np.isfinite(world).all()
        if not valid:
            self.reset('Invalid capture pose')
            return dict(hands=[], processing_ms=0., reason='No valid capture pose')
        observation, rows, evidence_ms = self.evidence.score(observation, world, prepared)
        result = self.backend.predict(observation, {}, world, prepared,
                                      feature_cache=self.evidence.features)
        pose_ms = result['processing_ms']
        result['processing_ms'] = (time.perf_counter() - started) * 1000
        self.last_debug = dict(views=rows, evidence_ms=evidence_ms, pose_ms=pose_ms,
                              association=self.backend.ownership.last_debug)
        return result
