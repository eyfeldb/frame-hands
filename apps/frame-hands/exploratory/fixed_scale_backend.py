"""Offline fixed-skeleton-scale ablation; no live launcher imports this class."""
import numpy as np
from live_backend import Backend


class ScaledBackend(Backend):
    def __init__(self,upstream,cameras,hand_scale=1.,**kwargs):
        if not np.isfinite(hand_scale) or not .5<=hand_scale<=1.5:
            raise ValueError('Fixed hand scale must be finite and within 0.5..1.5')
        super().__init__(upstream,cameras,**kwargs)
        self.hand_scale=float(hand_scale)
        if self.hand_scale!=1.:
            from lib.common.hand import scaled_hand_model
            self.hand=scaled_hand_model(self.hand,self.hand_scale)
            # Condition the pose regressor on the same skeleton used for skinning.
            # Resizing final mesh vertices alone would be a different experiment.
            self.skeleton=type(self.skeleton)(self.hand.joint_rotation_axes.float(),self.hand.joint_rest_positions.float()*.001)
