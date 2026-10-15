"""Reusable opt-in UmeTrack inference and native mesh skinning; no driver I/O."""
import hashlib
import json
from pathlib import Path
import sys
from runtime_metadata import installed_revision
import time
import numpy as np
import torch
from frame_data import Ownership, NAMES, crop_camera
from replay_umetrack import warp


class Backend:
    def __init__(self, upstream, cameras, threads=4, association="fresh"):
        from prepare_umetrack import COMMIT
        if installed_revision(upstream)!=COMMIT:
            raise ValueError('Unpinned UmeTrack source')
        if hashlib.sha256((Path(upstream)/'dataset/generic_hand_model.json').read_bytes()).hexdigest()!='96649ddf7d036be3357dadd7c13db325a18638c6713583fbbbdd71446ce07564':
            raise ValueError('Unexpected generic hand mesh')
        sys.path.insert(0, str(Path(upstream).resolve()))
        from lib.models.model_loader import load_pretrained_model
        from lib.models.umetrack_model import InputFrameData, InputFrameDesc, InputSkeletonData
        from lib.common.hand import HandModel
        from lib.common.hand_skinning import _skin_points, skin_landmarks
        weights = Path(upstream)/'pretrained_models/pretrained_weights.torch'
        if hashlib.sha256(weights.read_bytes()).hexdigest() != '72eab0ea2c5c4518188ede4cf62e98f62eb1e4ef13628c0f20ad3fc56d1bf18d':
            raise ValueError('Unpinned model weights')
        torch.set_num_threads(threads)
        self.model = load_pretrained_model(str(weights)).eval()
        raw = json.loads((Path(upstream)/'dataset/generic_hand_model.json').read_text())
        self.hand = HandModel(**{k: torch.tensor(v) if isinstance(v, list) else v for k, v in raw.items()})
        self.skeleton = InputSkeletonData(self.hand.joint_rotation_axes.float(), self.hand.joint_rest_positions.float()*.001)
        self.Frame, self.Desc = InputFrameData, InputFrameDesc
        self.skin, self.landmarks = _skin_points, skin_landmarks
        self.triangles = self.hand.mesh_triangles.numpy().astype(np.int32)
        self.cameras, self.ownership, self.previous = cameras, Ownership(), {}
        if association not in ('fresh','temporal','bearing','continuity'):raise ValueError('Unknown association mode')
        self.association=association
        if association in ('temporal','bearing','continuity'):
            from temporal_association import TemporalAssociation
            self.ownership=TemporalAssociation(cameras,retain_depth=association=='bearing',reattach_prior=association=='continuity')

    def mesh(self, angles, wrist, side):
        wrist = torch.as_tensor(wrist, dtype=torch.float32).clone()
        if side:
            wrist[:, 0] *= -1
        with torch.inference_mode():
            vertices = self.skin(self.hand.joint_rest_positions, self.hand.joint_rotation_axes,
                self.hand.dense_bone_weights.unsqueeze(0), torch.as_tensor(angles, dtype=torch.float32),
                self.hand.mesh_vertices, wrist).numpy()
        if not np.isfinite(vertices).all():
            raise ValueError('Nonfinite mesh')
        return vertices

    def predict(self, observation, images, world, prepared=None, feature_cache=None):
        started = time.perf_counter()
        stamp = observation['raw_ns']
        if world is None or (self.association in ('temporal','bearing','continuity') and (np.shape(world)!=(4,4) or not np.isfinite(world).all())):
            self.previous = {}
            if self.association in ('temporal','bearing','continuity'):self.ownership.reset()
            return dict(hands=[], processing_ms=0, reason='No valid capture pose')
        cameras = {n:c.copy(camera_to_world_xf=world@c.camera_to_world_xf) for n,c in self.cameras.items()}
        assigned, candidates, scores = self.ownership.assign(observation,world) if self.association in ('temporal','bearing','continuity') else self.ownership.assign(observation)
        if self.association in ('temporal','bearing','continuity') and self.ownership.last_debug.get('reason'):
            return dict(hands=[],processing_ms=(time.perf_counter()-started)*1000,reason=self.ownership.last_debug['reason'])
        crops, intrinsics, extrinsics, ranges, included, memory = [], [], [], [], [], []
        feature_keys=[]
        for side, ident in sorted(assigned.items()):
            first = len(crops)
            for v in sorted(candidates[ident], key=lambda v:NAMES.index(v['camera'])):
                name = v['camera']
                if prepared is not None:
                    item=prepared.get((name,v['id']))
                    if item is None:continue
                    image,meta=item
                    crops.append(image[:,::-1].copy() if side else image)
                    intrinsics.append(meta['intrinsics'])
                    crop_xf=world@np.array(meta['camera_to_head_mm'])
                    if side:crop_xf[:,0]*=-1
                else:
                    if name not in images or images[name].shape != (cameras[name].height,cameras[name].width):continue
                    try:crop=crop_camera(cameras[name],v['points'],side,world)
                    except ValueError:continue
                    crops.append(warp(cameras[name],crop,images[name]))
                    intrinsics.append(crop.uv_to_window_matrix())
                    crop_xf=crop.camera_to_world_xf
                xf=np.linalg.inv(crop_xf)
                xf[:3,3]*=.001
                extrinsics.append(xf)
                feature_keys.append((name,v['id'],side))
            if len(crops) == first:
                continue
            included.append((side,ident))
            ranges.append((first,len(crops)))
            old = self.previous.get(side)
            memory.append(bool(old and old[0] == ident and 0 < stamp-old[1] <= 100_000_000))
        predictions = []
        if included:
            data = self.Frame(torch.tensor(np.stack(crops)/255.,dtype=torch.float32),
                torch.tensor(np.stack(intrinsics),dtype=torch.float32),torch.tensor(np.stack(extrinsics),dtype=torch.float32))
            indices = torch.tensor([s for s,_ in included])
            desc = self.Desc(torch.tensor(ranges),indices,torch.tensor(memory),indices)
            with torch.inference_mode():
                if feature_cache is None:
                    inferred = self.model.regress_pose_use_skeleton(data,desc,self.skeleton)
                else:
                    from feature_reuse import PreparedRegressor
                    if not hasattr(self,'prepared_regressor'):self.prepared_regressor=PreparedRegressor(self.model,self.skeleton,device=feature_cache.features.device)
                    features=feature_cache.select(stamp,feature_keys,data.left_images)
                    inferred=self.prepared_regressor.regress(data,desc,features)
                for i,(side,ident) in enumerate(included):
                    wrist = inferred.wrist_xfs[i].clone()
                    wrist[:3,3] *= 1000
                    angles = inferred.joint_angles[i]
                    vertices = self.mesh(angles,wrist,side)
                    predictions.append(dict(side=side,source_id=ident,memory=memory[i],right_score=scores[ident],
                        vertices_world_mm=vertices.tolist(),joint_angles=angles.tolist(),wrist_world_mm=wrist.tolist(),
                        uncertainty_raw=inferred.landmark_uncertainty_sigmas[i].tolist()))
        if self.association in ('temporal','bearing','continuity'):
            for p in predictions:
                p['source_views']=[dict(camera=v['camera'],source_id=v['id']) for v in candidates[p['source_id']]]
        self.previous = {p['side']:(p['source_id'],stamp) for p in predictions}
        return dict(hands=predictions,processing_ms=(time.perf_counter()-started)*1000)
