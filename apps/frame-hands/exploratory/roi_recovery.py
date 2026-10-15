"""Causal, opt-in offline ROI recovery. No device/driver/publication operations."""
import copy
from dataclasses import dataclass
import numpy as np
import torch
from frame_data import warp,NAMES
from probe_roi_continuation import pose_crop
from probe_roi_presence import verify_hand,PresenceModel
from score_roi_presence import supported_views
from feature_reuse import PreparedRegressor
from replay_umetrack import COMMON


def pose_state(hand):
    # Retain only recurrent crop geometry; copying 788 mesh vertices each frame
    # adds avoidable CPU latency and is unnecessary for subsequent inference.
    return {k:copy.deepcopy(hand[k]) for k in ('side','source_id','joint_angles','wrist_world_mm')}


@dataclass
class Slot:
    hand: dict
    anchor_ns: int
    supported_ns: int
    prediction_ns: int=0


class RecoveryState:
    # Conservative initial engineering limits, frozen before full replay.
    MAX_UNSUPPORTED_NS=650_000_000
    MAX_ANCHOR_AGE_NS=2_000_000_000
    MAX_FRAME_GAP_NS=100_000_000

    def __init__(self):self.slots={};self.stamp=None

    def begin(self,stamp,valid,anchors):
        if self.stamp is not None and stamp<=self.stamp:
            raise ValueError('Recovery requires strictly increasing captures')
        if not valid or (self.stamp is not None and stamp-self.stamp>self.MAX_FRAME_GAP_NS):self.slots={}
        self.stamp=stamp
        if not valid:return {}
        sides=[h['side'] for h in anchors]
        if len(set(sides))!=len(sides) or any(s not in (0,1) for s in sides):raise ValueError('Duplicate or invalid anchor side')
        for h in anchors:
            side=h['side'];previous=self.slots.get(side)
            same_generation=previous is not None and previous.hand['source_id']==h['source_id']
            stereo=len({v['camera'] for v in h.get('source_views',[])})>=2
            # A one-camera birth can be a watch/background false positive. It
            # remains baseline output but cannot establish a recovery lineage.
            if stereo or same_generation:self.slots[side]=Slot(pose_state(h),stamp,stamp)
            else:self.slots.pop(side,None)
        self.slots={s:v for s,v in self.slots.items() if stamp-v.supported_ns<=self.MAX_UNSUPPORTED_NS and stamp-v.anchor_ns<=self.MAX_ANCHOR_AGE_NS}
        return {s:v for s,v in self.slots.items() if s not in sides}

    def finish(self,proposals,accepted):
        for h in proposals:
            side=h['side']
            if side not in self.slots:raise ValueError('A proposal cannot create a hand')
            slot=self.slots[side]
            if h['source_id']!=slot.hand['source_id']:raise ValueError('Proposal cannot change anatomical generation')
            slot.hand=pose_state(h);slot.prediction_ns=self.stamp
            if side in accepted:slot.supported_ns=self.stamp


def exclusive_support(rows,own,others):
    """Require that verifier palm is closer to this proposal than any other slot."""
    palm=[0,5,9,13,17];accepted=[]
    for row in supported_views(rows):
        camera=row['camera'];observed=np.asarray(row['model_points'])[palm]
        if camera not in own:continue
        distance=float(np.median(np.linalg.norm(observed-own[camera][palm],axis=1)))
        if not np.isfinite(distance):continue
        conflict=False
        for other in others:
            if camera not in other:continue
            alternative=float(np.median(np.linalg.norm(observed-other[camera][palm],axis=1)))
            if np.isfinite(alternative) and alternative<=distance:conflict=True;break
        if not conflict:accepted.append(row)
    return accepted


class RoiRecovery:
    def __init__(self,backend,models,device='cpu',reuse_crop_geometry=False):
        self.backend=backend;self.state=RecoveryState();self.reuse_crop_geometry=reuse_crop_geometry
        self.regressor=PreparedRegressor(backend.model,backend.skeleton,device=device)
        self.regressor.warmup();self.verifier=PresenceModel(models)

    def project(self,hand,world):
        from lib.common.hand_skinning import skin_landmarks
        wrist=torch.tensor(hand['wrist_world_mm'],dtype=torch.float32)
        if hand['side']:wrist[:,0]*=-1
        with torch.inference_mode():xyz=skin_landmarks(self.backend.hand,torch.tensor(hand['joint_angles']),wrist).numpy()
        result={}
        for name,base in self.backend.cameras.items():
            camera=base.copy(camera_to_world_xf=world@base.camera_to_world_xf)
            eye=camera.world_to_eye(xyz)
            if (eye[:,2]<=0).any():continue
            projected=np.full((21,2),np.nan);uv=camera.eye_to_window(eye)
            for native,mp in COMMON:projected[mp]=uv[native]
            result[name]=projected
        return result

    def step(self,stamp,world,images,anchors,control='real'):
        candidates=self.state.begin(stamp,world is not None,anchors)
        if world is None:return [],dict(proposals=[],evidence=[],decisions=[],attempted=0)
        if not candidates:return list(anchors),dict(proposals=[],evidence=[],decisions=[],attempted=0)
        b=self.backend;crops=[];intrinsics=[];extrinsics=[];ranges=[];included=[];memory=[];view_names=[]
        cameras={name:c.copy(camera_to_world_xf=world@c.camera_to_world_xf) for name,c in b.cameras.items()}
        for side,slot in sorted(candidates.items()):
            first=len(crops);names=[];prepared_points=None
            if self.reuse_crop_geometry:
                from lib.tracker.perspective_crop import _get_crop_points_from_hand_pose
                from lib.tracker.tracking_result import SingleHandPose
                prior=SingleHandPose(np.array(slot.hand['joint_angles'],np.float32),np.array(slot.hand['wrist_world_mm'],np.float32),hand_confidence=0.)
                with torch.inference_mode():prepared_points=_get_crop_points_from_hand_pose(b.hand,prior,side,63)
            for name in NAMES:
                if name not in images:continue
                try:crop=pose_crop(b,slot.hand,cameras[name],world,prepared_points)
                except ValueError:crop=None
                if crop is None:continue
                pixels=warp(cameras[name],crop,images[name][0])
                if control=='zero':pixels.fill(0)
                crops.append(pixels);intrinsics.append(crop.uv_to_window_matrix());xf=np.linalg.inv(crop.camera_to_world_xf);xf[:3,3]*=.001;extrinsics.append(xf);names.append(name)
            if len(crops)>first:
                included.append(side);ranges.append((first,len(crops)));view_names.append(names)
                memory.append(slot.prediction_ns>0 and 0<stamp-slot.prediction_ns<=RecoveryState.MAX_FRAME_GAP_NS)
        proposals=[]
        if included:
            frame=b.Frame(torch.tensor(np.stack(crops)/255.,dtype=torch.float32),torch.tensor(np.stack(intrinsics),dtype=torch.float32),torch.tensor(np.stack(extrinsics),dtype=torch.float32))
            sides=torch.tensor(included);desc=b.Desc(torch.tensor(ranges),sides,torch.tensor(memory),sides)
            with torch.inference_mode():
                result=self.regressor.regress(frame,desc,self.regressor.extract(frame.left_images))
                for i,side in enumerate(included):
                    wrist=result.wrist_xfs[i].clone();wrist[:3,3]*=1000;angles=result.joint_angles[i];sigma=result.landmark_uncertainty_sigmas[i]
                    proposals.append(dict(side=side,source_id=candidates[side].hand['source_id'],memory=memory[i],hypothesis_only=True,observation_origin='roi_proposal',source_views=[dict(camera=name,origin='previous-pose ROI') for name in view_names[i]],vertices_world_mm=b.mesh(angles,wrist,side).tolist(),joint_angles=angles.tolist(),wrist_world_mm=wrist.tolist(),uncertainty_raw=sigma.tolist()))
        projections={h['side']:self.project(h,world) for h in anchors};evidence={}
        for h in proposals:
            evidence[h['side']],projections[h['side']]=verify_hand(self.verifier,b.hand,h,b.cameras,images,world,control)
        accepted=[];decisions=[]
        for h in proposals:
            side=h['side'];rows=evidence[side]
            support=exclusive_support(rows,projections[side],[v for s,v in projections.items() if s!=side])
            decisions.append(dict(side=side,source_id=h['source_id'],supported=bool(support),presence_supported=bool(supported_views(rows)),supported_cameras=[r['camera'] for r in support],anchor_age_ms=(stamp-candidates[side].anchor_ns)/1e6,unsupported_age_ms=(stamp-candidates[side].supported_ns)/1e6))
            if support:
                output=copy.deepcopy(h);output['hypothesis_only']=False;output['observation_origin']='roi_image_verified'
                accepted.append(output)
        self.state.finish(proposals,{h['side'] for h in accepted})
        return list(anchors)+accepted,dict(proposals=proposals,evidence=[r for rows in evidence.values() for r in rows],decisions=decisions,attempted=len(candidates))
