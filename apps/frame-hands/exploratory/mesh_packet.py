"""Private mesh packet helpers. Vertices are metres, source/model coordinates mm."""
import json
import os
from pathlib import Path
import time
import numpy as np


def packet(prediction, triangles, current_head, sequence, age_ms=0):
    world_to_head=np.linalg.inv(current_head)
    hands=[]
    for p in prediction['hands']:
        vertices=np.array(p['vertices_world_mm'])
        vertices=(vertices@world_to_head[:3,:3].T+world_to_head[:3,3])*.001
        a,b,c=(vertices[triangles[:,i]] for i in range(3))
        face=np.cross(b-a,c-a)
        if p['side']:face*=-1
        normals=np.zeros_like(vertices)
        for i in range(3):np.add.at(normals,triangles[:,i],face)
        normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
        hands.append(dict(side=p['side'],source_id=p['source_id'],vertices_head_m=vertices.tolist(),normals_head=normals.tolist()))
    return dict(version=1,sequence=sequence,capture_wall_s=time.time()-age_ms/1000,age_ms=age_ms,
                hands=hands,triangles=triangles.reshape(-1).tolist())


def publish(path,value):
    path=Path(path)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,separators=(',',':'),allow_nan=False),encoding='utf-8')
    # Godot can briefly hold the old file open. Keep the previous complete packet
    # if Windows denies replacement; never truncate a packet being read.
    try:os.replace(temporary,path)
    except PermissionError:return False
    return True


def rig_data(hand):
    vertices=hand.mesh_vertices.numpy()*.001
    triangles=hand.mesh_triangles.numpy().astype(np.int32)
    a,b,c=(vertices[triangles[:,i]] for i in range(3))
    face=np.cross(b-a,c-a)
    normals=np.zeros_like(vertices)
    for i in range(3):np.add.at(normals,triangles[:,i],face)
    normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
    landmark_weights=np.zeros((21,17))
    for i,(indices,weights) in enumerate(zip(hand.landmark_rest_bone_indices.tolist(),hand.landmark_rest_bone_weights.tolist())):
        for bone,weight in zip(indices,weights):
            if weight:landmark_weights[i,int(bone)]=weight
    return dict(landmarks_m=(hand.landmark_rest_positions.numpy()*.001).tolist(),landmark_weights=landmark_weights.tolist(),version=1,vertices_m=vertices.tolist(),normals=normals.tolist(),triangles=triangles.reshape(-1).tolist(),
        weights=hand.dense_bone_weights.tolist(),axes=hand.joint_rotation_axes.tolist(),rest_m=(hand.joint_rest_positions*.001).tolist(),limits=hand.joint_limits.tolist())


def pose_packet(prediction,current_head,sequence,capture_ns,age_ms=0,context_valid=True):
    head=np.array(current_head).copy();head[:3,3]*=.001
    hands=[]
    for hand in prediction['hands']:
        wrist=np.array(hand['wrist_world_mm']).copy();wrist[:3,3]*=.001
        hands.append(dict(side=hand['side'],source_id=hand['source_id'],wrist_world_m=wrist.tolist(),joint_angles=hand['joint_angles']))
    return dict(version=2,sequence=sequence,capture_ns=capture_ns,published_wall_s=time.time(),age_ms=age_ms,
        context_valid=context_valid,head_world_m=head.tolist(),hands=hands)
