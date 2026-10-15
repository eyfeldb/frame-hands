# Copyright 2021-2022, Collabora, Ltd. (adapted preprocessing)
# SPDX-License-Identifier: BSL-1.0
"""Offline Mercury detector/model feasibility; no live caller or output injection.

Preprocessing follows Monado22d5c936 hg_model.cpp (BSL-1.0, Collabora).
The separate model files stay private and must match their pinned LFS hashes.
"""
import hashlib
from pathlib import Path
import cv2
import numpy as np

MODEL_HASHES={
 'grayscale_detection_160x160.onnx':'1f1a039a266e13dc186bb884430ebd9c8216bdda680ab08a533d4c671f27ed36',
 'grayscale_keypoint_jan18.onnx':'40c0daa598cedb993b54fff17685231b7465d6db342656c401b01c2029efd1d5'}


def normalize(image):
    if image.ndim!=2 or image.dtype!=np.uint8:raise ValueError('Expected grayscale uint8')
    out=image.astype(np.float32)/255.
    mean,std=cv2.meanStdDev(out)
    if std[0,0]==0:return None
    out*=.25/std[0,0]
    mean,_=cv2.meanStdDev(out);out+=.5-mean[0,0]
    return out


def blackbar(image,rotation):
    if rotation not in (0,90,180,270):raise ValueError('Cardinal rotation required')
    h,w=image.shape;ow,oh=(h,w) if rotation in (90,270) else (w,h)
    s=min(160/ow,160/oh);tx=(160-ow*s)/2;ty=(160-oh*s)/2
    # Exact pinned source matrices, including pixel-centre translation.
    matrices={0:[[s,0,tx],[0,s,ty]],90:[[0,s,tx],[-s,0,ty+159]],
              180:[[-s,0,tx+159],[0,-s,-ty+159]],
              270:[[0,-s,-tx+159],[s,0,ty]]}
    forward=np.array(matrices[rotation],np.float32)
    return cv2.warpAffine(image,forward,(160,160)),cv2.invertAffineTransform(forward),forward


class MercuryModels:
    def __init__(self,models):
        import onnxruntime as ort
        self.version=ort.__version__;self.sessions={}
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
        for name,expected in MODEL_HASHES.items():
            path=Path(models)/name
            if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Unverified Mercury model')
            self.sessions[name]=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        self.detector=self.sessions['grayscale_detection_160x160.onnx']
        self.keypoints=self.sessions['grayscale_keypoint_jan18.onnx']

    def detect(self,image,rotation):
        small,back,forward=blackbar(image,rotation);values=normalize(small)
        if values is None:return [],small,dict(reason='No image variance; skipped model')
        exists,cx,cy,size=self.detector.run(['hand_exists','cx','cy','size'],{'inputImg':values[None,None]})
        if not all(np.isfinite(v).all() for v in (exists,cx,cy,size)):raise ValueError('Nonfinite detector output')
        rows=[]
        for side in (0,1):
            center=np.array([(cx[0,side]+1)*80,(cy[0,side]+1)*80,1.])@back.T
            extent=float(size[0,side]*320*np.linalg.norm(back[0,:2]))
            rows.append(dict(model_side=side,presence=float(exists[0,side]),found=bool(exists[0,side]>.3),center_px=center.tolist(),size_px=extent))
        return rows,small,dict(reason='Raw model slots, not verified anatomical identity',forward=forward.tolist())

    def keypoint_schema_smoke(self):
        feed={x.name:np.zeros(x.shape,np.float32) for x in self.keypoints.get_inputs()}
        output=self.keypoints.run(None,feed)
        return dict(inputs=[dict(name=x.name,shape=x.shape,type=x.type) for x in self.keypoints.get_inputs()],outputs=[dict(name=x.name,shape=list(v.shape),finite=bool(np.isfinite(v).all())) for x,v in zip(self.keypoints.get_outputs(),output)],note='Zero-tensor execution only; no camera crop or tracking quality claim.')
