"""Offline independent landmark-model evidence for saved ROI pose hypotheses.

No admission threshold or live publication. Presence is uncalibrated; crop
geometry comes from the hypothesis, so this is not independent ground truth.
"""
import argparse,hashlib,json,sys,time,shutil
from pathlib import Path
import cv2,numpy as np,torch
from frame_data import Recording,load_calibration
from replay_umetrack import COMMON
from run_mesh_trial import pose_matrix

# Native roi_from_points subset, excluding nonhomologous thumb CMC (MP1).
ROI_POINTS=[0,2,3,5,6,9,10,13,14,17,18]


def disagreement_details(observed, projected, crop_size):
    """Report finger-specific disagreement; none of these values is accuracy.

    Median joint agreement can conceal all five wrong fingertips. Preserve the
    existing admission metric and expose that blind spot without a new gate.
    """
    observed=np.asarray(observed,float);projected=np.asarray(projected,float)
    indices=[mp for _,mp in COMMON];tips=[4,8,12,16,20];palm=[0,5,9,13,17]
    if observed.shape!=(21,2) or projected.shape!=(21,2) or not np.isfinite(crop_size) or crop_size<=0:
        raise ValueError('Invalid disagreement geometry')
    error=np.linalg.norm(observed[indices]-projected[indices],axis=1)
    if not np.isfinite(error).all():raise ValueError('Nonfinite common landmarks')
    tip_error=np.linalg.norm(observed[tips]-projected[tips],axis=1)/crop_size
    palm_error=np.linalg.norm(observed[palm]-projected[palm],axis=1)/crop_size
    return dict(median_disagreement_px=float(np.median(error)),
        normalized_disagreement=float(np.median(error)/crop_size),
        tip_disagreement_fraction=tip_error.tolist(),
        max_tip_disagreement_fraction=float(tip_error.max()),
        median_palm_disagreement_fraction=float(np.median(palm_error)))


def crop_matrix(points):
    points=np.asarray(points,float)
    wrist=points[0];middle=((points[5]+points[13])*.5+points[9])*.5
    delta=middle-wrist
    if not np.isfinite(points[ROI_POINTS]).all() or np.linalg.norm(delta)<1e-5:
        raise ValueError('Invalid projected hand geometry')
    rotation=np.pi/2-np.arctan2(-delta[1],delta[0])
    c,s=np.cos(rotation),np.sin(rotation);rotate=np.array([[c,-s],[s,c]])
    bounds=points[ROI_POINTS];center=(bounds.min(0)+bounds.max(0))*.5
    local=(bounds-center)@rotate
    lo,hi=local.min(0),local.max(0);extent=hi-lo
    center+=rotate@((lo+hi)*.5+np.array([0.,-.1*extent[1]]))
    size=float(extent.max()*2.)
    if not 4<=size<=4096:raise ValueError('Implausible crop extent')
    matrix=np.column_stack((rotate*(size/224),center-rotate@np.array([size/2,size/2])))
    return matrix.astype(np.float32),size


class PresenceModel:
    def __init__(self,models):
        import ncnn
        self.ncnn=ncnn;self.version=ncnn.__version__
        self.net=ncnn.Net();self.net.opt.num_threads=1
        self.net.opt.use_vulkan_compute=False
        self.net.opt.use_fp16_packed=False;self.net.opt.use_fp16_storage=False;self.net.opt.use_fp16_arithmetic=False
        for kind,loader in [('param',self.net.load_param),('bin',self.net.load_model)]:
            if loader(str(models/f'hand.ncnn.{kind}')):raise ValueError('Could not load hand model')

    def infer(self,pixels):
        pixels=np.ascontiguousarray(pixels,dtype=np.uint8)
        if pixels.shape!=(224,224):raise ValueError('Expected 224-square grayscale crop')
        ncnn=self.ncnn
        mat=ncnn.Mat.from_pixels(pixels,ncnn.Mat.PixelType.PIXEL_GRAY2RGB,224,224)
        mat.substract_mean_normalize([], [1/255]*3)
        ex=self.net.create_extractor()
        if ex.input('in0',mat):raise ValueError('Input rejected')
        outputs=[]
        for name in ('out0','out1','out2'):
            status,result=ex.extract(name)
            if status:raise ValueError('Output extraction failed')
            outputs.append(np.array(result).copy().reshape(-1))
        points,presence,right=outputs
        if not all(np.isfinite(x).all() for x in outputs):raise ValueError('Nonfinite model result')
        return points.reshape(21,3)[:,:2],float(presence[0]),float(right[0])


def verify_hand(model,hand_model,h,cameras,images,world,control='real'):
    """Current-image evidence only; no state or acceptance decision."""
    from lib.common.hand_skinning import skin_landmarks
    wrist=torch.tensor(h['wrist_world_mm'],dtype=torch.float32)
    if h['side']:wrist[:,0]*=-1
    with torch.inference_mode():xyz=skin_landmarks(hand_model,torch.tensor(h['joint_angles']),wrist).numpy()
    results=[];projections={}
    for name,base in cameras.items():
        if name not in images:continue
        cam=base.copy(camera_to_world_xf=world@base.camera_to_world_xf)
        eye=cam.world_to_eye(xyz);uv=cam.eye_to_window(eye)
        if (eye[:,2]<=0).any():continue
        projected=np.full((21,2),np.nan)
        for native,mp in COMMON:projected[mp]=uv[native]
        projections[name]=projected
        try:matrix,size=crop_matrix(projected)
        except ValueError:continue
        pixels=cv2.warpAffine(images[name][0],matrix,(224,224),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        if control=='zero':pixels.fill(0)
        started=time.perf_counter();lm,presence,right=model.infer(pixels);elapsed=(time.perf_counter()-started)*1000
        image_points=np.column_stack((lm,np.ones(21)))@matrix.T
        details=disagreement_details(image_points,projected,size)
        results.append(dict(side=h['side'],source_id=h['source_id'],camera=name,input_control=control,presence=presence,right=right,**details,crop_size_px=size,processing_ms=elapsed,model_points=image_points.tolist()))
    return results,projections


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('session','predictions','upstream','models','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.upstream));torch.set_num_threads(1)
    from lib.common.hand import HandModel
    from lib.common.hand_skinning import skin_landmarks
    raw=json.loads((a.upstream/'dataset/generic_hand_model.json').read_text())
    hand=HandModel(**{k:torch.tensor(v) if isinstance(v,list) else v for k,v in raw.items()})
    rows={}
    for line in a.predictions.open():
        r=json.loads(line)
        if r['hands'] and r['capture_pose']['valid']:rows[r['capture_raw_ns']]=r
    if not rows:raise ValueError('No hypotheses to inspect')
    a.output.mkdir(parents=True,exist_ok=False);snapshot=a.output/'snapshot';snapshot.mkdir()
    for path in Path(__file__).parent.glob('*.py'):shutil.copy2(path,snapshot/path.name)
    model=PresenceModel(a.models);calibration=a.session/'calibration'
    cameras=load_calibration(calibration if calibration.exists() else a.session/'frame/snapshot/device/persist')
    results=[];rec=Recording(a.session/'frame/sets.bin');times=[r['seconds'] for r in rows.values()]
    for seconds,stamp,images in rec.frames(min(times)-.001,max(times)+.001):
        if stamp not in rows:continue
        r=rows[stamp];world=pose_matrix(r['capture_pose']['capture'])
        for h in r['hands']:
            evidence,_=verify_hand(model,hand,h,cameras,images,world,r.get('input_control','real'))
            results.extend(dict(seconds=seconds,stamp=stamp,**view) for view in evidence)
    (a.output/'rows.json').write_text(json.dumps(results)+'\n')
    summary=dict(scope=__doc__,views=len(results),ncnn=model.version,model_sha256={kind:hashlib.sha256((a.models/f'hand.ncnn.{kind}').read_bytes()).hexdigest() for kind in ('param','bin')},args={k:str(v) for k,v in vars(a).items()},source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),per_side={str(side):dict(views=sum(r['side']==side for r in results),presence_p05_p50_p95=np.percentile([r['presence'] for r in results if r['side']==side],[5,50,95]).tolist(),disagreement_fraction_p05_p50_p95=np.percentile([r['normalized_disagreement'] for r in results if r['side']==side],[5,50,95]).tolist()) for side in (0,1) if any(r['side']==side for r in results)})
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
