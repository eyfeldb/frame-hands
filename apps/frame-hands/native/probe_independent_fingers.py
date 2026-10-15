"""Independent grayscale keypoint evidence on exact native-tracker failure crops.

Sparse host CPU diagnostic, not a model replacement or anatomical ground truth.
"""
import argparse,json,sys,time,hashlib,platform
from pathlib import Path
import numpy as np,cv2
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE.parent/'exploratory'))
from frame_data import Recording,read_audit,load_calibration
from mercury_probe import MercuryModels,MODEL_HASHES
from mercury_keypoints import StereoCrop,estimate
from probe_overlap_acquisition import affine
from audit_native_lengths import sha
from audit_crop_expansion import epi
from review_native_landmarks import MP_SEGMENTS

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('session','crop-review','trace','models','runtime','upstream','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();sys.path.insert(0,str(a.runtime.resolve()));sys.path.insert(0,str(a.upstream));a.output.mkdir(exist_ok=False,parents=True)
 review=json.loads((a.crop_review/'result.json').read_text());traced=json.loads((a.trace/'result.json').read_text());assert traced['outputs_match'];assert sha(a.trace/'result.json')==review['trace_result_sha256']
 rec=Recording(a.session/'frame/sets.bin');cams=load_calibration(a.session/'frame/snapshot/device/persist');audits=read_audit(a.trace/'views.txt');models=MercuryModels(a.models);rows=[]
 for selected in review['review']:
  assert [v['camera'] for v in selected['crops']]==['slam_left','slam_right']
  display_label='raised-finger-conflict' if selected['label']=='visible-fist' else selected['label']
  sec=selected['seconds'];raw=selected['raw_ns'];frames=list(rec.frames(sec-1e-7,sec+1e-7));assert len(frames)==1 and frames[0][1]==raw;pair=[];canvas=np.full((740,1280,3),24,np.uint8);cv2.putText(canvas,f"{display_label} | {sec:.3f}s side{selected['side']}",(8,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(240,240,240),1,cv2.LINE_AA)
  for view_index,entry in enumerate(selected['crops']):
   name=entry['camera'];roi=entry['roi'];image=frames[0][2][name][0];camera=cams[name];crop=StereoCrop(camera,roi[:2],roi[2],selected['side'])
   if not rows and view_index==0:
    for _ in range(3):estimate(models,crop,image)
   start=time.perf_counter();result,pixels=estimate(models,crop,image);elapsed=(time.perf_counter()-start)*1000
   assert 'points_raw' in result;zero,_=estimate(models,crop,np.zeros_like(image));assert not zero['active']
   native=next(v for v in audits[raw]['views'] if v['camera']==name and v['id']==selected['source'])
   matrix=affine(roi[:2],roi[2]*1.3,roi[3],300);inverse=cv2.invertAffineTransform(matrix);display=cv2.warpAffine(image,matrix,(300,300),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
   for col in range(4):
    pic=cv2.cvtColor(pixels if col==3 else display,cv2.COLOR_GRAY2BGR)
    if col in (1,2):
     points=np.array(native['points'] if col==1 else result['points_raw']);pp=np.c_[points,np.ones(21)]@inverse.T
     for i,j in MP_SEGMENTS:cv2.line(pic,tuple(np.rint(pp[i]).astype(int)),tuple(np.rint(pp[j]).astype(int)),(230,170,60) if col==1 else (60,230,70),2,cv2.LINE_AA)
     for finger,tip in enumerate((8,12,16,20)):cv2.putText(pic,('I','M','R','L')[finger],tuple(np.rint(pp[tip]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1,cv2.LINE_AA)
    canvas[65+view_index*340:365+view_index*340,col*320:col*320+300]=cv2.resize(pic,(300,300));label=('image',f"native {native['presence']:.3f}",f"Mercury {result['presence']:.3f}",'Mercury input')[col];cv2.putText(canvas,label,(col*320+3,53+view_index*340),cv2.FONT_HERSHEY_SIMPLEX,.47,(240,240,240),1,cv2.LINE_AA)
   cv2.imwrite(str(a.output/(selected['label']+'-'+name+'-input.png')),pixels)
   pair.append(dict(camera=name,native_points=native['points'].tolist(),native_presence=native['presence'],roi=roi,result=result,host_ms=elapsed,zero=zero,input_pixels_sha256=hashlib.sha256(pixels.tobytes()).hexdigest()))
  path=a.output/(selected['label']+'.png');assert cv2.imwrite(str(path),canvas)
  ep=[epi(cams,[np.array(v[key] if key=='native_points' else v['result']['points_raw']) for v in pair]) for key in ('native_points','mercury')]
  rows.append(dict(case=selected,display_label=display_label,views=pair,index_middle_epipolar_rms_degrees=[float(np.sqrt(np.mean(e[5:13]**2))) for e in ep],image_sha256=sha(path)));print(selected['label'],[round(v['result']['presence'],4) for v in pair],flush=True)
 report=dict(scope=__doc__,rows=rows,runtime=dict(python=platform.python_version(),machine=platform.machine(),system=platform.system(),onnxruntime=models.version,provider='CPUExecutionProvider',threads=1,opencv=cv2.__version__,numpy=np.__version__),models=MODEL_HASHES,source_sha256={n:sha(BASE.parent/'exploratory'/n) for n in ('mercury_keypoints.py','mercury_probe.py','frame_data.py')},runner_sha256=sha(Path(__file__)),crop_review_sha256=sha(a.crop_review/'result.json'),limitations=['Historical visible-fist artifact identifier means raised-finger-conflict, not a fist.','Ten preselected failure views; not a representative benchmark or physical pose ground truth.','Mercury stereographic128x128 sampling and native224x224 affine sampling differ. This compares complete observation paths, not only weights.','No temporal prior or solver; current crop/side hypotheses come from existing tracker.','Zero image controls skip inference on zero variance, so cannot prove real-background rejection.','No native Frame execution, timing, camera or publication. Separate model redistribution terms remain unresolved.'])
 (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
