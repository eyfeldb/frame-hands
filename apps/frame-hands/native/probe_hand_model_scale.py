"""Compare official Lite/Full hand models on byte-identical original native crops.
Saved-crop CPU diagnostic; neither optical truth nor a native benchmark.
"""
import argparse,json,sys,time,platform
from pathlib import Path
import numpy as np,cv2
BASE=Path(__file__).resolve().parent;sys.path[:0]=[str(BASE),str(BASE.parent/'exploratory')]
from run_tracker_rig_check import sha,stats
from review_native_landmarks import MP_SEGMENTS
from frame_data import load_calibration
from audit_crop_expansion import epi

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('run','runtime','session','upstream'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();sys.path.insert(0,str(a.runtime.resolve()));sys.path.insert(0,str(a.upstream));from ai_edge_litert.interpreter import Interpreter
 from importlib.metadata import version
 manifest=json.loads((a.run/'models.json').read_text());requests=json.loads((a.run/'requests.json').read_text())['requests'];native=np.loadtxt(a.run/'inputs/native.txt');assert native.shape==(len(requests),51)
 models={};schema={}
 for model in manifest:
  path=a.run/'models'/(model['name']+'.tflite');assert sha(path)==model['sha256'];m=Interpreter(model_path=str(path),num_threads=1);m.allocate_tensors();inp=m.get_input_details();out=m.get_output_details();assert len(inp)==1 and list(inp[0]['shape'])==[1,224,224,3] and inp[0]['dtype']==np.float32
  assert [(d['name'],d['shape'].tolist()) for d in out]==[('Identity',[1,63]),('Identity_1',[1,1]),('Identity_2',[1,1]),('Identity_3',[1,63])]
  models[model['name']]=(m,inp[0]['index'],{d['name']:d['index'] for d in out});schema[model['name']]=dict(inputs=[dict(name=d['name'],shape=d['shape'].tolist()) for d in inp],outputs=[dict(name=d['name'],shape=d['shape'].tolist()) for d in out])
 def predict(name,x):
  m,ind,outputs=models[name];start=time.perf_counter();m.set_tensor(ind,x);m.invoke();v={n:m.get_tensor(i).copy() for n,i in outputs.items()};elapsed=(time.perf_counter()-start)*1000
  assert all(np.isfinite(z).all() for z in v.values());return dict(screen=v['Identity'].reshape(21,3).tolist(),presence=float(v['Identity_1'][0,0]),right=float(v['Identity_2'][0,0]),world=v['Identity_3'].reshape(21,3).tolist()),elapsed
 records=[];timings={n:[] for n in models};outdir=a.run/'review';outdir.mkdir(exist_ok=False)
 for r,row in zip(requests,native):
  assert int(row[0])==r['index'];matrix=row[3:9].reshape(2,3);np.testing.assert_allclose(row[9:].reshape(21,2),r['native_points'],atol=1e-6,rtol=0);assert abs(row[1]-r['native_presence'])<1e-7 and abs(row[2]-r['native_right'])<1e-7
  pixels=np.fromfile(a.run/'inputs'/(str(r['index'])+'.gray'),dtype=np.uint8).reshape(224,224);x=np.repeat((pixels.astype(np.float32)*np.float32(1/255))[None,:,:,None],3,axis=3)
  results={}
  for name in models:
   if not records:
    for _ in range(5):predict(name,x)
   result,elapsed=predict(name,x);repeat,_=predict(name,x);assert result==repeat;timings[name].append(elapsed)
   pts=np.array(result['screen'])[:,:2];result['points_raw']=(np.c_[pts,np.ones(21)]@matrix.T).tolist();results[name]=result
  inv=cv2.invertAffineTransform(matrix);current=np.c_[np.array(r['native_points']),np.ones(21)]@inv.T;delta=np.linalg.norm(current-np.array(results['lite']['screen'])[:,:2],axis=1)
  records.append(dict(**r,matrix=matrix.tolist(),models=results,lite_vs_native_crop_px=dict(mean=float(delta.mean()),maximum=float(delta.max())),native_gray_sha256=sha(a.run/'inputs'/(str(r['index'])+'.gray'))))
 cases=[];cams=load_calibration(a.session/'frame/snapshot/device/persist')
 for label in dict.fromkeys(r['case'] for r in records):
  selected=[r for r in records if r['case']==label];canvas=np.full((750,1280,3),24,np.uint8);cv2.putText(canvas,label+' / exact native224 crop / I=index M=middle',(6,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(240,240,240),1,cv2.LINE_AA)
  for v,r in enumerate(selected):
   pixels=np.fromfile(a.run/'inputs'/(str(r['index'])+'.gray'),dtype=np.uint8).reshape(224,224);inv=cv2.invertAffineTransform(np.array(r['matrix']));native_pts=np.c_[np.array(r['native_points']),np.ones(21)]@inv.T;values=[native_pts,np.array(r['models']['lite']['screen'])[:,:2],np.array(r['models']['full']['screen'])[:,:2]]
   for col in range(4):
    pic=cv2.cvtColor(pixels,cv2.COLOR_GRAY2BGR)
    if col:
     pp=values[col-1]
     for i,j in MP_SEGMENTS:cv2.line(pic,tuple(np.rint(pp[i]).astype(int)),tuple(np.rint(pp[j]).astype(int)),(230,170,60) if col<3 else (60,230,70),1,cv2.LINE_AA)
     for finger,tip in enumerate((8,12,16,20)):cv2.putText(pic,('I','M','R','L')[finger],tuple(np.rint(pp[tip]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1,cv2.LINE_AA)
    title=('image',f"native P{r['native_presence']:.3f}",f"Lite P{r['models']['lite']['presence']:.3f}",f"Full P{r['models']['full']['presence']:.3f}")[col]
    canvas[85+330*v:385+330*v,320*col:320*col+300]=cv2.resize(pic,(300,300));cv2.putText(canvas,r['camera']+' '+title,(320*col+3,72+330*v),cv2.FONT_HERSHEY_SIMPLEX,.43,(240,240,240),1,cv2.LINE_AA)
  path=outdir/(label+'.png');assert cv2.imwrite(str(path),canvas);geometric=None
  if len(selected)==2:
   geometric={name:float(np.sqrt(np.mean(epi(cams,[np.array(r['models'][name]['points_raw']) for r in selected])[5:13]**2))) for name in models}
  cases.append(dict(case=label,views=len(selected),image_sha256=sha(path),index_middle_epipolar_rms_degrees=geometric))
 controls={name:predict(name,np.zeros((1,224,224,3),np.float32))[0] for name in models}
 report=dict(scope=__doc__,models=manifest,schema=schema,records=records,cases=cases,zero_controls=controls,host_invoke_and_copy_ms={n:stats(v) for n,v in timings.items()},native_geometry_reproduced=True,repeat_exact=True,lite_native_point_crop_px=stats([r['lite_vs_native_crop_px']['maximum'] for r in records]),runtime=dict(python=platform.python_version(),platform=platform.platform(),litert=version('ai-edge-litert'),numpy=np.__version__,opencv=cv2.__version__,threads=1,backend='CPU XNNPACK via compatibility Interpreter API'),provenance=dict(requests_sha256=sha(a.run/'requests.json'),native_output_sha256=sha(a.run/'inputs/native.txt'),exporter_sha256=sha(BASE/'export_native_hand_crops.cpp'),probe_sha256=sha(Path(__file__))),limitations=['Saved baseline-selected crops, including known failures; not independent tracking accuracy or coverage.','Per-view landmark points and handedness are hypotheses. Epipolar agreement is not anatomical truth.','Native ncnn outputs reproduced exactly; LiteRT uses the same byte crops with documented RGB0..1 input.','Host timings exclude camera/cropping/tracker/fitting/rendering, not Frame speed.','No temporal tracking, fusion, model conversion or runtime promotion. No accepted holdout inputs.'])
 (a.run/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('host_invoke_and_copy_ms','lite_native_point_crop_px','cases')},indent=2))
if __name__=='__main__':main()
