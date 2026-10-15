"""Exhaustive offline acquisition diagnostic, not native scheduler or live parity.

Uses frozen model weights and source formulas. OpenCV affine sampling and x64
FP32 differ from native ncnn ARM FP16. Results cannot reproduce live counters.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
import cv2,numpy as np
from frame_data import Recording,read_audit,load_calibration,unproject
from probe_roi_presence import PresenceModel


def affine(center,size,rotation,n):
    c,s=np.cos(rotation),np.sin(rotation);r=np.array([[c,-s],[s,c]])
    linear=(r*size/n).astype(np.float32)
    translation=np.asarray(center)-(linear.sum(axis=1,dtype=np.float32)*np.float32(n)).astype(float)/2
    return np.column_stack((linear,translation)).astype(np.float32)


def hand_matrix(points):
    # Full native roi_from_points subset, including observed thumb CMC.
    points=np.asarray(points,float);wrist=points[0];middle=((points[5]+points[13])*.5+points[9])*.5
    delta=middle-wrist;rotation=np.pi/2-np.arctan2(-delta[1],delta[0])
    c,s=np.cos(rotation),np.sin(rotation);r=np.array([[c,-s],[s,c]])
    bounds=points[[0,1,2,3,5,6,9,10,13,14,17,18]];center=(bounds.min(0)+bounds.max(0))*.5
    local=(bounds-center)@r;lo,hi=local.min(0),local.max(0);extent=hi-lo
    center+=r@((lo+hi)*.5+np.array([0.,-.1*extent[1]]));size=float(extent.max()*2)
    return affine(center,size,rotation,224),size


def native_clahe(img):
    # Frozen nets.cpp's 4x4 CLAHE, including its residual-bin distribution.
    n=img.shape[0];ts=n//4;area=ts*ts;clip=max(1,int(2*area/256));lut=np.empty((4,4,256),np.uint8)
    for y in range(4):
        for x in range(4):
            hist=np.bincount(img[y*ts:(y+1)*ts,x*ts:(x+1)*ts].ravel(),minlength=256)
            excess=int(np.maximum(hist-clip,0).sum());hist=np.minimum(hist,clip)+excess//256;hist[:excess%256]+=1
            lut[y,x]=np.minimum(255,(hist.cumsum().astype(np.float32)*np.float32(255/area)+np.float32(.5)).astype(int))
    pos=(np.arange(n,dtype=np.float32)+np.float32(.5))/ts-np.float32(.5)
    lo=np.clip(np.floor(pos).astype(int),0,3);hi=np.minimum(lo+1,3);w=np.clip(pos-lo,0,1).astype(np.float32)
    y0,y1=lo[:,None],hi[:,None];x0,x1=lo[None,:],hi[None,:];wx,wy=w[None,:],w[:,None]
    top=lut[y0,x0,img]*(1-wx)+lut[y0,x1,img]*wx;bottom=lut[y1,x0,img]*(1-wx)+lut[y1,x1,img]*wx
    return (top*(1-wy)+bottom*wy+np.float32(.5)).astype(np.uint8)


def tile_rois(camera):
    base=[];w,h=camera.width,camera.height
    for frac,gx,gy in ((.45,3,3),(.65,2,2),(1.,1,1)):
        size=frac*max(w,h)
        for i in range(gx):
            for j in range(gy):
                center=np.array([size/2+(w-size)*i/(gx-1) if gx>1 else w/2,size/2+(h-size)*j/(gy-1) if gy>1 else h/2])
                xf=camera.camera_to_world_xf;ray=unproject(camera,center)@xf[:3,:3].T;p=xf[:3,3]+ray*450
                d=p-np.array([170 if p[0]>0 else -170,-250,80]);d/=np.linalg.norm(d)
                uv=camera.eye_to_window(camera.world_to_eye(np.stack([p,p+d*50])));delta=uv[1]-uv[0]
                base.append((center,size,float(np.arctan2(delta[0],-delta[1]))))
    return base+[(c,s,r+a) for c,s,r in base for a in (-np.pi/2,-np.pi/4,np.pi/4,np.pi/2,np.pi)]


class PalmModel:
    def __init__(self,models):
        import ncnn
        self.ncnn=ncnn;self.net=ncnn.Net();self.net.opt.num_threads=1;self.net.opt.use_vulkan_compute=False
        self.net.opt.use_fp16_packed=False;self.net.opt.use_fp16_storage=False;self.net.opt.use_fp16_arithmetic=False
        for kind,loader in [('param',self.net.load_param),('bin',self.net.load_model)]:
            if loader(str(models/f'palm.ncnn.{kind}')):raise ValueError('Palm model load failed')
        self.anchors=np.array([[(x+.5)*stride,(y+.5)*stride] for stride,per in ((8,2),(16,6)) for y in range(192//stride) for x in range(192//stride) for _ in range(per)])

    def infer(self,image,roi):
        center,size,rotation=roi;matrix=affine(center,size,rotation,192)
        pixels=cv2.warpAffine(image,matrix,(192,192),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
        pixels=native_clahe(pixels);n=self.ncnn;mat=n.Mat.from_pixels(pixels,n.Mat.PixelType.PIXEL_GRAY2RGB,192,192);mat.substract_mean_normalize([],[1/255]*3)
        ex=self.net.create_extractor();assert ex.input('in0',mat)==0
        outputs=[]
        for name in ('out0','out1'):
            status,result=ex.extract(name);assert status==0;outputs.append(np.array(result).copy())
        raw=outputs[0].reshape(-1,18);logit=outputs[1].ravel();assert len(raw)==len(logit)==len(self.anchors)
        indices=np.flatnonzero(logit>np.float32(np.log(.2/.8)));indices=indices[np.argsort(-logit[indices],kind='stable')]
        scores=1/(1+np.exp(-np.clip(logit[indices].astype(float),-100,100)));boxes=raw[indices].astype(float);boxes[:,:2]+=self.anchors[indices]
        boxes[:,4:]+=np.tile(self.anchors[indices],(1,7));used=set();out=[]
        for i in range(len(indices)):
            if i in used:continue
            members=[]
            for j in range(i,len(indices)):
                if j in used:continue
                inter=np.maximum(0,np.minimum(boxes[i,:2]+boxes[i,2:4]/2,boxes[j,:2]+boxes[j,2:4]/2)-np.maximum(boxes[i,:2]-boxes[i,2:4]/2,boxes[j,:2]-boxes[j,2:4]/2)).prod()
                union=boxes[i,2:4].prod()+boxes[j,2:4].prod()-inter
                if j!=i and inter/(union+1e-9)<=.3:continue
                members.append(j);used.add(j)
            b=np.average(boxes[members],axis=0,weights=scores[members]);kp=np.column_stack((b[4:].reshape(7,2),np.ones(7)))@matrix.T
            c=np.r_[b[:2],1]@matrix.T;sz=b[2:4]*size/192;delta=kp[2]-kp[0];rot=np.pi/2-np.arctan2(-delta[1],delta[0]);shift=np.array([.5*sz[1]*np.sin(rot),-.5*sz[1]*np.cos(rot)])
            out.append(dict(score=float(scores[i]),members=len(members),roi=[(c+shift).tolist(),float(max(sz)*2.6),float(rot)]))
        return out,len(indices)


def same_hand(a,b):
    size=float(np.ptp(b,axis=0).max());dist=np.linalg.norm(a-b,axis=1)
    return dict(wrist_close=bool(dist[0]<.5*size),wrist_fraction=float(dist[0]/size),palm_close=bool(dist[9]<.5*size),mean_close=bool(dist.mean()<.25*size),palm_fraction=float(dist[9]/size),mean_fraction=float(dist.mean()/size))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('session','review','upstream','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.upstream));a.output.mkdir(parents=True,exist_ok=False)
    snap=a.session/'frame/snapshot';models=snap/'models/ncnn';manifest=json.loads((a.session/'snapshot/roi-source-build.json').read_text())
    for name,digest in manifest['files'].items():
        if name.startswith('upstream/track/'):
            path=snap/'source'/name.removeprefix('upstream/');assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    for name,digest in manifest['models'].items():assert hashlib.sha256((snap/'models/ncnn'/Path(name).name).read_bytes()).hexdigest()==digest
    review=json.loads((a.review/'report.json').read_text());selected=[dict(gap=g['index'],**next(s for s in g['selected'] if s['label']=='during')) for g in review['reports'] if g['index']>=2]
    selected += [dict(gap='separated',**review['reports'][0]['selected'][0]),dict(gap='reacquired',**review['reports'][5]['selected'][-1])]
    protocol=dict(selection=selected,scope=__doc__,tiles_per_camera=84,threshold=.2,hand_presence=.5,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),models=manifest['models'],source_verified=True,control='All-zero images through same84tiles per camera; selected inputs frozen before inference.')
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    cameras=load_calibration(a.session/'calibration');audit=read_audit(a.session/'frame/live-views.txt');rec=Recording(a.session/'frame/sets.bin');palm=PalmModel(models);hand=PresenceModel(models);rows=[]
    for selection in selected+[dict(gap='zero',source_seconds=selected[0]['source_seconds'],capture_raw_ns=selected[0]['capture_raw_ns'])]:
        seconds=selection['source_seconds'];_,stamp,images=next(rec.frames(seconds-1e-6,seconds+1e-6));assert stamp==selection['capture_raw_ns']
        for name,camera in cameras.items():
            started=time.perf_counter();image=images[name][0].copy()
            if selection['gap']=='zero':image.fill(0)
            observed=[v for v in audit[stamp]['views'] if v['camera']==name] if selection['gap']!='zero' else []
            existing=[hand_matrix(v['points'])[0][:,2]+hand_matrix(v['points'])[0][:,:2]@np.array([112,112]) for v in observed]
            proposals=[];above=0
            for ti,tile in enumerate(tile_rois(camera)):
                found,count=palm.infer(image,tile);above+=count
                for proposal in found:
                    center,size,rotation=proposal['roi'];matrix=affine(center,size,rotation,224)
                    pixels=cv2.warpAffine(image,matrix,(224,224),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP);points,presence,right=hand.infer(pixels)
                    xy=np.column_stack((points,np.ones(21)))@matrix.T
                    proposals.append(dict(tile=ti,**proposal,presence=presence,right=right,points=xy.tolist(),would_skip_near=any(np.linalg.norm(np.asarray(center)-c)<.5*size for c in existing),duplicates=[same_hand(v['points'],xy) for v in observed]))
            accepted=[v for v in proposals if v['presence']>=.5];kept=[]
            for v in sorted(accepted,key=lambda x:-x['presence']):
                if not any((lambda d:d['palm_close'] or d['mean_close'])(same_hand(np.array(o['points']),np.array(v['points']))) for o in kept):kept.append(v)
            canvas=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
            for idx,v in enumerate(kept):
                pts=np.rint(v['points']).astype(int);color=((37*idx+80)%256,(133*idx+100)%256,255)
                for start in (1,5,9,13,17):
                    for x,y in zip([0,start,start+1,start+2],[start,start+1,start+2,start+3]):cv2.line(canvas,tuple(pts[x]),tuple(pts[y]),color,2)
                cv2.putText(canvas,str(idx),tuple(pts[0]),cv2.FONT_HERSHEY_SIMPLEX,.7,color,2)
            assert cv2.imwrite(str(a.output/f'{selection["gap"]}-{name}.jpg'),canvas)
            row=dict(case=selection['gap'],camera=name,seconds=seconds,stamp=stamp,anchors_above_threshold=above,proposals=proposals,unique_native_gate=kept,elapsed_s=time.perf_counter()-started);rows.append(row)
            print(json.dumps(dict(case=row['case'],camera=name,anchors=above,proposals=len(proposals),valid=len(accepted),unique=len(kept))),flush=True)
            (a.output/'rows.json').write_text(json.dumps(rows)+'\n')
    (a.output/'summary.json').write_text(json.dumps(dict(protocol=protocol,rows=len(rows),ncnn=hand.version),indent=2)+'\n')
if __name__=='__main__':main()
