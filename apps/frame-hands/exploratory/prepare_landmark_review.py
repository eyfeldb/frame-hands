"""Prepare raw-only exact-frame fingertip review sheets; no pose output is read."""
import argparse,hashlib,json
from pathlib import Path
import cv2,numpy as np
from frame_data import Recording,read_audit


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',type=Path,required=True);p.add_argument('--seconds',nargs='+',type=float,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);audit=read_audit(a.session/'frame/live-views.txt');recording=Recording(a.session/'frame/sets.bin');first=recording.index[0][0]
    targets={min(audit,key=lambda t:abs((t-first)/1e9-seconds)) for seconds in a.seconds};manifest=[]
    for seconds,stamp,images in recording.frames(min(a.seconds)-.1,max(a.seconds)+.1):
        if stamp not in targets:continue
        folder=a.output/f'{seconds:06.3f}';folder.mkdir();cards=[];entries=[]
        for name,(pixels,_) in images.items():
            raw_path=folder/(name+'.png');cv2.imwrite(str(raw_path),pixels)
            views=[v for v in audit[stamp]['views'] if v['camera']==name]
            for v in views:
                points=np.asarray(v['points']);lo=np.maximum(0,np.floor(points.min(0)-60).astype(int));hi=np.minimum([pixels.shape[1],pixels.shape[0]],np.ceil(points.max(0)+60).astype(int));crop=pixels[lo[1]:hi[1],lo[0]:hi[0]]
                scale=min(2.,440/max(crop.shape));scaled=cv2.resize(cv2.cvtColor(crop,cv2.COLOR_GRAY2BGR),None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST);card=np.zeros((500,500,3),np.uint8);card[50:50+scaled.shape[0],45:45+scaled.shape[1]]=scaled
                for x in range(int(np.ceil(lo[0]/20)*20),hi[0],20):
                    px=int(round(45+(x-lo[0])*scale));cv2.line(card,(px,44),(px,49),(220,220,220));cv2.putText(card,str(x),(px-8,39),cv2.FONT_HERSHEY_SIMPLEX,.3,(220,220,220),1)
                for y in range(int(np.ceil(lo[1]/20)*20),hi[1],20):
                    py=int(round(50+(y-lo[1])*scale));cv2.line(card,(39,py),(44,py),(220,220,220));cv2.putText(card,str(y),(5,py+3),cv2.FONT_HERSHEY_SIMPLEX,.3,(220,220,220),1)
                cv2.putText(card,f'{seconds:.3f}s {name} source {v["id"]}',(8,18),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
                path=folder/f'{name}-source{v["id"]}.png';cv2.imwrite(str(path),card);cards.append(card)
                entries.append(dict(camera=name,source_id=v['id'],crop_origin=lo.tolist(),crop_end=hi.tolist(),scale=scale,canvas_origin=[45,50],path=str(path.relative_to(a.output)),raw_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest()))
        if cards:
            if len(cards)%2:cards.append(np.zeros_like(cards[0]))
            sheet=np.vstack([np.hstack(cards[i:i+2]) for i in range(0,len(cards),2)]);cv2.imwrite(str(folder/'sheet.png'),sheet)
        manifest.append(dict(seconds=seconds,stamp=stamp,crops=entries))
    (a.output/'manifest.json').write_text(json.dumps(dict(scope=__doc__+' Detector boxes choose broad crops only; no landmark/skeleton/side overlay. Annotator must record uncertainty and omit occluded points.',session=str(a.session),requested_seconds=a.seconds,frames=manifest),indent=2)+'\n')
    print(json.dumps([dict(seconds=r['seconds'],stamp=r['stamp'],crops=len(r['crops'])) for r in manifest]))

if __name__=='__main__':main()
