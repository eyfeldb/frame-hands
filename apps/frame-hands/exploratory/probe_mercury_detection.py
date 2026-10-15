"""Sparse saved-image Mercury detector comparison; no device/runtime access."""
import argparse,hashlib,json,sys,time,shutil
from pathlib import Path
import cv2
import numpy as np
from frame_data import Recording
from mercury_probe import MercuryModels,MODEL_HASHES

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['protocol','models','runtime','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.runtime.resolve()));a.output.mkdir(parents=True,exist_ok=False)
    protocol=json.loads(a.protocol.read_text());shutil.copy2(a.protocol,a.output/'protocol.json')
    model=MercuryModels(a.models);schema=model.keypoint_schema_smoke();records={};reports=[]
    cv2.setNumThreads(1)
    for case in protocol['cases']:
        session=Path(case['session'])
        if str(session) not in records:records[str(session)]=Recording(session/'frame/sets.bin')
        rec=records[str(session)];nearest=min(rec.index,key=lambda r:abs((r[0]-rec.index[0][0])/1e9-case['seconds']))
        seconds=(nearest[0]-rec.index[0][0])/1e9
        if abs(seconds-case['seconds'])>.04:raise ValueError('No matching capture')
        _,stamp,images=next(rec.frames(seconds-.000001,seconds+.000001));panels=[];rows=[]
        for name,rotation in [('slam_left',90),('slam_right',270)]:
            image=images[name][0]
            if not reports:
                for _ in range(3):model.detect(image,rotation)
            start=time.perf_counter();detections,small,debug=model.detect(image,rotation);elapsed=(time.perf_counter()-start)*1000
            display=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
            for d in detections:
                color=(255,230,0) if d['model_side']==0 else (0,140,255)
                if d['found'] and d['size_px']>0:
                    x,y=d['center_px'];size=d['size_px']/2
                    cv2.rectangle(display,(round(x-size),round(y-size)),(round(x+size),round(y+size)),color,3)
                    cv2.putText(display,f"slot{d['model_side']} p={d['presence']:.3f}",(round(x-size),round(y-size)-8),cv2.FONT_HERSHEY_SIMPLEX,.75,color,2)
            panel=cv2.resize(display,(528,512));cv2.putText(panel,name+' '+case['name'],(8,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
            panels.append(panel);cv2.imwrite(str(a.output/(case['name']+'-'+name+'-input.png')),small)
            rows.append(dict(camera=name,rotation=rotation,image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),detections=detections,detector_ms=elapsed,debug=debug))
        cv2.imwrite(str(a.output/(case['name']+'-boxes.jpg')),cv2.hconcat(panels))
        reports.append(dict(case=case,seconds=seconds,capture_raw_ns=stamp,views=rows))
    blank=model.detect(np.zeros((1024,1056),np.uint8),90)
    summary=dict(scope=protocol['scope'],runtime=model.version,provider='CPUExecutionProvider',threads=1,opencv=cv2.__version__,numpy=np.__version__,models=MODEL_HASHES,keypoint_schema=schema,blank=dict(detections=blank[0],debug=blank[2]),reports=reports,source_sha256={f:hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest() for f in ['mercury_probe.py','probe_mercury_detection.py']})
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(cases=len(reports),camera_views=sum(len(r['views']) for r in reports),detected_slots=sum(d['found'] for r in reports for v in r['views'] for d in v['detections']),output=str(a.output))))
if __name__=='__main__':main()
