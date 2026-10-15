"""Exact raw-camera comparison for largest stereo-start proxy improvement/regression."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from frame_data import Recording
from probe_mercury_keypoints import draw


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('result','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);protocol=json.loads((a.result/'protocol.json').read_text());rec=Recording(Path(protocol['session'])/'frame/sets.bin');audit=json.loads((a.result/'audit.json').read_text());rows=[json.loads(line) for line in (a.result/'frames.jsonl').read_text().splitlines()];reports=[]
    for ci,event in enumerate([audit['largest_regressions'][0],audit['largest_improvements'][0]]):
        index=next(i for i,row in enumerate(rows) if row['stamp']==event['stamp']);panels=[];window=[]
        for offset in (-3,0,3):
            row=rows[index+offset];_,stamp,images=next(rec.frames(row['seconds']-1e-6,row['seconds']+1e-6));assert stamp==row['stamp'];fit=next((f for f in row['fits'] if f['side']==event['side'] and f['branch']==event['branch']),None)
            if fit is None:continue
            window.append(dict(seconds=row['seconds'],fit=fit))
            for name in ('slam_left','slam_right'):
                raw=images[name][0];uv=np.vstack([fit['projected'][name],fit['candidate_projected'][name]]);lo=np.maximum(uv.min(0)-45,0).astype(int);hi=np.minimum(uv.max(0)+45,[raw.shape[1],raw.shape[0]]).astype(int)
                columns=[cv2.cvtColor(raw,cv2.COLOR_GRAY2BGR),draw(raw,fit['projected'][name],(255,255,0)),draw(raw,fit['candidate_projected'][name],(0,255,0))]
                panel=cv2.hconcat([cv2.resize(image[lo[1]:hi[1],lo[0]:hi[0]],(384,384)) for image in columns]);cv2.putText(panel,name+' '+str(round(row['seconds'],4))+' raw/native/stereo',(5,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);panels.append(panel)
        cv2.imwrite(str(a.output/f'case-{ci}.jpg'),cv2.vconcat(panels));reports.append(dict(selection=event,window=window))
    (a.output/'report.json').write_text(json.dumps(dict(reports=reports,selection='Largest fallback-carry proxy regression and improvement, with neighboring frames. Same exact raw captures; diagnostic selection.'),indent=2)+'\n')


if __name__=='__main__':main()
