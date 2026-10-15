"""Exact raw comparisons at the largest unconditioned continuing change per window."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from frame_data import Recording
from probe_mercury_keypoints import draw


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('result','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);summary=json.loads((a.result/'summary.json').read_text());audit=json.loads((a.result/'audit.json').read_text());rows=summary['reports'];reports=[]
    for ci,event in enumerate(audit['selected_visual_review']):
        index=next(i for i,row in enumerate(rows) if row['case']['name']==event['case']['name']);rec=Recording(Path(event['case']['session'])/'frame/sets.bin');panels=[];window=[]
        for offset in (-3,0,3):
            position=index+offset
            if not 0<=position<len(rows):continue
            row=rows[position]
            if row['case']['name'].rsplit('-',1)[0]!=event['group'] or 'skipped' in row:continue
            arms=[row['arms']['off-compensated'+('' if mode=='original' else '-'+mode)] for mode in ('original','curl','both')]
            if any('fit' not in arm for arm in arms):continue
            _,stamp,images=next(rec.frames(row['seconds']-1e-6,row['seconds']+1e-6));assert stamp==row['stamp'];window.append(dict(case=row['case'],arms=arms))
            for camera in ('slam_left','slam_right'):
                raw=images[camera][0];uv=np.vstack([arm['projected'][camera] for arm in arms]);lo=np.maximum(uv.min(0)-40,0).astype(int);hi=np.minimum(uv.max(0)+40,[raw.shape[1],raw.shape[0]]).astype(int)
                columns=[cv2.cvtColor(raw,cv2.COLOR_GRAY2BGR)]+[draw(raw,arm['projected'][camera],color) for arm,color in zip(arms,[(255,255,0),(0,140,255),(0,255,0)])]
                panel=cv2.hconcat([cv2.resize(image[lo[1]:hi[1],lo[0]:hi[0]],(320,320)) for image in columns]);cv2.putText(panel,camera+' '+str(round(row['seconds'],4))+' raw/native/curl/both',(5,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);panels.append(panel)
        cv2.imwrite(str(a.output/f'case-{ci}.jpg'),cv2.vconcat(panels));reports.append(dict(event=event,window=window))
    (a.output/'report.json').write_text(json.dumps(dict(reports=reports),indent=2)+'\n')


if __name__=='__main__':main()
