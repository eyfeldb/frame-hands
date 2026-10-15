"""Review the largest crop-switch discontinuity for each side against raw images."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from frame_data import Recording
from probe_mercury_keypoints import draw


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--result',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    audit=json.loads((a.result/'audit.json').read_text());protocol=json.loads((a.result/'protocol.json').read_text());parent=json.loads((Path(protocol['source'])/'protocol.json').read_text());origin=json.loads((Path(parent['source'])/'protocol.json').read_text());rec=Recording(Path(origin['session'])/'frame/sets.bin')
    with (a.result/'frames.jsonl').open() as f:rows=[json.loads(line) for line in f]
    cases=[]
    for side in (0,1):
        event=next(e for e in audit['largest_switch_steps'] if e['side']==side and e['branch']=='fallback_reset');index=next(i for i,r in enumerate(rows) if r['seconds']==event['seconds']);pair=rows[index-1:index+1];report=dict(event=event,frames=[]);panels=[]
        for row in pair:
            _,stamp,images=next(rec.frames(row['seconds']-.000001,row['seconds']+.000001));assert stamp==row['stamp'];frame=dict(seconds=row['seconds'],stamp=stamp,fits={})
            for name in ('slam_left','slam_right'):
                image=images[name][0];columns=[cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)]
                for branch in ('box','fallback_carry','fallback_reset'):
                    fit=next((f for f in row['fits'] if f['side']==side and f['branch']==branch),None);canvas=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
                    if fit is not None:
                        canvas=draw(canvas,fit['projected'][name],(255,255,0) if side==0 else (0,140,255));frame['fits'][branch]=fit
                    columns.append(canvas)
                sheet=cv2.hconcat([cv2.resize(c,(384,round(384*image.shape[0]/image.shape[1]))) for c in columns]);cv2.putText(sheet,name+' '+str(round(row['seconds'],4))+' raw/box/carry/reset',(5,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);panels.append(sheet)
            report['frames'].append(frame)
        deltas={}
        for branch in ('box','fallback_carry','fallback_reset'):
            before=report['frames'][0]['fits'].get(branch);after=report['frames'][1]['fits'].get(branch)
            if before is None or after is None:continue
            d=np.linalg.norm(np.asarray(after['world_joints_mm'])-np.asarray(before['world_joints_mm']),axis=1)
            deltas[branch]=dict(wrist_step_mm=float(d[0]),max_joint_step_mm=float(d.max()),max_joint_index=int(d.argmax()),image_joint_step_median_px={name:float(np.median(np.linalg.norm(np.asarray(after['projected'][name])-np.asarray(before['projected'][name]),axis=1))) for name in ('slam_left','slam_right')},residual_before=before['fit']['reprojection_error'],residual_after=after['fit']['reprojection_error'])
        report['deltas']=deltas;cases.append(report);cv2.imwrite(str(a.output/('side-'+str(side)+'.jpg')),cv2.vconcat(panels));print(side,json.dumps(deltas))
    (a.output/'report.json').write_text(json.dumps(dict(selection='Largest reset-switch max-joint step per side from declared replay audit; diagnostic selection, not a held-out score.',cases=cases,legend='Time-grouped camera rows; columns raw / original / fallback carry / fallback reset. Only the selected anatomical side is shown.'),indent=2)+'\n')


if __name__=='__main__':main()
