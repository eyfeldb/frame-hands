"""Exact-frame visual review of constant-stability agreement changes."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from frame_data import Recording
from probe_mercury_keypoints import draw


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--result',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((a.result/'protocol.json').read_text());parent=json.loads((Path(protocol['source'])/'protocol.json').read_text());origin=json.loads((Path(parent['source'])/'protocol.json').read_text());rec=Recording(Path(origin['session'])/'frame/sets.bin')
    audit=json.loads((a.result/'audit.json').read_text());selected=[]
    for group in ('long_prefix_side0_box','long_evaluation_side0_box'):
        selected.append(next(r for r in audit['largest_image_agreement_regressions'] if r['group']==group))
    selected.append(next(r for r in audit['largest_image_agreement_improvements'] if r['group'].startswith('long_')))
    with (a.result/'frames.jsonl').open() as f:rows=[json.loads(line) for line in f]
    reports=[]
    for ci,event in enumerate(selected):
        index=next(i for i,r in enumerate(rows) if r['stamp']==event['stamp']);branches=('box','box_constant') if event['group'].endswith('_box') else ('fallback_carry','fallback_constant');sheets=[];window=[]
        for offset in (-3,0,3):
            row=rows[index+offset];_,stamp,images=next(rec.frames(row['seconds']-.000001,row['seconds']+.000001));assert stamp==row['stamp'];fits={b:next((f for f in row['fits'] if f['side']==event['side'] and f['branch']==b),None) for b in branches};window.append(dict(seconds=row['seconds'],fits=fits))
            for name in ('slam_left','slam_right'):
                image=images[name][0];uv=np.vstack([f['projected'][name] for f in fits.values() if f is not None]);lo=np.maximum(uv.min(0)-45,0).astype(int);hi=np.minimum(uv.max(0)+45,[image.shape[1],image.shape[0]]).astype(int)
                columns=[cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)]
                for f in fits.values():columns.append(draw(image,f['projected'][name],(255,255,0) if event['side']==0 else (0,140,255)) if f else cv2.cvtColor(image,cv2.COLOR_GRAY2BGR))
                panel=cv2.hconcat([cv2.resize(c[lo[1]:hi[1],lo[0]:hi[0]],(384,384)) for c in columns]);cv2.putText(panel,name+' '+str(round(row['seconds'],4))+' raw/native/constant',(5,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);sheets.append(panel)
        cv2.imwrite(str(a.output/('case-'+str(ci)+'.jpg')),cv2.vconcat(sheets));reports.append(dict(event=event,window=window))
    (a.output/'report.json').write_text(json.dumps(dict(selection='Worst saved image-agreement regression in each latest left-hand phase, plus largest reported long-sequence improvement. Diagnostic selection, not independent validation.',reports=reports),indent=2)+'\n')
    print(json.dumps(selected,indent=2))


if __name__=='__main__':main()
