"""Counterfactual duplicate-gate ablation on frozen fresh-crop proposals."""
import argparse,json,hashlib,sys
from pathlib import Path
import cv2,numpy as np
from frame_data import Recording,read_audit
from probe_overlap_acquisition import same_hand


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('session','probe','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);rows=json.loads((a.probe/'rows.json').read_text());rec=Recording(a.session/'frame/sets.bin');audit=read_audit(a.session/'frame/live-views.txt');result=[]
    for row in rows:
        observed=[v['points'] for v in audit[row['stamp']]['views'] if v['camera']==row['camera']] if row['case']!='zero' else []
        arms={}
        for mode in ('original','late','joint_only','wrist_guard'):
            kept=list(observed);added=[]
            for index,v in sorted(enumerate(row['proposals']),key=lambda x:-x[1]['presence']):
                if v['presence']<.5 or (mode=='original' and v['would_skip_near']):continue
                xy=np.array(v['points']);duplicates=[same_hand(o,xy) for o in kept]
                if any(d['mean_close'] or (mode not in ('joint_only','wrist_guard') and d['palm_close']) or (mode=='wrist_guard' and d['palm_close'] and d['wrist_close']) for d in duplicates):continue
                kept.append(xy);added.append(index)
            arms[mode]=added
        result.append(dict(case=row['case'],camera=row['camera'],seconds=row['seconds'],arms=arms))
        _,stamp,images=next(rec.frames(row['seconds']-1e-6,row['seconds']+1e-6));assert stamp==row['stamp'];panels=[]
        for mode,indices in arms.items():
            pixels=images[row['camera']][0].copy()
            if row['case']=='zero':pixels.fill(0)
            img=cv2.cvtColor(pixels,cv2.COLOR_GRAY2BGR)
            for index in indices:
                xy=np.rint(row['proposals'][index]['points']).astype(int)
                for start in (1,5,9,13,17):
                    for x,y in zip([0,start,start+1,start+2],[start,start+1,start+2,start+3]):cv2.line(img,tuple(xy[x]),tuple(xy[y]),(0,255,255),2)
                cv2.putText(img,str(index),tuple(xy[0]),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,0,255),2)
            cv2.putText(img,f'{mode}: {len(indices)} added / {len(observed)} saved views',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
            panels.append(cv2.resize(img,(528,512)))
        assert cv2.imwrite(str(a.output/f'{row["case"]}-{row["camera"]}.jpg'),cv2.hconcat(panels))
        print(json.dumps(dict(case=row['case'],camera=row['camera'],added={k:len(v) for k,v in arms.items()})),flush=True)
    out=dict(rows=result,scope='Frozen all-tiles proposals and same-frame saved detector anchors. Not native scheduler replay or candidate temporal rollout; added count is not accuracy. joint_only removes the palm-center-only duplicate criterion but retains mean landmark distance. wrist_guard additionally requires a close wrist for the palm-only criterion, reusing its0.5size threshold. No threshold fitting or deployment.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),probe_sha256=hashlib.sha256((a.probe/'rows.json').read_bytes()).hexdigest())
    (a.output/'audit.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
