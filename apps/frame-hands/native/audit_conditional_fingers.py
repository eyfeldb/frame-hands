"""Audit conditional-prior failures without retraining or tuning parameters."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from probe_conditional_fingers import conditional, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--public',type=Path,required=True)
    args=parser.parse_args()
    run=args.run
    report=json.loads((run/'summary.json').read_text())
    manifest=json.loads((args.public/'manifest.json').read_text())
    if sha(args.public/'manifest.json')!=report['provenance']['public_manifest_sha256']:
        raise ValueError('Public manifest changed')
    records=[r for r in manifest['records'] if '/'.join((r['sequence'].split('/')[1],r['sequence'].split('/')[-2])) in report['split']['validation_groups']]
    data=np.load(run/'validation.npz',allow_pickle=False)
    prior=np.load(run/'conditional-prior.npz',allow_pickle=False)
    cases=json.loads((run/'cases.json').read_text())
    raw=[]
    for truth,finger in zip(data['truth'],data['finger']):
        known=np.ones(20,bool);known[4*finger:4*finger+4]=False
        raw.append(conditional(prior['mean'],prior['covariance'],truth,known)[0][~known])
    hidden=4*data['finger'][:,None]+np.arange(4)
    target=data['truth'][np.arange(len(raw))[:,None],hidden]
    covered=np.abs(np.array(raw)-target)<=1.96*data['std']
    groups={'all':np.ones(len(raw),bool),'independent_finger_exceptions':data['exception']}
    groups.update({name:data['finger']==i for i,name in enumerate(('thumb','index','middle','ring','pinky'))})
    groups.update({domain:np.array([records[c['record']]['sequence'].split('/')[1]==domain for c in cases]) for domain in ('real','synthetic')})
    counts=Counter(c['record'] for c in cases if c['exception'])
    audit=dict(nominal_95pct_per_angle_coverage={name:dict(cases=int(mask.sum()),coverage=float(covered[mask].mean())) for name,mask in groups.items()},exception_records=[dict(record=i,sequence=records[i]['sequence'],entry=records[i]['entry'],cases=n) for i,n in sorted(counts.items())],all_exception_cases_have_history=bool(np.all(data['frame'][data['exception']]>0)),source_sha256=sha(Path(__file__)),validation_sha256=sha(run/'validation.npz'),summary_sha256=sha(run/'summary.json'))
    # One middle exceptional frame per record; selection is deterministic, not best/worst.
    import cv2
    canvas=np.full((370*len(counts),1000,3),245,np.uint8)
    def text(pos,message,color='black'):
        colors={'black':(0,0,0),'green':(0,120,0),'gray':(100,100,100),'blue':(200,0,0),'red':(0,0,200)}
        cv2.putText(canvas,message,(pos[0],pos[1]+12),cv2.FONT_HERSHEY_SIMPLEX,.40,colors[color],1,cv2.LINE_AA)
    for row,record_index in enumerate(sorted(counts)):
        indices=[i for i,c in enumerate(cases) if c['exception'] and c['record']==record_index]
        index=indices[len(indices)//2];case=cases[index];record=records[record_index]
        archive=args.public/record['directory']/'training.npz'
        if sha(archive)!=record['archive_sha256']:raise ValueError('Changed archive')
        images=np.load(archive,allow_pickle=False)['images'][case['frame']]
        y=row*370
        text((10,y+8),record['sequence']+' entry '+str(record['entry'])+' frame '+str(case['frame'])+' | masked INDEX',color='black')
        for view in range(2):
            im=cv2.resize(np.uint8(np.clip(images[view],0,1)*255),(288,288))
            canvas[y+35:y+323,10+view*300:298+view*300]=im[:,:,None]
        text((620,y+35),'Index angles (degrees); solved label is target',color='black')
        for j,name in enumerate(('truth','hold_first','mean','conditional','copy_visible')):
            values=np.degrees(data[name][index,4:8]);color={'truth':'black','hold_first':'green','mean':'gray','conditional':'blue','copy_visible':'red'}[name]
            text((620,y+65+j*44),name+': '+', '.join(f'{v:.1f}' for v in values),color=color)
        text((10,y+335),'Existing public crops; artificial masking, not real occlusion. Adjacent cases are correlated.',color='black')
    if not cv2.imwrite(str(run/'exception-review.png'),canvas):raise RuntimeError('Image write failed')
    audit['review_sha256']=sha(run/'exception-review.png')
    (run/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
