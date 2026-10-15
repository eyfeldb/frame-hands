"""Audit public chirality errors with source groups and exact paired crop panels."""
import argparse,collections,hashlib,json
from pathlib import Path
import cv2,numpy as np,torch
from tiny_chirality import load_classifier
from train_chirality import predict
from distill_encoder import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('public','run','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    manifest=json.loads((a.public/'manifest.json').read_text());training=json.loads((a.run/'manifest.json').read_text())
    if sha(a.public/'manifest.json')!=training['public_manifest_sha256']:raise ValueError('Public manifest changed')
    validation=set(training['public_validation_sequences']);records=[];images=[];seen=set()
    for row in manifest['records']:
        if row['sequence'] not in validation:continue
        path=a.public/row['directory']/'training.npz'
        if sha(path)!=row['archive_sha256']:raise ValueError('Public archive changed')
        with np.load(path,allow_pickle=False) as data:
            for frame,pair in enumerate(data['images']):
                for camera,image in enumerate(pair):
                    u8=np.rint(image*255).astype(np.uint8)
                    digest=hashlib.sha256(u8.tobytes()).hexdigest();mirror=hashlib.sha256(np.ascontiguousarray(u8[:,::-1]).tobytes()).hexdigest()
                    if digest in seen:continue
                    seen.update((digest,mirror));images.append(image[None]);records.append(dict(**row,frame=frame,camera=camera,source_hand_idx=int(np.asarray(data['hand_idx']).flat[0])))
    torch.set_num_threads(1);model=load_classifier(a.run/'best.torch',device='cpu')
    x=torch.from_numpy(np.stack(images));q=predict(model,x);qm=predict(model,x.flip(-1))
    eligible=(np.maximum(q,1-q)>=.98)&(np.maximum(qm,1-qm)>=.98)&(np.abs(q+qm-1)<=.05)
    # Normal source crops are canonical left. Mirrored source crops are right.
    wrong_normal=q>=.5;wrong_mirror=qm<.5
    groups={};errors=[];trace=[]
    for i,row in enumerate(records):
        group=groups.setdefault(row['group'],dict(normal_crops=0,eligible_normal_crops=0,wrong_eligible_normal=0,wrong_eligible_mirror=0,wrong_normal=0))
        group['normal_crops']+=1;group['eligible_normal_crops']+=int(eligible[i]);group['wrong_eligible_normal']+=int(eligible[i]&wrong_normal[i]);group['wrong_eligible_mirror']+=int(eligible[i]&wrong_mirror[i]);group['wrong_normal']+=int(wrong_normal[i])
        result=dict(**row,q_right=float(q[i]),mirrored_q_right=float(qm[i]),eligible=bool(eligible[i]),wrong_normal=bool(wrong_normal[i]))
        trace.append(result)
        if eligible[i] and wrong_normal[i]:errors.append((i,result))
    (a.output/'trace.json').write_text(json.dumps(trace,indent=2)+'\n')
    # At most one crop from each source record, so a consecutive frame burst does
    # not visually dominate the sheet. Ranking is confidence, not random choice.
    chosen=[];entries=set()
    for i,row in sorted(errors,key=lambda item:-item[1]['q_right']):
        key=(row['sequence'],row['entry'])
        if key in entries:continue
        entries.add(key);chosen.append((i,row))
        if len(chosen)>=12:break
    panels=[]
    for i,row in chosen:
        with np.load(a.public/row['directory']/'training.npz',allow_pickle=False) as data:pair=data['images'][row['frame']]
        panel=np.zeros((244,576,3),np.uint8)
        for col,(label,value) in enumerate((('target: LEFT',images[i][0]),('mirror: RIGHT',images[i][0][:,::-1]),('other camera',pair[1-row['camera']]))):
            tile=cv2.cvtColor(cv2.resize(np.rint(value*255).astype(np.uint8),(192,192)),cv2.COLOR_GRAY2BGR)
            panel[52:,col*192:(col+1)*192]=tile
            cv2.putText(panel,label,(col*192+5,44),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
        title=f"{row['group']} {row['sequence'].split('/')[-2]} e{row['entry']} f{row['frame']} c{row['camera']} q={q[i]:.4f}/{qm[i]:.4f}"
        cv2.putText(panel,title,(5,17),cv2.FONT_HERSHEY_SIMPLEX,.40,(255,255,255),1);panels.append(panel)
    if panels and not cv2.imwrite(str(a.output/'confident-errors.jpg'),np.vstack(panels)):raise ValueError('Image write failed')
    result=dict(scope='Offline selected public validation crop audit. No new training, Frame workload or confidence calibration.',checkpoint_sha256=sha(a.run/'best.torch'),normal_images=len(images),groups=groups,selected_error_panels=[row for _,row in chosen],source_sha256=sha(Path(__file__)))
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
