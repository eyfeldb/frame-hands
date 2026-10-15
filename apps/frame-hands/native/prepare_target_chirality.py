"""Prepare target-hand landmark conditioning without reading the latest local holdout."""
import argparse,hashlib,io,json,sys
from pathlib import Path
import cv2,numpy as np,torch
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE.parent/'exploratory'))
from train_chirality import local_images
from frame_data import load_calibration,read_audit,unproject
from replay_mesh_trial import packets
from replay_umetrack import COMMON
from distill_encoder import sha


def local_points(session,keys,upstream):
    sys.path.insert(0,str(upstream))
    cameras=load_calibration(session/'calibration');points={}
    for meta,_ in packets(session/'transport.bin'):
        if meta.get('waiting'):continue
        views={(v['camera'],v['id']):v for v in next(iter(read_audit(io.StringIO(meta['audit'])).values()))['views']}
        for crop in meta['crops']:
            key=(crop['camera'],crop['id']);v=views[key];cam=cameras[crop['camera']]
            xf=np.asarray(crop['camera_to_head_mm']);K=np.asarray(crop['intrinsics'])
            if not np.allclose(xf[:3,3],cam.camera_to_world_xf[:3,3],atol=.0001):raise ValueError('Crop/source camera center differs')
            rays=unproject(cam,np.asarray(v['points'])[[j for _,j in COMMON]])@cam.camera_to_world_xf[:3,:3].T@xf[:3,:3]
            if np.any(rays[:,2]<=0):raise ValueError('Target behind crop')
            uv=rays@K.T;uv=uv[:,:2]/uv[:,2:]
            points[(meta['stamp'],*key)]=uv
    return np.asarray([points[k['stamp'],k['camera'],k['source_id']] for k in keys],np.float32)


def public_points(root,row,upstream):
    import msgpack
    sys.path.insert(0,str(upstream))
    from lib.common.hand import HandModel
    from lib.common.hand_skinning import skin_landmarks
    record=root/'records'/Path(row['directory']).name
    manifest=json.loads((record/'manifest.json').read_text());raw=(record/'labels.record').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=manifest['files']['labels']['range_sha256']:raise ValueError('Source labels changed')
    labels=msgpack.unpackb(raw,raw=False,strict_map_key=True,max_array_len=1000000,max_map_len=128,max_str_len=4096,max_bin_len=8000000)
    # The source hand model is in its original physical side and millimeters.
    # Saved crop extrinsics already include canonical-right-image reflection.
    hand=HandModel(**{k:torch.tensor(v) if isinstance(v,list) else v for k,v in labels['hand_model'].items()})
    angles=torch.tensor(labels['joint_angles'],dtype=torch.float32);wrist=torch.tensor(labels['wrist'],dtype=torch.float32)
    hand=hand._replace(**{k:v.unsqueeze(0).expand(len(angles),*v.shape) for k,v in hand._asdict().items() if isinstance(v,torch.Tensor)})
    with torch.inference_mode():xyz=skin_landmarks(hand,angles,wrist).numpy()*.001
    xyz=xyz[:,[i for i,_ in COMMON]]
    with np.load(root/row['directory']/'training.npz',allow_pickle=False) as a:
        xf=a['extrinsics'];K=a['intrinsics']
        eye=np.einsum('fvij,fpj->fvpi',xf[:,:,:3,:3],xyz)+xf[:,:,:3,3][:,:,None,:]
        uv=np.einsum('fvij,fvpj->fvpi',K,eye)
        if np.any(uv[...,2]<=0):raise ValueError('Public target behind crop')
        return (uv[...,:2]/uv[...,2:]).astype(np.float32)


def panel(image,points,title):
    im=cv2.cvtColor(cv2.resize(np.rint(image*255).astype(np.uint8),(288,288)),cv2.COLOR_GRAY2BGR)
    for i,p in enumerate(points):cv2.circle(im,tuple(np.rint(p*3).astype(int)),3,(0,255,255) if i else (255,0,255),-1)
    cv2.putText(im,title,(5,18),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
    return im


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('train','train-evidence','development','development-evidence','public','upstream','reference-run','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False);torch.set_num_threads(1)
    if any('143912Z' in str(v) for v in (a.train,a.development)):raise ValueError('Latest holdout excluded')
    panes=[];datasets={};local_hashes=set()
    for name,session,evidence in [('train',a.train,a.train_evidence),('development',a.development,a.development_evidence)]:
        x,y,strong,keys,hashes=local_images(session,evidence);uv=local_points(session,keys,a.upstream)
        if hashes&local_hashes:raise ValueError('Local split duplicate')
        local_hashes|=hashes
        np.savez_compressed(a.output/(name+'.npz'),images=x,labels=y,strong=strong,points=uv)
        datasets[name]=dict(images=len(x),strong=int(strong.sum()),archive_sha256=sha(a.output/(name+'.npz')),transport_sha256=sha(session/'transport.bin'),evidence_sha256=sha(evidence))
        for i in np.linspace(0,len(x)-1,3,dtype=int):panes.append(panel(x[i,0],uv[i],f'{name} {i}'))
    original=json.loads((a.reference_run/'manifest.json').read_text());manifest=json.loads((a.public/'manifest.json').read_text())
    if sha(a.public/'manifest.json')!=original['public_manifest_sha256']:raise ValueError('Changed public subset')
    validation=set(original['public_validation_sequences']);splits={k:dict(images=[],points=[],groups=[]) for k in ('public_train','public_validation')};seen={k:set() for k in splits}
    for row in manifest['records']:
        name='public_validation' if row['sequence'] in validation else 'public_train'
        path=a.public/row['directory']/'training.npz'
        if sha(path)!=row['archive_sha256']:raise ValueError('Changed public archive')
        uv=public_points(a.public,row,a.upstream).reshape(-1,20,2)
        with np.load(path,allow_pickle=False) as archive:
            for image,points in zip(archive['images'].reshape(-1,96,96),uv):
                u8=np.rint(image*255).astype(np.uint8);digest=hashlib.sha256(u8.tobytes()).hexdigest();mirror=hashlib.sha256(np.ascontiguousarray(u8[:,::-1]).tobytes()).hexdigest()
                if digest in local_hashes or mirror in local_hashes:raise ValueError('Public/local duplicate')
                if digest in seen[name]:continue
                seen[name].update((digest,mirror));splits[name]['images'].append(image[None]);splits[name]['points'].append(points);splits[name]['groups'].append(row['group'])
        # Include the previously reviewed confident contact errors and both sides.
        if (row['group'],row['entry'],row['sequence'].split('/')[-2]) in [('real/hand_hand',22,'00023'),('real/hand_hand',37,'00046'),('synthetic/separate_hand',229,'00026')]:
            with np.load(path,allow_pickle=False) as archive:
                for index in (0,17):panes.append(panel(archive['images'].reshape(-1,96,96)[index],uv[index],f"{row['group']} {row['entry']} {index}"))
    if seen['public_train']&seen['public_validation']:raise ValueError('Public split duplicate')
    for name,data in splits.items():
        np.savez_compressed(a.output/(name+'.npz'),**{k:np.asarray(v) for k,v in data.items()})
        datasets[name]=dict(images=len(data['images']),archive_sha256=sha(a.output/(name+'.npz')))
    cv2.imwrite(str(a.output/'target-review.jpg'),np.vstack([np.hstack(panes[i:i+3]) for i in range(0,len(panes),3)]))
    result=dict(scope=__doc__,datasets=datasets,points_order=COMMON,point_units='96x96 crop pixels; first point wrist; all20common model/detector landmarks',local_labels='Strong independent teacher labels, not physical ground truth',public_labels='Canonical-left source crops, points projected from source physical hand model and recorded labeled pose; data augmentation will mirror pixels and points together',public_manifest_sha256=sha(a.public/'manifest.json'),validation_sequences=sorted(validation),source_sha256=sha(Path(__file__)))
    (a.output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(datasets,indent=2))


if __name__=='__main__':main()
