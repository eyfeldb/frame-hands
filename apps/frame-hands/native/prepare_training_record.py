"""Prepare a bounded pinned public record with the upstream calibrated crop path."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import cv2,numpy as np
from fetch_training_record import COMMIT,digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('record','upstream','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    import msgpack
    manifest=json.loads((a.record/'manifest.json').read_text())
    if manifest['commit']!=COMMIT or not manifest['sequence'].endswith('/training'):raise ValueError('Unapproved source/split')
    upstream_sha=subprocess.check_output(['git','-C',str(a.upstream),'rev-parse','HEAD'],text=True).strip()
    if upstream_sha!='e263ec0844abf73457468a050a75853444d33d69':raise ValueError('Changed upstream code version')
    data={}
    for name in ('mono','labels'):
        raw=(a.record/(name+'.record')).read_bytes();spec=manifest['files'][name]
        if len(raw)!=spec['range_bytes'] or digest(raw)!=spec['range_sha256']:raise ValueError('Changed source bytes')
        data[name]=raw
    spec=manifest['files']['mono']
    if spec['dtype_code']!=1:raise ValueError('Unsupported pixels')
    pixels=np.frombuffer(data['mono'],np.uint8).reshape(spec['shape'])
    if pixels.ndim!=4 or pixels.shape[1]!=2:raise ValueError('Expected stereo sequence')
    labels=msgpack.unpackb(data['labels'],raw=False,strict_map_key=True,max_array_len=1000000,max_map_len=128,max_str_len=4096,max_bin_len=8000000)
    if not isinstance(labels,dict):raise ValueError('Expected label mapping')
    sys.path.insert(0,str(a.upstream))
    from lib.batched_dataset.data_transform import preprocess
    model_input,targets=preprocess(dict(mono=pixels,labels=labels),(96,96))
    crops=model_input.left_images.numpy()
    arrays=dict(images=crops,intrinsics=model_input.intrinsics.numpy(),extrinsics=model_input.extrinsics_xf.numpy(),hand_idx=model_input.hand_idx.numpy(),
        generic_joint_angles=targets.preds_targets.joint_angles.numpy(),generic_wrist=targets.preds_targets.wrist_xfs.numpy())
    if crops.shape!=(*pixels.shape[:2],96,96) or not 0<=crops.min()<=crops.max()<=1 or not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('Invalid prepared tensors')
    a.output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(a.output/'training.npz',**arrays)
    panels=[]
    for frame in (0,len(pixels)//2,len(pixels)-1):
        row=[]
        for view in (0,1):
            for title,im in (('public original',pixels[frame,view]),('canonical crop',np.rint(crops[frame,view]*255).astype(np.uint8))):
                image=cv2.cvtColor(cv2.resize(im,(250,250)),cv2.COLOR_GRAY2BGR)
                cv2.rectangle(image,(0,0),(250,28),(0,0,0),-1)
                cv2.putText(image,f'{frame}/{view} {title}',(5,19),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
                row.append(image)
        panels.append(np.hstack(row))
    if not cv2.imwrite(str(a.output/'crop-review.jpg'),np.vstack(panels)):raise ValueError('Review write failed')
    report=dict(scope='Public training-data preparation only; no model training or Frame runtime change.',source_manifest=manifest,upstream_code_commit=upstream_sha,msgpack_version=msgpack.__version__,
        transform_source_sha256={name:digest((a.upstream/name).read_bytes()) for name in ('lib/batched_dataset/data_transform.py','lib/batched_dataset/sample.py','lib/common/crop.py')},
        arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in arrays.items()},pixel_range=[float(crops.min()),float(crops.max())],
        archive_sha256=digest((a.output/'training.npz').read_bytes()),units='Upstream transform converts mm to meters; images float32 [0,1], canonical left-hand crops',
        limitations='One16-frame stereo sequence is only a format/geometry probe. Partial source hashes are local audit hashes, not verification of whole upstream LFS object; no generalization claim.')
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_manifest','transform_source_sha256')},indent=2))


if __name__=='__main__':main()
