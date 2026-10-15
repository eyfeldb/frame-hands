"""Convert the user's installed Valve glove meshes; do not redistribute their assets."""
import argparse,hashlib,json,struct
from pathlib import Path
import numpy as np

def export(source,dest,side):
    blob=source.read_bytes();magic,version,length=struct.unpack_from('<III',blob)
    if magic!=0x46546c67 or version!=2 or length!=len(blob):raise ValueError('Invalid GLB')
    chunks={};at=12
    while at<len(blob):
        size,kind=struct.unpack_from('<II',blob,at);chunks[kind]=blob[at+8:at+8+size];at+=8+size
    doc=json.loads(chunks[0x4e4f534a]);binary=chunks[0x004e4942]
    def accessor(index):
        a=doc['accessors'][index];v=doc['bufferViews'][a['bufferView']]
        if a.get('sparse') or v.get('byteStride'):raise ValueError('Unsupported packed accessor')
        dtype={5123:'<u2',5125:'<u4',5126:'<f4'}[a['componentType']]
        cols={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}[a['type']]
        return np.frombuffer(binary,dtype=dtype,count=a['count']*cols,offset=v.get('byteOffset',0)+a.get('byteOffset',0)).reshape(-1,cols).copy()
    primitive=doc['meshes'][0]['primitives'][0];a=primitive['attributes'];skin=doc['skins'][0]
    if primitive.get('mode',4)!=4:raise ValueError('Need triangles')
    names=['Root','wrist','finger_thumb_0','finger_thumb_1','finger_thumb_2','finger_thumb_end']
    for f in ('index','middle','ring','pinky'):names += [f'finger_{f}_meta',f'finger_{f}_0',f'finger_{f}_1',f'finger_{f}_2',f'finger_{f}_end']
    names += [f'finger_{f}_aux' for f in ('thumb','index','middle','ring','pinky')]
    suffix='_r' if side else '_l'
    def canonical(name):return name.replace(suffix+'_','_').removesuffix(suffix)
    mapping=[names.index(canonical(doc['nodes'][i]['name'])) for i in skin['joints']]
    pos=accessor(a['POSITION']);normal=accessor(a['NORMAL']);weights=accessor(a['WEIGHTS_0']);joint=accessor(a['JOINTS_0']).astype(int)
    bind=accessor(skin['inverseBindMatrices']).reshape(-1,4,4).transpose(0,2,1)
    if len(mapping)!=len(bind) or not np.isfinite(bind).all():raise ValueError('Skin matrix mismatch')
    indices=accessor(primitive['indices']).reshape(-1)
    if len(pos)>65000 or len(indices)%3 or indices.max()>=len(pos):raise ValueError('Invalid topology')
    used,indices=np.unique(indices,return_inverse=True)
    pos,normal,weights,joint=pos[used],normal[used],weights[used],joint[used]
    sums=weights.sum(1)
    if not np.isfinite(pos).all() or not np.isfinite(weights).all() or (weights<0).any() or (sums<=0).any():raise ValueError('Invalid weights')
    normalization_error=float(np.max(abs(sums-1)))
    weights=weights/sums[:,None]
    matrices=np.repeat(np.eye(4)[None],31,axis=0)
    for i,b in enumerate(mapping):matrices[b]=bind[i]
    mapped=np.array(mapping)[joint]
    # Row-major matrices; vertices contain position, normal, 4 bone ids, 4 weights.
    data=struct.pack('<8sIIII',b'FHMSH01',side,len(pos),len(indices),31)
    data+=matrices.astype('<f4').tobytes()
    for i in range(len(pos)):data+=struct.pack('<6f4I4f',*pos[i],*normal[i],*mapped[i],*weights[i])
    data+=indices.astype('<u4').tobytes();dest.write_bytes(data)
    return dict(side=side,source_sha256=hashlib.sha256(blob).hexdigest(),asset_sha256=hashlib.sha256(data).hexdigest(),vertices=len(pos),triangles=len(indices)//3,max_original_weight_sum_error=normalization_error)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    report=[export(a.source/f'vr_glove_{side}_model_slim.glb',a.output/f'{side}.mesh',i) for i,side in enumerate(('left','right'))]
    (a.output/'assets.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
