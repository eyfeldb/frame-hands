"""Download one bounded public TRAINING record via verified HTTP byte ranges.

No full-file fallback, testing splits, private uploads or model integration.
Pinned LFS identity is recorded; a partial range cannot verify the whole LFS SHA.
"""
import argparse,hashlib,json,re,urllib.request
from pathlib import Path
import numpy as np
COMMIT='355df534a1c761ac416427925485b311f22ce50e'
RAW=f'https://raw.githubusercontent.com/facebookresearch/UmeTrack_data/{COMMIT}/'
MEDIA=f'https://media.githubusercontent.com/media/facebookresearch/UmeTrack_data/{COMMIT}/'


def digest(data):return hashlib.sha256(data).hexdigest()


def small(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'Accept-Encoding':'identity'}),timeout=45) as response:
        if response.status!=200:raise ValueError('Metadata HTTP status')
        data=response.read(1_000_001)
        if len(data)>1_000_000:raise ValueError('Metadata exceeded cap')
        return data


def parse_index(data):
    if len(data)%8 or len(data)<64:raise ValueError('Truncated index')
    values=np.frombuffer(data,dtype='<i8')
    magic,version,code,itemsize,n,s=map(int,values[:6])
    if magic!=0x584449544E54 or version!=1 or code not in (1,8) or itemsize!=1 or not 0<n<=100000 or s<0:
        raise ValueError('Unsupported index header')
    if len(values)!=6+2*(n+1)+s:raise ValueError('Index length mismatch')
    dims=values[6:6+n+1];offsets=values[6+n+1:6+2*(n+1)];sizes=values[6+2*(n+1):]
    if dims[0]!=0 or dims[-1]!=s or np.any(np.diff(dims)<0) or offsets[0]!=0 or np.any(np.diff(offsets)<=0) or np.any(sizes<=0):
        raise ValueError('Invalid offsets/dimensions')
    shapes=[sizes[dims[i]:dims[i+1]].tolist() for i in range(n)]
    if code==1 and any(int(np.prod(shape,dtype=np.int64))!=offsets[i+1]-offsets[i] for i,shape in enumerate(shapes)):
        raise ValueError('Tensor extent mismatch')
    return dict(code=code,count=n,offsets=offsets.tolist(),shapes=shapes,total_bytes=int(offsets[-1]))


def range_bytes(url,start,end,total):
    expected=end-start+1
    if not 0<expected<=8_000_000:raise ValueError('Record exceeds bounded download cap')
    request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'})
    with urllib.request.urlopen(request,timeout=45) as response:
        if response.status!=206:raise ValueError(f'Range unsupported (HTTP{response.status}); refusing full-file download')
        wanted=f'bytes {start}-{end}/{total}'
        if response.headers.get('Content-Range')!=wanted:raise ValueError('Unexpected Content-Range')
        data=response.read(expected+1)
        if len(data)!=expected:raise ValueError('Truncated/oversized range')
        return data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequence',default='torch_data/real/hand_hand/00000/training')
    p.add_argument('--entry',type=int,default=0)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not re.fullmatch(r'torch_data/(real|synthetic)/(hand_hand|separate_hand)/[0-9]{5}/training',a.sequence):
        p.error('Only explicitly named upstream TRAINING sequences are permitted')
    a.output.mkdir(parents=True,exist_ok=False)
    report=dict(scope=__doc__,commit=COMMIT,sequence=a.sequence,entry=a.entry,files={})
    for name in ('mono','labels'):
        prefix=a.sequence+'/'+name+'.torch'
        idx_bytes=small(RAW+prefix+'.idx');index=parse_index(idx_bytes)
        if not 0<=a.entry<index['count']:raise ValueError('Record index out of range')
        pointer=small(RAW+prefix+'.bin').decode('ascii')
        match=re.fullmatch(r'version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize ([0-9]+)\n?',pointer)
        if not match or int(match[2])!=index['total_bytes']:raise ValueError('LFS/index extent mismatch')
        start,end=index['offsets'][a.entry],index['offsets'][a.entry+1]-1
        (a.output/(name+'.idx')).write_bytes(idx_bytes)
        (a.output/(name+'-lfs-pointer.txt')).write_text(pointer)
        data=range_bytes(MEDIA+prefix+'.bin',start,end,index['total_bytes'])
        (a.output/(name+'.record')).write_bytes(data)
        report['files'][name]=dict(url=MEDIA+prefix+'.bin',range_start=start,range_end=end,range_bytes=len(data),range_sha256=digest(data),index_sha256=digest(idx_bytes),lfs_whole_sha256=match[1],lfs_whole_bytes=index['total_bytes'],whole_hash_verified=False,dtype_code=index['code'],shape=index['shapes'][a.entry])
        (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
