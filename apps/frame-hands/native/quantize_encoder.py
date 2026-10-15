"""Build one full-encoder INT8 candidate from designated training crops only."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','models','tools','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);dest=a.output.resolve();crops=dest/'calibration';crops.mkdir()
    meta=json.loads((a.data/'manifest.json').read_text());models=json.loads((a.models/'manifest.json').read_text())
    for name in ('backbone.ncnn.param','backbone.ncnn.bin'):
        if sha(a.models/name)!=models['sha256'][name]:raise ValueError('Changed full encoder')
    rows=[];paths=[];rng=np.random.default_rng(50105)
    for kind,count in [('train',2048),('public_train',1024)]:
        path=a.data/(kind+'.npz')
        if sha(path)!=meta['datasets'][kind]['archive_sha256']:raise ValueError('Changed training data')
        with np.load(path,allow_pickle=False) as data:images=data['images']
        if images.dtype!=np.float32 or images.shape[1:]!=(1,96,96) or not np.isfinite(images).all() or images.min()<0 or images.max()>1:raise ValueError('Unexpected normalized image format')
        indices=sorted(rng.choice(len(images),count,replace=False).tolist())
        for index in indices:
            for mirror in (False,True):
                x=np.ascontiguousarray(images[index,...,::-1] if mirror else images[index],dtype='<f4')
                target=crops/f'{kind}-{index:05d}-{int(mirror)}.npy';np.save(target,x,allow_pickle=False);paths.append(target.relative_to(dest).as_posix())
        rows.append(dict(split=kind,archive_sha256=sha(path),normal_images=count,indices=indices))
    listing=dest/'calibration.txt';listing.write_text('\n'.join(paths)+'\n',encoding='utf-8')
    exe=a.tools/'ncnn2table.exe';converter=a.tools/'ncnn2int8.exe'
    table=dest/'backbone.table';param=dest/'backbone.ncnn.param';binary=dest/'backbone.ncnn.bin'
    commands=[[str(exe.resolve()),str((a.models/'backbone.ncnn.param').resolve()),str((a.models/'backbone.ncnn.bin').resolve()),str(listing),str(table),'shape=[96,96,1]','thread=4','method=kl','type=1'],[str(converter.resolve()),str((a.models/'backbone.ncnn.param').resolve()),str((a.models/'backbone.ncnn.bin').resolve()),str(param),str(binary),str(table)]]
    report=dict(scope='Post-training full-encoder INT8 candidate. Training inputs only; no native speed or pose fidelity claim.',method='kl',seed=50105,normal_images=3072,mirrored_and_normal_images=len(paths),preprocessing='Existing normalized float32 CHW arrays; exact width reversal for mirror; no resize or normalization in ncnn2table npy path.',datasets=rows,source_model_sha256={n:sha(a.models/n) for n in ('backbone.ncnn.param','backbone.ncnn.bin')},tool_sha256={n:sha(a.tools/n) for n in ('ncnn2table.exe','ncnn2int8.exe','ncnn.dll')},source_sha256=sha(__file__))
    (dest/'manifest-start.json').write_text(json.dumps(report,indent=2)+'\n')
    started=time.perf_counter()
    for i,command in enumerate(commands):
        print('Starting',Path(command[0]).name,flush=True)
        with (dest/f'command-{i}.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800,cwd=dest)
    lines=param.read_text().splitlines()[2:];convs=[line for line in lines if line.split()[0]=='Convolution'];quantized=[line for line in convs if any(field.startswith('8=') and field!='8=0' for field in line.split())]
    if len(convs)!=29 or len(quantized)!=29:raise ValueError('Expected all29encoder convolutions quantized')
    report.update(seconds=time.perf_counter()-started,quantized_convolutions=len(quantized),sha256={n:sha(dest/n) for n in ('backbone.table','backbone.ncnn.param','backbone.ncnn.bin','calibration.txt')})
    (dest/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='datasets'},indent=2))

if __name__=='__main__':main()
