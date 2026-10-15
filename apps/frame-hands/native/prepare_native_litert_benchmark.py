"""Prepare 64 deterministic saved crops and pinned models; never touch Frame."""
import argparse, json, shutil
from pathlib import Path
from litert_benchmark_core import sha

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence',type=Path,required=True)
    parser.add_argument('--package',type=Path,required=True)
    args=parser.parse_args(); root=args.package
    if (root/'manifest.json').exists(): raise RuntimeError('Bundle already frozen')
    source=json.loads((args.sequence/'result.json').read_text(encoding='utf-8-sig'))
    if len(source['records']) != 796: raise RuntimeError('Expected frozen 796-view sequence')
    (root/'inputs').mkdir(); (root/'models').mkdir(); (root/'scripts').mkdir()
    records=[]
    for offset in range(64):
        row=source['records'][(offset*795)//63]; name='inputs/'+str(row['index'])+'.gray'
        source_path=args.sequence/name
        if source_path.stat().st_size != 224*224 or sha(source_path)!=row['native_gray_sha256']:
            raise RuntimeError('Original crop changed')
        shutil.copy2(source_path,root/name)
        records.append({'index':row['index'],'file':name,'case':row['case'],
                        'reference':{model:{k:row['models'][model][k] for k in ('screen','world','presence','right')} for model in ('lite','full')}})
    for model in source['models']:
        path=args.sequence/'models'/(model['name']+'.tflite')
        if sha(path)!=model['sha256']: raise RuntimeError('Model changed')
        shutil.copy2(path,root/'models'/path.name)
    for name in ('litert_benchmark_core.py','run_native_litert_benchmark.py','workflow_guard.py'):
        shutil.copy2(Path(__file__).with_name(name),root/'scripts'/name)
    files={p.relative_to(root).as_posix():sha(p) for directory in ('inputs','models','scripts','runtime') for p in sorted((root/directory).rglob('*')) if p.is_file()}
    for name in ('wheels.json','elf.json'): files[name]=sha(root/name)
    manifest={'scope':'Private saved-crop native benchmark package; no camera or accepted holdout',
              'sequence_result_sha256':sha(args.sequence/'result.json'),'selection':'64 floor(i*795/63) ordered views from frozen 796',
              'models':source['models'],'records':records,'files':files}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'files':len(files),'crops':len(records),'manifest_sha256':sha(root/'manifest.json')}))
if __name__=='__main__': main()
