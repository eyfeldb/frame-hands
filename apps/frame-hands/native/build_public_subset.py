"""Build a bounded diverse public TRAINING subset; saved-data research only."""
import argparse,collections,concurrent.futures,hashlib,json,subprocess,sys,threading,time
from pathlib import Path
import numpy as np
from fetch_training_record import COMMIT,RAW,small,parse_index


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('tree','upstream','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--per-group',type=int,default=16)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    if not 1<=a.per_group<=16 or not 1<=a.workers<=4:p.error('Bounded limits:1..16sequences/group,1..4workers')
    a.output.mkdir(parents=True,exist_ok=False)
    tree=json.loads(a.tree.read_text());groups=collections.defaultdict(list)
    for row in tree['tree']:
        path=row['path']
        if path.startswith('torch_data/') and path.endswith('/training/mono.torch.bin'):
            parts=path.split('/');groups['/'.join(parts[1:3])].append('/'.join(parts[:-1]))
    if set(groups)!={'real/hand_hand','real/separate_hand','synthetic/hand_hand','synthetic/separate_hand'}:
        raise ValueError('Unexpected dataset groups')
    selection=[]
    for group,sequences in sorted(groups.items()):
        sequences=sorted(sequences)
        ids=np.linspace(0,len(sequences)-1,a.per_group,dtype=int)
        selection += [(group,sequences[i]) for i in ids]
    manifest=dict(scope=__doc__,commit=COMMIT,tree_metadata_sha256=hashlib.sha256(a.tree.read_bytes()).hexdigest(),selection='Evenly spaced sequence ranks per group, records at one-third/two-thirds of each sequence',
        selected_sequences=selection,max_records=128,max_payload_bytes=300000000,workers=a.workers,training_only=True)
    (a.output/'plan.json').write_text(json.dumps(manifest,indent=2)+'\n')
    script=Path(__file__).resolve().parent;lock=threading.Lock();reserved=0;done=[];failures=[]
    started=time.perf_counter()
    def work(item):
        nonlocal reserved
        group,sequence=item
        idx=parse_index(small(RAW+sequence+'/mono.torch.idx'))
        label_index=parse_index(small(RAW+sequence+'/labels.torch.idx'))
        if idx['count']!=label_index['count']:raise ValueError('Unaligned public record indices')
        entries=sorted({idx['count']//3,min(idx['count']-1,2*idx['count']//3)})
        reports=[]
        for entry in entries:
            ident=sequence.replace('/','_')+f'_entry{entry:05d}'
            record=a.output/'records'/ident;prepared=a.output/'prepared'/ident
            # Reserve exact indexed pixel and label extents before downloading.
            # The lower-level fetcher also verifies each response range/length.
            estimate=idx['offsets'][entry+1]-idx['offsets'][entry]+label_index['offsets'][entry+1]-label_index['offsets'][entry]
            with lock:
                if reserved+estimate>manifest['max_payload_bytes']:raise ValueError('Subset byte budget exceeded')
                reserved+=estimate
            record.parent.mkdir(exist_ok=True);prepared.parent.mkdir(exist_ok=True)
            with (a.output/(ident+'.log')).open('w') as log:
                commands=[['fetch_training_record.py','--sequence',sequence,'--entry',str(entry),'--output',str(record)],
                    ['prepare_training_record.py','--record',str(record),'--upstream',str(a.upstream),'--output',str(prepared)]]
                for cmd in commands:
                    subprocess.run([sys.executable,str(script/cmd[0]),*cmd[1:]],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
            report=json.loads((prepared/'manifest.json').read_text())
            source=report['source_manifest']
            if source['commit']!=COMMIT or source['sequence']!=sequence or source['entry']!=entry:raise ValueError('Source identity mismatch')
            if sum(v['range_bytes'] for v in source['files'].values())!=estimate:raise ValueError('Payload differs from reserved indexed budget')
            row=dict(group=group,sequence=sequence,entry=entry,directory=str(prepared.relative_to(a.output)),archive_sha256=report['archive_sha256'],frames=report['arrays']['images']['shape'][0],payload_bytes=sum(v['range_bytes'] for v in source['files'].values()))
            reports.append(row)
            with lock:
                done.append(row)
                (a.output/'progress.json').write_text(json.dumps(dict(completed=done,failures=failures,reserved_bytes=reserved),indent=2)+'\n')
                print(len(done),'records',sequence,entry,flush=True)
        return reports
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(work,item):item for item in selection}
        for future in concurrent.futures.as_completed(futures):
            try:future.result()
            except Exception as error:
                with lock:failures.append(dict(sequence=futures[future][1],error=str(error)))
                for pending in futures:pending.cancel()
                print('FAILED',failures[-1],flush=True)
                break
    if failures:
        (a.output/'failure.json').write_text(json.dumps(failures,indent=2)+'\n')
        raise RuntimeError('Subset incomplete; inspect preserved per-record logs, do not train on partial output')
    report=dict(**manifest,records=sorted(done,key=lambda r:(r['sequence'],r['entry'])),record_count=len(done),frames=sum(r['frames'] for r in done),payload_bytes=sum(r['payload_bytes'] for r in done),seconds=time.perf_counter()-started)
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('records','selected_sequences')},indent=2))


if __name__=='__main__':main()
