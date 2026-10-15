"""Counterfactual offline publication gate using exact saved per-capture evidence.

Precomputed baseline association and poses remain authoritative; suppressing a
birth here does not rerun association or prevent its internal state pollution.
Only current rows enter the gate, even though the evidence file is preloaded.
"""
import argparse,hashlib,json,shutil
from pathlib import Path
from acquisition_gate import AcquisitionGate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('baseline','presence','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    rows={}
    for row in json.loads(a.presence.read_text()):rows.setdefault(row['stamp'],[]).append(row)
    snapshot=a.output/'snapshot';snapshot.mkdir()
    for name in ('acquisition_gate.py','replay_acquisition_gate.py','score_roi_presence.py'):
        shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    gate=AcquisitionGate();counts=dict(frames=0,baseline_hands=0,output_hands=0,suppressed=0);births={}
    with a.baseline.open() as f,(a.output/'predictions.jsonl').open('x') as o,(a.output/'decisions.jsonl').open('x') as trace:
        for line in f:
            row=json.loads(line);stamp=row['capture_raw_ns']
            hands,decisions=gate.step(stamp,row['capture_pose']['valid'],row['hands'],rows.get(stamp,[]))
            for d in decisions:
                k=str(d['source_id']);b=births.setdefault(k,dict(side=d['side'],first_ns=stamp,accepted_ns=None,suppressed=0,first_reason=None))
                if d['accepted'] and b['accepted_ns'] is None:b['accepted_ns']=stamp;b['first_reason']=d['reason']
                b['suppressed']+=not d['accepted']
            counts['frames']+=1;counts['baseline_hands']+=len(row['hands']);counts['output_hands']+=len(hands);counts['suppressed']+=len(row['hands'])-len(hands)
            o.write(json.dumps(dict(row,hands=hands),allow_nan=False)+'\n')
            trace.write(json.dumps(dict(stamp=stamp,decisions=decisions))+'\n')
    for b in births.values():b['initial_delay_ms']=None if b['accepted_ns'] is None else (b['accepted_ns']-b['first_ns'])/1e6
    report=dict(scope=__doc__,counts=counts,births=births,args={k:str(v) for k,v in vars(a).items()},source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in snapshot.glob('*.py')})
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(counts))

if __name__=='__main__':main()
