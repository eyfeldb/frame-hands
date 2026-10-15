"""Wall-clock native replay attribution; saved data only, existing replay guards."""
import argparse,collections,hashlib,json,runpy,sys,time
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'exploratory'))
from native_pipeline import NativeIdentityPipeline


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    args,remaining=parser.parse_known_args()
    if args.report.exists():raise FileExistsError(args.report)
    samples=collections.defaultdict(list);frames=[]
    original=NativeIdentityPipeline.__init__
    def init(self,*a,**kw):
        original(self,*a,**kw)
        def wrap(obj,name,label):
            method=getattr(obj,name)
            def measured(*a,**kw):
                blocks=sum(self.bank.elapsed.values());started=time.perf_counter()
                try:return method(*a,**kw)
                finally:samples[label].append(((time.perf_counter()-started)*1000,sum(self.bank.elapsed.values())-blocks))
            setattr(obj,name,measured)
        wrap(self.evidence,'score','evidence_total')
        wrap(self.evidence.regressor,'extract','evidence_extract')
        wrap(self.evidence.regressor,'regress','evidence_regress')
        wrap(self.evidence.backend,'landmarks','evidence_landmarks')
        wrap(self.backend,'predict','pose_total')
        wrap(self.backend.ownership,'assign','association')
        wrap(self.backend.prepared_regressor,'regress','pose_regress')
        wrap(self.backend,'mesh','mesh')
        method=self.predict
        def predict(*a,**kw):
            before={k:len(v) for k,v in samples.items()};start=time.perf_counter()
            result=method(*a,**kw)
            frames.append(dict(ms=(time.perf_counter()-start)*1000,hands=len(result['hands']),stages={k:sum(row[0] for row in v[before.get(k,0):]) for k,v in samples.items()}))
            return result
        self.predict=predict
    NativeIdentityPipeline.__init__=init
    sys.argv=[str(BASE/'replay_native.py'),*remaining]
    runpy.run_path(str(BASE/'replay_native.py'),run_name='__main__')
    rows={}
    for name,values in samples.items():
        x=np.asarray(values);rows[name]=dict(calls=len(x),total_ms=float(x[:,0].sum()),neural_ms=float(x[:,1].sum()),outside_neural_ms=float((x[:,0]-x[:,1]).sum()),per_call_ms_p50_p95=np.percentile(x[:,0],[50,95]).tolist())
    result=dict(scope='Wall attribution; nested categories overlap and instrumentation adds overhead. Same saved replay and existing guards; no capture/render.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),frames=len(frames),stages=rows)
    args.report.write_text(json.dumps(result,indent=2)+'\n')
    args.report.with_suffix('.frames.json').write_text(json.dumps(frames)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
