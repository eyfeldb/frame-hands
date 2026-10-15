"""Compare source filters using recorded unfiltered poses; no model/camera/XR access."""
import argparse,json,subprocess,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
for name in ('poses','executable','output'):p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--historical-unguarded',action='store_true')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
rows=[]
with (a.output/'input.txt').open('x') as f:
 for line in a.poses.read_text().splitlines():
  v=line.split()
  if len(v)!=194:raise ValueError('Need exact model/filtered/raw recording')
  t=float(v[0]);ident=int(v[1]);data=np.array(list(map(float,v[5:]))).reshape(3,21,3)
  f.write(' '.join(map(str,[round(t*1e9),ident,*data[2].reshape(-1)]))+'\n');rows.append((t,ident,data))
subprocess.run([str(a.executable.resolve()),str((a.output/'input.txt').resolve()),str((a.output/'output.txt').resolve())]+(['--historical-unguarded'] if a.historical_unguarded else []),check=True)
lines=(a.output/'output.txt').read_text().splitlines();tracks={};difference=[]
if len(lines)!=len(rows):raise ValueError('Incomplete output')
for row,line in zip(rows,lines):
 t,ident,data=row;v=line.split();stamp,got=map(int,v[:2])
 if got!=ident or stamp!=round(t*1e9) or len(v)!=130:raise ValueError('Bad recorded join')
 control=np.array(list(map(float,v[3:66]))).reshape(21,3);candidate=np.array(list(map(float,v[67:130]))).reshape(21,3)
 difference.append(float(np.max(np.abs(control-data[1])))*1000)
 tracks.setdefault(ident,[]).append((t,data[2],control,candidate))
def curl(p):
 tip=np.linalg.norm(p[:,[8,12,16,20]]-p[:,0:1],axis=2)
 base=np.linalg.norm(p[:,[5,9,13,17]]-p[:,0:1],axis=2)
 return np.mean(tip/np.maximum(base,.001),axis=1)
reports=[]
for ident,rows in tracks.items():
 t=np.array([r[0] for r in rows]);poses=np.array([r[1:] for r in rows]);raw=curl(poses[:,0]);results=[]
 if len(rows)<100 or np.max(np.diff(t))>.1:continue
 mask=t>=t[0]+.25
 for mode in (1,2):
  curve=curl(poses[:,mode]);errors=[float(np.sqrt(np.mean((curve[mask]-np.interp(t[mask]-lag/1000,t,raw))**2))) for lag in range(0,202,2)]
  idx=int(np.argmin(errors));results.append(dict(best_fit_lag_ms=idx*2,zero_lag_curl_rmse=errors[0],best_lag_rmse=errors[idx],boundary=idx in (0,len(errors)-1)))
 reports.append(dict(id=ident,samples=len(rows),control=results[0],candidate=results[1]))
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
report=dict(scope=__doc__,guard_enabled=not a.historical_unguarded,rows=len(lines),recorded_control_max_coordinate_difference_mm=max(difference),tracks=reports,limitations=['Saved poses have rounded timestamps/coordinates, so replay is not bit-exact to unrounded native execution.','Curl phase measures software delay against noisy raw estimates, not true physical latency or accuracy.','No new identity/visibility policy; native speed and wearer acceptance untested.'],provenance=dict(poses_sha256=sha(a.poses),executable_sha256=sha(a.executable),source_sha256=sha(__file__)))
(a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
