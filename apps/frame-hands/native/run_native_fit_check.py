"""Bounded saved-observation native fitter check; no cameras, XR or accepted runtime changes."""
import argparse,hashlib,json,math,platform,subprocess,time
from pathlib import Path
from workflow_guard import native_workflow

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 root=a.bundle.resolve();manifest=root/'manifest.json';meta=json.loads(manifest.read_text())
 for name,expected in meta['sha256'].items():
  if Path(name).name!=name or hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:raise ValueError('Invalid bundle file: '+name)
 if platform.machine()!='aarch64':raise RuntimeError('ARM64 required')
 if a.output.exists():raise FileExistsError(a.output)
 with native_workflow():
  a.output.mkdir(parents=True,exist_ok=False);begin=time.monotonic()
  subprocess.run([str(root/'evaluate-rig-fit'),str(root/'generic.fhrig'),str(root/'observations.txt'),str((a.output/'fitted.txt').resolve())],capture_output=True,text=True,timeout=40,check=True)
  actual=[list(map(float,x.split())) for x in (a.output/'fitted.txt').read_text().splitlines()];expected=[list(map(float,x.split())) for x in (root/'expected.txt').read_text().splitlines()]
  if len(actual)!=450 or len(expected)!=len(actual):raise ValueError('Incomplete output')
  worst=0.;times=[]
  for x,y in zip(actual,expected):
   if len(x)!=99 or len(y)!=99 or not all(math.isfinite(v) for v in x+y) or x[:2]!=y[:2] or x[8]!=y[8]:raise ValueError('Format/visibility/side mismatch')
   worst=max(worst,max(abs(v-w) for v,w in zip(x[36:],y[36:])))
   if x[5]>x[4]+1e-9:raise ValueError('Objective increased')
   if x[0]:times.append(x[7])
  if worst>.01:raise ValueError('Host/native fitted position mismatch: '+str(worst))
  times.sort();quantile=lambda p:times[min(len(times)-1,int((len(times)-1)*p))]
  report=dict(scope=__doc__,cases=len(actual),max_host_native_position_difference_mm=worst,runtime_us=dict(median=quantile(.5),p95=quantile(.95),maximum=max(times)),elapsed_wall_s=time.monotonic()-begin,manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),environment=dict(machine=platform.machine(),kernel=platform.release(),python=platform.python_version(),os_release=Path('/etc/os-release').read_text()),limitations=['Single batch with mixed synthetic corruption, no warmup exclusion; component timing only.','Host output equivalence; host correctness separately checked against independent synthetic reference.','No detector, calibration, temporal/cold-start policy, display latency or wearer acceptance.'])
  (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
