"""Guarded saved-observation two-stage rig check. No cameras or XR."""
import argparse,hashlib,json,math,platform,subprocess,time
from pathlib import Path
from workflow_guard import native_workflow

def read_rows(path):
 rows=[list(map(float,line.split())) for line in path.read_text().splitlines()]
 if any(not all(math.isfinite(x) for x in row) for row in rows):raise ValueError('Nonfinite output')
 return rows

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.bundle.resolve();manifest=root/'manifest.json';meta=json.loads(manifest.read_text())
 for name,expected in meta['sha256'].items():
  if Path(name).name!=name or hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:raise ValueError('Changed bundle file: '+name)
 count=meta['cases'];profile_name=meta.get('profile','generic.fhrig')
 if Path(profile_name).name!=profile_name or profile_name not in meta['sha256']:raise ValueError('Unverified profile')
 if not 1<=count<=512 or platform.machine()!='aarch64':raise ValueError('Wrong count/architecture')
 if a.output.exists():raise FileExistsError(a.output)
 with native_workflow():
  a.output.mkdir(parents=True,exist_ok=False);begin=time.monotonic()
  subprocess.run([str(root/'evaluate-rig-fit'),str(root/profile_name),str(root/'initialization.txt'),str((a.output/'initial.txt').resolve())],check=True,capture_output=True,text=True,timeout=30)
  initial=read_rows(a.output/'initial.txt');templates=read_rows(root/'pixel-template.txt')
  if len(initial)!=count or len(templates)!=count or any(len(x)!=99 for x in initial) or any(len(x)!=199 for x in templates):raise ValueError('Incomplete initialization')
  with (a.output/'pixel-input.txt').open('x') as f:
   for x,y in zip(initial,templates):
    if x[0]!=1 or x[8]!=y[0]:raise ValueError('Invalid initialized hand')
    f.write(' '.join(format(v,'.17g') for v in x[8:36]+y[28:])+'\n')
  subprocess.run([str(root/'evaluate-rig-pixels'),str(root/profile_name),str(root/'cameras.txt'),str((a.output/'pixel-input.txt').resolve()),str((a.output/'fitted.txt').resolve())],check=True,capture_output=True,text=True,timeout=30)
  actual=read_rows(a.output/'fitted.txt');expected=read_rows(root/'expected.txt');worst=0.;times=[]
  if len(actual)!=count or len(expected)!=count:raise ValueError('Incomplete pixel fits')
  for i,(x,y) in enumerate(zip(actual,expected)):
   if len(x)!=99 or len(y)!=99 or x[:2]!=y[:2] or x[8]!=y[8] or x[5]>x[4]+1e-9:raise ValueError('Invalid output/side/objective')
   worst=max(worst,max(abs(v-w) for v,w in zip(x[36:],y[36:])));times.append(x[7]+initial[i][7])
  if worst>.05:raise ValueError('Host/native position difference exceeded .05mm: '+str(worst))
  times.sort();q=lambda p:times[int((len(times)-1)*p)]
  report=dict(scope=__doc__,cases=count,profile_name=profile_name,profile_kind=(root/profile_name).read_text().splitlines()[0].split()[4],max_host_native_position_difference_mm=worst,combined_fit_us=dict(median=q(.5),p95=q(.95),maximum=max(times)),elapsed_wall_s=time.monotonic()-begin,manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),environment=dict(machine=platform.machine(),kernel=platform.release(),python=platform.python_version(),os_release=Path('/etc/os-release').read_text()),limitations=['Combined3D initialization and image refinement only; excludes detector, capture, rendering and interprocess/file overhead.','Host output equivalence is not physical pose accuracy.','Single batch, no warmup exclusion. No accepted runtime changed.'])
  (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
