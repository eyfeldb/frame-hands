"""Guarded saved-pose ARM64 rig check. No Torch, cameras, capture, XR or driver updates."""
import argparse,hashlib,json,math,platform,subprocess,time
from pathlib import Path
from workflow_guard import native_workflow

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 root=a.bundle.resolve();meta=json.loads((root/'manifest.json').read_text())
 for name,expected in meta['sha256'].items():
  if Path(name).name!=name:raise ValueError('Unexpected manifest path')
  if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:raise ValueError('Changed bundle file: '+name)
 if platform.machine()!='aarch64':raise RuntimeError('ARM64 Frame check required')
 if a.output.exists():raise FileExistsError(a.output)
 with native_workflow():
  a.output.mkdir(parents=True,exist_ok=False)
  begin=time.monotonic()
  result=subprocess.run([str(root/'evaluate-rig'),str(root/'generic.fhrig'),str(root/'poses.txt'),str((a.output/'native.txt').resolve())],capture_output=True,text=True,timeout=20,check=True)
  actual=[list(map(float,line.split())) for line in (a.output/'native.txt').read_text().splitlines()]
  expected=[list(map(float,line.split())) for line in (root/'expected.txt').read_text().splitlines()]
  if len(actual)!=len(expected) or len(actual)!=512:raise ValueError('Incomplete native poses')
  worst=0.
  for x,y in zip(actual,expected):
   if len(x)!=64 or len(y)!=64 or x[0]!=y[0] or not all(math.isfinite(v) for v in x+y):raise ValueError('Pose format mismatch')
   worst=max(worst,max(abs(v-w) for v,w in zip(x[1:],y[1:])))
  if not worst<1e-6:raise ValueError('Native rig mismatch')
  bench=subprocess.run([str(root/'benchmark-rig'),str(root/'generic.fhrig'),str(root/'poses.txt')],capture_output=True,text=True,timeout=20,check=True)
  report=dict(scope=__doc__,max_reference_difference_mm=worst,benchmark=json.loads(bench.stdout),elapsed_wall_s=time.monotonic()-begin,environment=dict(machine=platform.machine(),kernel=platform.release(),python=platform.python_version(),os_release=Path('/etc/os-release').read_text()),manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),limitations=['FK component only; not detector, inverse fitting, complete tracker latency or live acceptance.','Saved generic profiles and poses only; no personal scan.'])
  (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
