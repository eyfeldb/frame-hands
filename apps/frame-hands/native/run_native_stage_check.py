"""Run the complete rig fitting stage on ARM64 saved observations under workflow locks."""
import argparse,hashlib,json,math,platform,subprocess
from pathlib import Path
from workflow_guard import native_workflow
from run_native_jacobian_check import stats

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('bundle','output'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();root=a.bundle.resolve();meta=json.loads((root/'manifest.json').read_text())
 if platform.machine()!='aarch64' or not 1<=meta['cases']<=5000:raise ValueError('Wrong target or unbounded data')
 for name,value in meta['sha256'].items():
  if Path(name).name!=name or hashlib.sha256((root/name).read_bytes()).hexdigest()!=value:raise ValueError('Changed bundle')
 required=['replay-rig-stage','generic.fhrig','side0.fhrig','side1.fhrig','cameras.txt','input.txt','host.txt']
 if any(n not in meta['sha256'] for n in required):raise ValueError('Unverified input')
 if a.output.exists():raise FileExistsError(a.output)
 def read(path):
  data=[list(map(float,line.split())) for line in path.read_text().splitlines()]
  if len(data)!=meta['cases'] or any(len(row)!=104 or not all(map(math.isfinite,row)) for row in data):raise ValueError('Invalid output')
  return data
 with native_workflow():
  a.output.mkdir(parents=True);target=a.output/'native.txt'
  result=subprocess.run([str(root/n) for n in required[:6]]+[str(target.resolve())],capture_output=True,text=True,timeout=90)
  (a.output/'stdout.txt').write_text(result.stdout);(a.output/'stderr.txt').write_text(result.stderr)
  if result.returncode:raise RuntimeError('Stage failed')
  actual=read(target);host=read(root/'host.txt');maximum=0.;history=0
  for x,y in zip(actual,host):
   if x[0]!=y[0]:raise ValueError('Native initialization mismatch')
   maximum=max(maximum,max(abs(v-w) for v,w in zip(x[41:],y[41:])));history+=x[5]!=y[5]
  stamps=[int(line.split()[0]) for line in (root/'input.txt').read_text().splitlines()];frames={}
  for stamp,row in zip(stamps,actual):frames[stamp]=frames.get(stamp,0)+row[1]
  report=dict(scope=__doc__,cases=meta['cases'],frames=len(frames),max_host_native_coordinate_difference_mm=maximum,history_selection_changes=history,numerical_screen_pass=maximum<=.05,native_per_hand_stage_us=stats([x[1] for x in actual]),native_per_frame_summed_stage_us=stats(list(frames.values())),manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),environment=dict(kernel=platform.release(),python=platform.python_version(),os_release=Path('/etc/os-release').read_text()),limitations=['One batch, no warmup exclusion or paired performance comparison; stage timings exclude file IO.','Saved stereo inputs/identities; no camera, detector, triangulation, monocular mode, XR, integration or physical accuracy validation.','Both workflow locks held. Wearer standalone versus streamed state unobserved; computation is native ARM64.'])
  (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
  if not report['numerical_screen_pass']:raise ValueError('Native numerical screen failed')
if __name__=='__main__':main()
