"""Guarded native causal replay, using saved inputs only."""
import argparse,hashlib,json,math,platform,subprocess
from pathlib import Path
from workflow_guard import native_workflow
from run_native_jacobian_check import stats

def read(path):
    data=[list(map(float,line.split())) for line in path.read_text().splitlines()]
    if not data or any(len(row)!=104 or not all(map(math.isfinite,row)) for row in data):raise ValueError('Invalid output')
    return data

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.bundle.resolve();meta=json.loads((root/'manifest.json').read_text())
    if platform.machine()!='aarch64' or not 1<=meta['cases']<=5000:raise ValueError('Wrong target or unbounded batch')
    for name,expected in meta['sha256'].items():
        if Path(name).name!=name or hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:raise ValueError('Changed file '+name)
    for name in ('evaluate-causal-rig','side0.fhrig','side1.fhrig','cameras.txt','input.txt','fresh.txt','guarded.txt'):
        if name not in meta['sha256']:raise ValueError('Missing required verified file')
    if a.output.exists():raise FileExistsError(a.output)
    with native_workflow():
        a.output.mkdir(parents=True);results=[];outputs={}
        for mode in ('fresh','guarded'):
            target=a.output/(mode+'.txt')
            subprocess.run([str(root/'evaluate-causal-rig'),str(root/'side0.fhrig'),str(root/'side1.fhrig'),str(root/'cameras.txt'),str(root/'input.txt'),str(target.resolve()),mode],check=True,capture_output=True,text=True,timeout=60)
            actual=read(target);expected=read(root/(mode+'.txt'));outputs[mode]=actual
            if len(actual)!=meta['cases'] or len(expected)!=meta['cases']:raise ValueError('Incomplete replay')
            worst=0.;history_changes=0
            for x,y in zip(actual,expected):
                if x[:2]!=y[:2] or x[8]!=y[8] or x[5]>x[4]+1e-9:raise ValueError('Invalid fit invariants')
                worst=max(worst,max(abs(v-w) for v,w in zip(x[36:99],y[36:99])));history_changes+=x[100]!=y[100]
            results.append(dict(mode=mode,max_host_native_coordinate_difference_mm=worst,history_selection_changes=history_changes,numerical_screen_pass=worst<=.05,pixel_fit_us=stats([x[7] for x in actual])))
        regressions=sum(g[5]>f[5]+1e-8 for f,g in zip(outputs['fresh'],outputs['guarded']))
        report=dict(scope=__doc__,cases=meta['cases'],results=results,guarded_objective_regressions=regressions,manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),environment=dict(machine=platform.machine(),kernel=platform.release(),python=platform.python_version(),os_release=Path('/etc/os-release').read_text()),limitations=['One paired batch, no warmup exclusion; timing includes seed scoring and optional second pixel fit, excludes 3D initialization/capture/detector/rendering.','Both locks held; no cameras/XR. Wearer mode not observed.','Numerical agreement is not physical pose accuracy or wearer acceptance.'])
        (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
