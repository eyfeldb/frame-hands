"""Guarded ARM64 Godot checks. Headless, XR off, saved poses only."""
import argparse,hashlib,json,os,platform,subprocess
from pathlib import Path
from workflow_guard import native_workflow

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','output','godot'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--godot-sha256',required=True);a=p.parse_args();root=a.bundle.resolve();meta=json.loads((root/'manifest.json').read_text())
    if platform.machine()!='aarch64' or sha(a.godot)!=a.godot_sha256:raise ValueError('Wrong target or runtime hash')
    for name,value in meta['sha256'].items():
        if Path(name).name!=name or sha(root/name)!=value:raise ValueError('Changed bundle file')
    for name in ('baseline_motion.gd','hand_motion.gd','test_motion.gd','replay_motion.gd','project.godot','observed.json','assumed-30hz.json','native-rig-motion.json'):
        if name not in meta['sha256']:raise ValueError('Unverified required file')
    if a.output.exists():raise FileExistsError(a.output)
    with native_workflow():
        a.output.mkdir(parents=True);results={}
        version=subprocess.run([str(a.godot),'--headless','--version'],capture_output=True,text=True,check=True,timeout=10).stdout.strip()
        for name in ('test','observed','assumed-30hz','native-rig-motion'):
            env=os.environ.copy();result=(a.output/(name+'-result.json')).resolve();env['FRAME_MOTION_INPUT']=str(root/(name+'.json'));env['FRAME_MOTION_RESULT']=str(result)
            script='res://test_motion.gd' if name=='test' else 'res://replay_motion.gd'
            run=subprocess.run([str(a.godot),'--headless','--path',str(root),'--xr-mode','off','--script',script],capture_output=True,text=True,env=env,timeout=30)
            (a.output/(name+'-stdout.txt')).write_text(run.stdout);(a.output/(name+'-stderr.txt')).write_text(run.stderr)
            if run.returncode or 'ERROR:' in run.stderr:raise RuntimeError('Godot check failed: '+name)
            results[name]=json.loads(result.read_text())
            if results[name].get('failures',0) or results[name].get('visibility_id_opacity_mismatches',0):raise ValueError('Motion invariant failed')
        report=dict(scope=__doc__,results=results,manifest_sha256=sha(root/'manifest.json'),godot_sha256=sha(a.godot),environment=dict(godot=version,kernel=platform.release(),python=platform.python_version(),os_release=Path('/etc/os-release').read_text()),limitations=['No camera, XR, rendering, input injection or accepted-driver change.','Pose replay, not physical accuracy or full native tracking integration. Wearer mode not observed.'])
        (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
