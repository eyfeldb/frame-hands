"""Replay a saved PC session through accepted continuity and conditional gap bridging."""
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('session','output','godot'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    subprocess.run([sys.executable,str(BASE/'prepare_paired_presentation.py'),'--session',str(a.session),'--candidate',str(a.session/'predictions.jsonl'),'--output',str(a.output/'timeline')],check=True)
    project=a.output/'project';shutil.copytree(BASE/'mesh-viewer',project,ignore=shutil.ignore_patterns('.godot','*.uid'))
    shutil.copy2(BASE/'piano-provider/replay_gap.gd',project/'replay_gap.gd')
    shutil.copy2(BASE.parent/'native/presentation/packet_policy.gd',project/'packet_policy.gd')
    data=json.loads((a.output/'timeline/baseline.json').read_text(encoding='utf-8'))
    data['limits']=json.loads((a.session/'rig.json').read_text(encoding='utf-8'))['limits']
    input_path=a.output/'input.json';input_path.write_text(json.dumps(data),encoding='utf-8')
    output=a.output/'result.json'
    env=dict(os.environ,FRAME_PACKET_POLICY_DISABLED='1',FRAME_MOTION_INPUT=str(input_path.resolve()),FRAME_MOTION_RESULT=str(output.resolve()))
    run=subprocess.run([str(a.godot.resolve()),'--headless','--xr-mode','off','--path',str(project.resolve()),'--script','res://replay_gap.gd'],env=env,capture_output=True,text=True,timeout=60,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    (a.output/'replay.log').write_text(run.stdout+run.stderr,encoding='utf-8')
    if run.returncode or 'SCRIPT ERROR' in run.stderr:raise RuntimeError(run.stdout+run.stderr)
    rows=[json.loads(line) for line in output.with_suffix('.json.frames.jsonl').open()]
    summaries=[]
    for variant in (0,1):
        sides=[]
        for side in (0,1):
            start=None;seen=False;gaps=[]
            for row in rows:
                if row['side']!=side:continue
                value=row['values'][variant];now=row['now']
                if value:
                    assert value['age']<(.220001 if variant else .170001)
                    for angle,(low,high) in zip(value['angles'],data['limits']):assert low-1e-7<=angle<=high+1e-7
                    seen=True
                    if start is not None:gaps.append((now-start)*1000);start=None
                elif seen and start is None:start=now
            sides.append(dict(completed=len(gaps),short_at_most_25ms=sum(g<=25.00001 for g in gaps),max_ms=max(gaps,default=0)))
        summaries.append(sides)
    result=dict(variants=['accepted','bridge'],gaps=summaries,paired=json.loads(output.read_text()),timeline=json.loads((a.output/'timeline/summary.json').read_text()),source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in project.iterdir() if f.is_file()},limits=['Reconstructed consumption/capture ages; not exact compositor timing.','No independent optical accuracy or negative-image validation.','Extra predicted wrist displacement is capped at10mm; total difference from accepted output also includes retained filter history.'])
    (a.output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(summaries))
if __name__=='__main__':main()
