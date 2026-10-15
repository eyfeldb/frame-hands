"""Offline paired replay of frozen PC poses; no inference, camera or XR startup."""
import argparse,hashlib,json,math,os,shutil,subprocess
from pathlib import Path

BASE=Path(__file__).resolve().parent

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('results','output','godot'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--sessions',nargs='+',required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    project=a.output/'project';shutil.copytree(BASE/'mesh-viewer',project,ignore=shutil.ignore_patterns('.godot','*.uid'))
    native=BASE.parent/'native/presentation'
    shutil.copy2(native/'packet_policy.gd',project/'packet_policy.gd')
    s=(native/'replay_packet_motion.gd').read_text()
    s=s.replace('const Candidate = preload("res://hand_motion.gd")','const Candidate = preload("res://continuity_motion.gd")')
    s=s.replace('    var observations :=', '''    for state in states[1]:
        if not state.configure_limits(data.limits):quit(4);return
        if OS.get_environment("FRAME_PC_CONTINUITY_DISABLED") == "1":
            state.correction_enabled = false
            state.anatomy_enabled = false
    var observations :=''')
    s=s.replace('"paired_replay_wall_ms":','"joint_clamps":states[1][0].joint_clamps+states[1][1].joint_clamps,"paired_replay_wall_ms":')
    (project/'replay_continuity.gd').write_text(s,encoding='utf-8')
    env=os.environ.copy();env['FRAME_PACKET_POLICY_DISABLED']='1'
    def godot(script,output,extra=None):
        local=env.copy();local['FRAME_MOTION_RESULT']=str(output.resolve())
        if extra:local.update(extra)
        run=subprocess.run([str(a.godot.resolve()),'--headless','--xr-mode','off','--path',str(project.resolve()),'--script','res://'+script],env=local,capture_output=True,text=True,timeout=90,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        output.with_suffix('.log').write_text(run.stdout+run.stderr,encoding='utf-8')
        if run.returncode or 'SCRIPT ERROR' in run.stderr:raise RuntimeError(run.stdout+run.stderr)
        return json.loads(output.read_text())
    controls=godot('test_continuity.gd',a.output/'controls.json')
    summaries=[]
    for name in a.sessions:
        session=a.results/name;rows=[json.loads(x) for x in (session/'predictions.jsonl').open()]
        rig=json.loads((session/'rig.json').read_text());first=rows[0]['capture_mono_ns'];events=[];previous=-1
        for row in rows:
            stamp=row['capture_mono_ns'];assert stamp>previous;previous=stamp
            t=(stamp-first)*1e-9;hands=[]
            for h in row['hands']:
                wrist=[list(r) for r in h['wrist_world_mm']]
                for r in range(3):wrist[r][3]*=.001
                hands.append(dict(side=h['side'],source_id=h['source_id'],joint_angles=h['joint_angles'],wrist_world_m=wrist))
            events.append(dict(now=t+.05,capture=t,stamp=t,valid=bool(row['capture_pose']['valid'] and row['current_pose']['valid']),hands=hands))
        data=dict(name=name,scope='Unchanged recorded PC world poses; assumed 50ms delivery and 90Hz presentation, not measured live timing',limits=rig['limits'],events=events,ticks=[i/90 for i in range(math.ceil((events[-1]['now']+.18)*90)+1)])
        folder=a.output/name;folder.mkdir();input_path=folder/'input.json';input_path.write_text(json.dumps(data,separators=(',',':')))
        params={'FRAME_MOTION_INPUT':str(input_path.resolve())}
        aa=godot('replay_continuity.gd',folder/'disabled.json',dict(params,FRAME_PC_CONTINUITY_DISABLED='1'))
        result=godot('replay_continuity.gd',folder/'candidate.json',params)
        repeat=godot('replay_continuity.gd',folder/'repeat.json',params)
        assert sha(folder/'candidate.json.frames.jsonl')==sha(folder/'repeat.json.frames.jsonl')
        assert sha(folder/'candidate.json.observations.jsonl')==sha(folder/'repeat.json.observations.jsonl')
        baseline_bad=0;candidate_bad=0;changed_angles=0;max_angle_delta=0.;compared=0
        for line in (folder/'disabled.json.frames.jsonl').open():
            frame=json.loads(line);assert frame['values'][0]==frame['values'][1]
        for line in (folder/'candidate.json.frames.jsonl').open():
            frame=json.loads(line);left,right=frame['values']
            assert bool(left)==bool(right)
            if not left:continue
            compared+=1
            for key in ('id','opacity','age','capture','held','prediction'):assert left[key]==right[key],key
            differences=[]
            for i,(x,y) in enumerate(zip(left['angles'],right['angles'])):
                low,high=rig['limits'][i]
                baseline_bad+=int(x<low-1e-7 or x>high+1e-7)
                candidate_bad+=int(y<low-1e-7 or y>high+1e-7)
                differences.append(abs(x-y))
            delta=max(differences);max_angle_delta=max(max_angle_delta,delta);changed_angles+=int(delta>1e-7)
        assert candidate_bad==0
        summary=dict(session=name,events=len(events),scope=data['scope'],disabled_exact=True,repeat_exact=True,compared_visible_hand_ticks=compared,baseline_out_of_limit_joint_samples=baseline_bad,candidate_out_of_limit_joint_samples=candidate_bad,changed_angle_hand_ticks=changed_angles,max_angle_difference_rad=max_angle_delta,paired=result,provenance=dict(predictions_sha256=sha(session/'predictions.jsonl'),rig_sha256=sha(session/'rig.json'),input_sha256=sha(input_path),output_sha256=sha(folder/'candidate.json.frames.jsonl')))
        summaries.append(summary);print(json.dumps(summary),flush=True)
    report=dict(scope=__doc__,controls=controls,results=summaries,sources={f.name:sha(f) for f in project.iterdir() if f.is_file()},runner_sha256=sha(Path(__file__)),limitations=['Presentation-only recorded-pose replay, no model changes or new images.','No physical accuracy or latency claim from assumed delivery.','Anatomical limits bound individual joint angles; they do not establish collision-free or correct whole-hand poses.','No live PC or SteamVR driver promotion.'])
    (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
