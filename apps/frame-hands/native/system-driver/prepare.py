"""Stage from the exact accepted native recording snapshot, without starting it."""
import argparse,hashlib,json,shutil
from pathlib import Path
BASE=Path(__file__).resolve().parent
HANDS=BASE.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime',choices=('accepted','steamvr-2.18.2'),default='accepted');p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    frozen=json.loads(a.checkpoint.read_text());trial=a.checkpoint.parent/'frame-trial-03'
    if a.output.exists():raise FileExistsError(a.output)
    for name,digest in frozen['files'].items():
        if name.startswith('source/viewer/') and sha(trial/name)!=digest:raise ValueError('Changed accepted viewer: '+name)
    out=a.output;out.mkdir(parents=True)
    for name in ('native_service.py','wire.py','control.py','install.py','build.sh','pose_service.cpp','verify.py','probe_capabilities.cpp','build_runtime_tracker.py','notify.cpp','test_trace_startup.cpp','test_seed_rejection.cpp','export_pc.py','controller_watch.cpp','controller_takeover.h','test_controller_takeover.cpp','install_controller_watch.py','menu.cpp','menu.py','install_menu.py'):
        shutil.copyfile(BASE/name,out/name)
    (out/'repair_exact_permissions.sh').write_text((HANDS/'camera-recovery/repair_exact_permissions.sh').read_text(),newline='\n')
    (out/'native').mkdir()
    for name in ('run_native_trial.py','workflow_guard.py','rig_pose_subscriber.py','rig_pose_bridge.py'):
        shutil.copyfile(trial/'source'/name,out/'native'/name)
    evidence=json.loads((HANDS/'native/evidence/native-wearer-20261008.json').read_text())
    config=evidence['config']
    # Device-verified 2026-10-09 after SteamOS 0.4.5 reboot; no arbitrary hash override.
    if a.runtime=='steamvr-2.18.2':
        config['files']['openvr']['sha256']='a07163c436b3d791a0aa2442029d2ef0ba40a772ebf1dc6006d610c3a7934ee2'
    (out/'accepted-config.json').write_text(json.dumps(config,indent=2)+'\n')
    processor=out/'processor';processor.mkdir()
    for name in ('baseline_motion.gd','hand_motion.gd','profile_assets.gd','profile-rigs.json','side0.json','side1.json','profile_rig.gd','base_rig.gd'):
        shutil.copyfile(trial/'source/viewer'/name,processor/name)
    shutil.copyfile(BASE/'native_bridge.gd',processor/'native_bridge.gd')
    from prepare_gap import prepare as prepare_gap
    prepare_gap(processor)
    shutil.copyfile(BASE/'test_fk.gd',processor/'test_fk.gd')
    shutil.copyfile(BASE/'test_native_gap.gd',processor/'test_native_gap.gd')
    (out/'build.sh').write_text((out/'build.sh').read_text(),newline='\n')
    shutil.copyfile(HANDS/'exploratory/piano-provider/rig_joints.gd',processor/'rig_joints.gd')
    (processor/'project.godot').write_text('config_version=5\n[application]\nconfig/name="Frame Native Hands Background"\n[rendering]\nrenderer/rendering_method="gl_compatibility"\n[xr]\nopenxr/enabled=false\n')
    shutil.copytree(HANDS/'pc-input',out/'pc-input',ignore=shutil.ignore_patterns('build','__pycache__','evidence','driver'))
    shutil.copytree(HANDS/'system-input/vendor',out/'vendor')
    target=out/'driver/fh_skeletal';shutil.copytree(HANDS/'pc-input/driver/fh_skeletal',target,ignore=shutil.ignore_patterns('bin'))
    (target/'bin/linuxarm64').mkdir(parents=True)
    for name in ('LICENSE-OpenVR','LICENSE-Frametop'):shutil.copyfile(out/'vendor'/name,target/name)
    fixture=[]
    with (trial/'subscriber/trace.jsonl').open() as stream:
        for line in stream:
            packet=json.loads(line).get('packet',{})
            if packet.get('context_valid') and len(packet.get('hands',[]))==2:fixture.append(packet)
            if len(fixture)==30:break
    if len(fixture)!=30:raise RuntimeError('Accepted native pose fixture unavailable')
    (out/'pose-fixture.json').write_text(json.dumps(fixture))
    print(json.dumps(dict(staged=str(out),baseline='tracker-rig-v9 / b9432e5',no_start=True)))
if __name__=='__main__':main()
