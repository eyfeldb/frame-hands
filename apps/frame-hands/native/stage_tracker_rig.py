"""Stage an isolated, hash-pinned original tracker with optional fixed-rig outputs."""
import argparse,json,shutil
from pathlib import Path
from audit_causal_rig import sha
PINNED={'main.cpp':'f45e92fc511a7e8517b084de1095370ee3baff89aa3373fdeb5b73fc542e762e','tracker.cpp':'17a2a7977ad9b5c630d9147e374940aa8a2f04230a482986bbd30c0e2538e8a1','tracker.h':'4136861486457f8d4fbc8099a2f45c560728e37d46b475e2b69675e2750007e8','replay.cpp':'2134ae7b9b8923c30cd8421bed19808bf1218034b2ee49e073add05fb56b732a'}

def replace_once(text,old,new):
 if text.count(old)!=1:raise ValueError('Patch anchor changed: '+old)
 return text.replace(old,new)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('snapshot','profiles','generic','output'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--capture-head',action='store_true');a=p.parse_args();root=a.output;native=Path(__file__).parent
 for n,value in PINNED.items():
  if sha(a.snapshot/'source/track'/n)!=value:raise ValueError('Unreviewed source '+n)
 root.mkdir(parents=True,exist_ok=False)
 for directory in ('track','camd','include'):shutil.copytree(a.snapshot/'source'/directory,root/directory)
 shutil.copytree(a.snapshot/'models',root/'models');shutil.copytree(a.snapshot/'device',root/'device')
 (root/'native').mkdir();(root/'pc-input').mkdir();(root/'profiles').mkdir()
 for n in ('recorded_head_poses.h','native_head_pose.h','capture_head_pose.h','whole_pose_seed.h','whole_pose_fitter.h','tracker_rig_hook.h','rig_snapshot.h','tracker_rig_adapter.h','rig_stage.h','causal_rig_fit.h','fit_hand_rig.h','fit_rig_pixels.h','rig_bound_step.h','hand_rig.h','test_tracker_rig_adapter.cpp'):shutil.copy2(native/n,root/'native'/n)
 shutil.copy2(native.parent/'pc-input/math.h',root/'pc-input/math.h')
 if a.capture_head:
  (root/'system-input/vendor').mkdir(parents=True);shutil.copy2(native.parent/'system-input/vendor/openvr.h',root/'system-input/vendor/openvr.h')
 shutil.copy2(a.generic,root/'profiles/generic.fhrig')
 for side in (0,1):shutil.copy2(a.profiles/f'side{side}.fhrig',root/'profiles'/f'side{side}.fhrig')
 header=root/'track/tracker.h';s=header.read_text();s=replace_once(s,'#include "joint_identity.h"','#include "joint_identity.h"\n#include "../native/tracker_rig_hook.h"')
 s=replace_once(s,'    Stats stats;','    Stats stats;\n    const std::vector<fhn::TrackerRigOutput>& rig_outputs() const {return rig_hook_.outputs();}\n    void publish_rig(int64_t raw,uint64_t capture,uint64_t now){rig_hook_.publish(raw,capture,now);}\n    void rig_capture_context(int64_t raw,uint64_t mono){rig_hook_.capture_context(raw,mono);}')
 s=replace_once(s,'    JointIdentity joint_identity_;','    JointIdentity joint_identity_;\n    fhn::TrackerRigHook rig_hook_;');header.write_text(s)
 tracker=root/'track/tracker.cpp';s=tracker.read_text();s=replace_once(s,': nets_(nets), pool_(pool), max_views_(max_views) {',': nets_(nets), pool_(pool), max_views_(max_views) {\n    rig_hook_.configure(cams);')
 s=replace_once(s,'    // views split off by a failed triangulation start over as new hands next frame','    // Fit current unfiltered evidence after final source-side recovery. No crop/filter mutation.\n    if (rig_hook_.enabled()) rig_hook_.update(out, views_now(), t_ns);\n    // views split off by a failed triangulation start over as new hands next frame');tracker.write_text(s)
 replay=root/'track/replay.cpp';s=replay.read_text();s=replace_once(s,'    bool cost = false;','    bool cost = false, all_sets = false;\n    std::string step_times;')
 s=replace_once(s,'        else if (a == "--cost") cost = true;','        else if (a == "--cost") cost = true;\n        else if (a == "--all-sets") all_sets = true;\n        else if (a == "--step-times" && more) step_times = argv[++i];')
 s=replace_once(s,'    Pool pool(threads, {2, 3, 4});','    Pool pool(threads, {5, 6, 7});\n    std::ofstream times;\n    if(!step_times.empty()){times.open(step_times);if(!times)return 1;times.precision(17);}')
 s=replace_once(s,'        if (t >= busy_until && t >= next_ns) {','        if (all_sets || (t >= busy_until && t >= next_ns)) {')
 s=replace_once(s,'            if (cost) {','            if(times.is_open()) times << t << " " << ms << " " << out.size() << "\\n";\n            if (cost) {');replay.write_text(s)
 main=root/'track/main.cpp';s=main.read_text();s=replace_once(s,'        const auto hands = tracker.step(images, int64_t(tmin));','        tracker.rig_capture_context(int64_t(tmin),uint64_t(int64_t(tmin)-raw_off));\n        const auto hands = tracker.step(images, int64_t(tmin));');s=replace_once(s,'        const uint64_t capture = uint64_t(int64_t(tmin) - raw_off);   // CLOCK_MONOTONIC','        const uint64_t capture = uint64_t(int64_t(tmin) - raw_off);   // CLOCK_MONOTONIC\n        tracker.publish_rig(int64_t(tmin),capture,mono_ns());');main.write_text(s)
 for n in ('run_tracker_rig_check.py','workflow_guard.py','build_native_tracker.py'):shutil.copy2(native/n,root/n)
 manifest=dict(scope=__doc__,capture_head_support=a.capture_head,base_commit='63e179d',pinned_tracker_sources=PINNED,replay_changes=['Optional all-sets mode for same-input comparisons; default keeps measured busy scheduling','Unmodified step wall time logged before optional cost estimate','Threads pinned to5,6,7 matching source live options'],sha256={str(p.relative_to(root)).replace('\\','/'):sha(p) for p in root.rglob('*') if p.is_file()},limitations=['Staging does not compile or run the tracker.','Optional rig outputs coexist with baseline; no rig-to-mesh/input publication yet.','Only forward stereo views fitted; explicit monocular/missing-view status.'])
 (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(dict(files=len(manifest['sha256']),manifest_sha256=sha(root/'manifest.json'))))
if __name__=='__main__':main()
