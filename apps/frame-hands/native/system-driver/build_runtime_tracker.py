"""Rebuild accepted v9 with a no-record trace sink and optional-seed projection guard."""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'native'))
from run_native_trial import configuration

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    spec=configuration(BASE/'accepted-config.json');original=spec['build'];meta=json.loads((original/'build.json').read_text())
    source=BASE/'tracker-runtime-source'
    if source.exists():raise RuntimeError('Runtime source already exists; preserve prior build')
    shutil.copytree(original/'source',source,ignore=shutil.ignore_patterns('__pycache__'))
    hook=source/'native/tracker_rig_hook.h';before=hook.read_text()
    old='||std::ifstream(path).good())';new='||(std::string(path)!="/dev/null"&&std::ifstream(path).good()))'
    if before.count(old)!=1:raise RuntimeError('Unexpected accepted trace guard')
    hook.write_text(before.replace(old,new))
    fitter=source/'native/whole_pose_fitter.h';before=fitter.read_text()
    old='            if(out.eligible){auto alternate=fit_rig_pixels(rig,seed,cameras,pixels,{},0,&fresh.angles);'
    new='            // History can put an observed fingertip behind a camera even when\n            // the current baseline is valid. Reject only this optional seed.\n            if(out.eligible){auto projected=rig.landmarks(seed.angles,seed.wrist,side);\n                for(int i=0;i<21;++i)for(int view=0;view<2;++view)if(pixels[i].present[view]){\n                    V uv;if(!cameras[view].project(projected[i],uv))out.eligible=false;}}\n            if(out.eligible){auto alternate=fit_rig_pixels(rig,seed,cameras,pixels,{},0,&fresh.angles);'
    if before.count(old)!=1:raise RuntimeError('Unexpected accepted whole-pose seed call')
    fitter.write_text(before.replace(old,new))
    causal=source/'native/causal_rig_fit.h';before=causal.read_text()
    old='        // Same prior in both branches: isolate initialization from temporal regularization.'
    new='        // A failed projection is a rejected observation, not a fatal service error.\n        if(!std::isfinite(out.used_history?out.history_seed_cost:out.fresh_seed_cost)){\n            out.fit.pose=fresh;h={false,source_id,stamp,{}};return out;\n        }\n        // Same prior in both branches: isolate initialization from temporal regularization.'
    compare='if(compare_final&&out.used_history){'
    if before.count(old)!=1 or before.count(compare)!=1:raise RuntimeError('Unexpected accepted causal seed calls')
    causal.write_text(before.replace(old,new).replace(compare,'if(compare_final&&out.used_history&&std::isfinite(out.fresh_seed_cost)){'))
    changes=[]
    for file in source.rglob('*'):
        if file.is_file() and sha(file)!=sha(original/'source'/file.relative_to(source)):changes.append(str(file.relative_to(source)))
    if sorted(changes)!=['native/causal_rig_fit.h','native/tracker_rig_hook.h','native/whole_pose_fitter.h']:raise RuntimeError('Unexpected source change: '+repr(changes))
    binary=BASE/'tracker-no-record'
    command=[str(arg).replace(str(original/'source'),str(source)).replace(str(original/'ft-hands-native-rig'),str(binary)) for arg in meta['command']]
    library=next(Path(arg) for arg in command if arg.endswith('/libncnn.a'))
    if sha(library)!=meta['provenance']['ncnn_library_sha256']:raise RuntimeError('Changed inference library')
    subprocess.run(command,check=True)
    result=dict(accepted_binary_sha256=meta['binary_sha256'],binary_sha256=sha(binary),changed_files=changes,change='Allow /dev/null trace and reject optional whole-pose seeds that cannot project; retain fresh baseline',command=command,original_hook_sha256=sha(original/'source/native/tracker_rig_hook.h'),runtime_hook_sha256=sha(hook),original_fitter_sha256=sha(original/'source/native/whole_pose_fitter.h'),runtime_fitter_sha256=sha(fitter),runtime_causal_sha256=sha(causal))
    (BASE/'tracker-runtime-build.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
