"""Build only the pinned offline Mercury kinematic DLL; no runtime or driver install."""
import argparse,hashlib,json,subprocess
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--with-forward-generator',action='store_true');p.add_argument('--with-stereo-initializer',action='store_true');p.add_argument('--with-curl-initializer',action='store_true');a=p.parse_args();w=a.workspace.resolve()
 m=w/'.local/hand-alternatives/monado';e=w/'.local/hand-alternatives/eigen-3.4.0';out=a.output.resolve() if a.output else w/'.local/hand-alternatives/mercury-solver';out.mkdir(parents=True,exist_ok=True)
 for path,revision in [(m,'22d5c936ccc7218e7d139c7d921456c49ff5aaf1'),(e,'3147391d946bb4b6c68edd901f2add6ac1f31f8c')]:
  actual=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
  if actual!=revision:raise ValueError('Unexpected dependency revision')
  if subprocess.check_output(['git','-C',str(path),'status','--porcelain','--untracked-files=no'],text=True).strip():raise ValueError('Modified dependency sources')
 config=out/'include/xrt';config.mkdir(parents=True,exist_ok=True)
 for name,body in [('xrt_config_os.h','#pragma once\n#include "'+(m/'src/xrt/include/xrt/xrt_config_os.h').as_posix()+'"\n'),('xrt_config_have.h','#pragma once\n'),('xrt_config_build.h','#pragma once\n')]:config.joinpath(name).write_text(body)
 tool=Path('C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207');sdk=Path('C:/Program Files (x86)/Windows Kits/10');version='10.0.26100.0'
 includes=[tool/'include',sdk/'Include'/version/'ucrt',sdk/'Include'/version/'shared',sdk/'Include'/version/'um',out/'include',config,m/'src/xrt/include',m/'src/xrt/auxiliary',m/'src/xrt/tracking/hand/mercury',m/'src/external/tinyceres/include',e]
 sources=[Path(__file__).with_name('bridge.cpp'),Path(__file__).with_name('scale_bridge.cpp'),m/'src/xrt/tracking/hand/mercury/hg_hand_size_opt.cpp',m/'src/xrt/tracking/hand/mercury/kine_lm/lm_main.cpp',m/'src/xrt/tracking/hand/mercury/kine_lm/lm_hand_init_guesser.cpp',m/'src/xrt/auxiliary/math/m_base.cpp']
 included_sources=[]
 if a.with_forward_generator or a.with_stereo_initializer or a.with_curl_initializer:
  if a.output is None:raise ValueError('Forward generator requires a separate explicit output directory')
  native=m/'src/xrt/tracking/hand/mercury/kine_lm/lm_main.cpp';extension=Path(__file__).with_name('forward.inc').resolve();wrapper=out/'kernel_forward.cpp'
  wrapper.write_text('#include \"'+native.as_posix()+'\"\n#include \"'+extension.as_posix()+'\"\n')
  sources[sources.index(native)]=wrapper;included_sources=[native,extension]
  if a.with_stereo_initializer or a.with_curl_initializer:
   stereo=Path(__file__).with_name('stereo_init.inc').resolve();wrapper.write_text(wrapper.read_text()+'#include \"'+stereo.as_posix()+'\"\n');included_sources.append(stereo)
  if a.with_curl_initializer:
   curl=Path(__file__).with_name('curl_init.inc').resolve();wrapper.write_text(wrapper.read_text()+'#include \"'+curl.as_posix()+'\"\n');included_sources.append(curl)
 command=[str(tool/'bin/Hostx64/x64/cl.exe'),'/nologo','/std:c++20','/EHsc','/O2','/MT','/LD','/bigobj','/DNOMINMAX','/D_CRT_SECURE_NO_WARNINGS']+['/I'+str(path) for path in includes]+[str(s.resolve()) for s in sources]+(['/DFH_SYNTHETIC_DIAGNOSTICS'] if a.with_forward_generator else [])+(['/DFH_STEREO_INITIALIZER'] if a.with_stereo_initializer or a.with_curl_initializer else [])+(['/DFH_CURL_INITIALIZER'] if a.with_curl_initializer else [])+['/Fe:'+str(out/'mercury_solver.dll'),'/link']+['/LIBPATH:'+str(path) for path in [tool/'lib/x64',sdk/'Lib'/version/'ucrt/x64',sdk/'Lib'/version/'um/x64']]+['winmm.lib']
 completed=subprocess.run(command,cwd=out,check=True);dll=out/'mercury_solver.dll'
 manifest=dict(forward_generator=a.with_forward_generator,stereo_initializer=a.with_stereo_initializer or a.with_curl_initializer,curl_initializer=a.with_curl_initializer,included_sources={str(s):hashlib.sha256(s.read_bytes()).hexdigest() for s in included_sources},msvc='14.44.35207',sdk=version,monado='22d5c936ccc7218e7d139c7d921456c49ff5aaf1',eigen='3147391d946bb4b6c68edd901f2add6ac1f31f8c',sources={str(s):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},dll_sha256=hashlib.sha256(dll.read_bytes()).hexdigest(),command=command)
 (out/'build.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(dict(dll=str(dll),sha256=manifest['dll_sha256'])))
if __name__=='__main__':main()
