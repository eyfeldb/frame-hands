"""Prepare v3: query both plane identities before any pixel borrowing."""
import hashlib
import json
from pathlib import Path
import shutil
from prepare_direct import prepare_direct, replace

def prepare_exact(source, output):
    expected='6aa6546334db9886a1bc0700108f17c1f903aed242e4143aa64dd309406f0ddd'
    if hashlib.sha256((source/'camd/xrcams.c').read_bytes()).hexdigest()!=expected:
        raise ValueError('Discovery source differs from reviewed baseline')
    prepare_direct(source, output)
    path=output/'camd/xrcams.c'
    code=path.read_text()
    code=replace(code,'bool xr_discover(xr_state_t *st,', '#include "exact_discovery.h"\n\nbool xr_discover(xr_state_t *st,')
    code=replace(code,'        build_groups(st);','        build_groups(st);\n        ok=exact_forward_groups(st,err,errn);')
    path.write_text(code,encoding='utf-8',newline='\n')
    shutil.copyfile(Path(__file__).with_name('exact_discovery.h'),output/'camd/exact_discovery.h')
    manifest=json.loads((output/'source-manifest.json').read_text())
    manifest['files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((output/'camd').iterdir()) if p.is_file()}
    (output/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
