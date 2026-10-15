"""Extract the pinned original smooth function for independent C++ equivalence checks."""
import argparse,hashlib,json,re
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
s=(a.source/'tracker.cpp').read_text();start=s.index('void Tracker::smooth(Hand &h, int64_t t_ns) {')
body=s[start:];level=0;end=None
for i,c in enumerate(body[body.index('{'):],body.index('{')):
    if c=='{':level+=1
    if c=='}':
        level-=1
        if level==0:end=i+1;break
if end is None:raise ValueError('Unbalanced source function')
body=body[:end].replace('void Tracker::smooth(Hand &h, int64_t t_ns)','inline void reference_smooth(ReferenceHand &h, int64_t t_ns)').replace('frame_rescue::','reference_rescue::')
# The test explicitly enables the same deployed guard; no process environment change.
body=body.replace('std::getenv("FRAME_HANDS_FILTER_GUARD")','"1"').replace('std::getenv("FRAME_HANDS_RESCUE_AUDIT")','nullptr')
header='#pragma once\n#ifndef M_PI\n#define M_PI 3.14159265358979323846\n#endif\n#include "geom.h"\n#include "stereo_sanity.h"\n#include <algorithm>\n#include <cstdint>\n#include <cstdio>\n#include <string>\nstruct ReferenceHand { V3 pts[21]{},smooth[21]{},dpalm{}; int64_t smooth_ns=0; unsigned frames=0; int id=1; };\nconstexpr double kMinCutoff=2.,kBeta=30.,kSpeedCutoff=1.5;\n'+body+'\n'
(a.output/'source_filter_reference.h').write_text(header)
for name in ('geom.h','stereo_sanity.h'):
    text=(a.source/name).read_text()
    if name=='stereo_sanity.h':text=text.replace('namespace frame_rescue {','namespace reference_rescue {')
    (a.output/name).write_text(text)
(a.output/'manifest.json').write_text(json.dumps(dict(source_sha256={n:hashlib.sha256((a.source/n).read_bytes()).hexdigest() for n in ('tracker.cpp','geom.h','stereo_sanity.h')},adjustments=['standalone function/struct/guard namespace names; Linux M_PI value supplied on Windows','guard explicitly enabled','logging disabled']),indent=2))
print('Extracted original source-stage reference.')
