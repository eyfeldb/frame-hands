"""Derive Valve bone coordinate conventions from the installed SteamVR skeletons."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('steamvr',type=Path)
    args=parser.parse_args()
    out=['// Derived numeric rest transforms; Valve OpenVR skeleton convention.\n#pragma once\n#include "math.h"\nnamespace fhs {\nstruct Rest {V p;Q q;};\ninline const Rest rest[2][31]={']
    evidence={}
    for side in ['left','right']:
        p=args.steamvr/f'resources/skeletons/vr_glove_{side}_skeleton.glb'
        raw=p.read_bytes();count=struct.unpack_from('<I',raw,12)[0];model=json.loads(raw[20:20+count])
        nodes=[model['nodes'][i] for i in model['skins'][0]['joints']]
        assert len(nodes)==31 and nodes[0]['scale']==[100.,100.,100.]
        out.append('{')
        for index,n in enumerate(nodes):
            v=[x*100 for x in n.get('translation',[0,0,0])];q=n.get('rotation',[0,0,0,1])
            # SteamVR applies a 180-degree Y correction to root children when
            # loading its GLB. Confirmed against GetSkeletalReferenceTransforms.
            # Exported GLB coordinates alone are not the runtime convention.
            if index==1 or index>=26:
                v=[-v[0],v[1],-v[2]]
                x,y,z,w=q;q=[z,w,-x,-y]
            out.append('{{'+','.join(f'{x:.12g}' for x in v)+'},{'+','.join(f'{x:.12g}' for x in [q[3],*q[:3]])+'}},')
        out.append('},');evidence[side]=dict(file=p.name,sha256=hashlib.sha256(raw).hexdigest())
    out.append('};\n}\n')
    root=Path(__file__).parent
    (root/'reference.h').write_text('\n'.join(out),newline='\n')
    (root/'reference-source.json').write_text(json.dumps(evidence,indent=2)+'\n')


if __name__=='__main__':main()
