"""Re-run an exact saved mesh-trial crop stream and recorded capture poses. No XR/device access."""
import argparse
import io
import json
from pathlib import Path
import struct
import sys
import cv2
import numpy as np
from frame_data import read_audit,load_calibration
from live_backend import Backend
from run_mesh_trial import read_exact,pose_matrix


def packets(path):
    with Path(path).open('rb') as f:
        while header:=f.read(4):
            if len(header)!=4:raise ValueError('Truncated transport header')
            size,=struct.unpack('<I',header)
            if not 0<size<1_000_000:raise ValueError('Invalid transport header')
            meta=json.loads(read_exact(f,size))
            lengths=meta.get('lengths',[])
            if len(lengths)>8 or any(not 0<n<8_000_000 for n in lengths):raise ValueError('Invalid image lengths')
            yield meta,[read_exact(f,n) for n in lengths]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--association',choices=('fresh','temporal','bearing','continuity'),default='fresh')
    p.add_argument('--memory',choices=('world','off'),default='world',help='Offline learned-memory ablation; association history is retained')
    a=p.parse_args()
    sys.path.insert(0,str(a.upstream))
    baseline={v['capture_raw_ns']:v for line in (a.session/'predictions.jsonl').open() if (v:=json.loads(line))}
    backend=Backend(a.upstream,load_calibration(a.session/'calibration'),association=a.association)
    a.output.mkdir(parents=True,exist_ok=False)
    differences=[];frames=0;hands=0
    with (a.output/'predictions.jsonl').open('x') as out:
        for meta,blobs in packets(a.session/'transport.bin'):
            if meta.get('waiting'):continue
            old=baseline.get(meta['stamp'])
            if old is None:
                backend.previous={};continue
            observation=next(iter(read_audit(io.StringIO(meta['audit'])).values()))
            prepared={(v['camera'],v['id']):(cv2.imdecode(np.frombuffer(b,np.uint8),0),v) for v,b in zip(meta['crops'],blobs)}
            world=pose_matrix(old['capture_pose']['capture']) if old['capture_pose']['valid'] else None
            if a.memory=='off':backend.previous={}
            result=backend.predict(observation,{},world,prepared)
            if not old['current_pose']['valid']:result['hands']=[]
            out.write(json.dumps(dict(capture_raw_ns=meta['stamp'],capture_mono_ns=old['capture_mono_ns'],capture_pose=old['capture_pose'],current_pose=old['current_pose'],**result),allow_nan=False)+'\n')
            frames+=1
            if a.association!='fresh' or a.memory!='world':
                hands+=len(result['hands']);continue
            for hand,reference in zip(result['hands'],old['hands']):
                if (hand['side'],hand['source_id'])!=(reference['side'],reference['source_id']):raise ValueError('Assignment mismatch')
                differences.append(float(np.max(np.abs(np.array(hand['vertices_world_mm'])-reference['vertices_world_mm']))));hands+=1
            if len(result['hands'])!=len(old['hands']):raise ValueError('Availability mismatch')
    summary=dict(association=a.association,memory=a.memory,frames=frames,hands=hands,maximum_vertex_difference_mm=max(differences,default=None),scope='Exact-input repeatability, not physical accuracy' if a.association=='fresh' and a.memory=='world' else 'Alternate association/memory on exact recorded crops; visual validation required')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
