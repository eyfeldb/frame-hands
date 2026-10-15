"""Integration checks for native mesh, exact image transport and live crop geometry."""
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import base64
import cv2
import numpy as np
from frame_data import Recording, read_audit, load_calibration, crop_camera, capture_poses, warp
from live_backend import Backend
from export_live import Tail
from mesh_packet import packet
from run_mesh_trial import response

ROOT=Path(__file__).resolve().parents[3]
UPSTREAM=ROOT/'.local/hand-alternatives/UmeTrack'
RESULTS=ROOT/'.local/hand-alternatives/results'

class TransportTests(unittest.TestCase):
    def test_binary_shell_framing(self):
        image=np.arange(9216,dtype=np.uint8).reshape(96,96)
        ok,png=cv2.imencode('.png',image)
        self.assertTrue(ok)
        meta=json.dumps(dict(lengths=[len(png)],crops=[dict(camera='slam_left',id=7)])).encode()
        raw=struct.pack('<I',len(meta))+meta+png.tobytes()
        received=response(io.BytesIO(base64.b64encode(raw)+b'\r\n'))
        self.assertEqual(raw,received[2])
        self.assertTrue(np.array_equal(cv2.imdecode(np.frombuffer(received[1][0],np.uint8),0),image))
        with self.assertRaises(EOFError):response(io.BytesIO(base64.b64encode(raw)))

    def test_partial_record_and_audit_wait(self):
        from frame_data import SET,CAM
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)
            row=CAM.pack(b'slam_left',2,2,100,110)
            raw=SET.pack(b'FHSET01\0',1,SET.size+CAM.size+4)+row+b'1234'
            (path/'sets.bin').write_bytes(raw[:-1])
            (path/'live-views.txt').write_text('set 100 200 0\n')
            t=Tail(path);t.update()
            self.assertEqual(len(t.records),0);self.assertEqual(len(t.audits),0)
            with (path/'sets.bin').open('ab') as f:f.write(raw[-1:])
            with (path/'live-views.txt').open('a') as f:f.write('end\n')
            t.update()
            self.assertEqual(list(t.records),[100]);self.assertEqual(list(t.audits),[100])


class MeshIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        manifest=json.loads((RESULTS/'v1-maestro/manifest.json').read_text())
        cls.session=Path(manifest['args']['session'])
        sys.path.insert(0,str(UPSTREAM))
        cls.cameras=load_calibration(cls.session/'frame/snapshot/device/persist')
        cls.raw=Backend(UPSTREAM,cls.cameras)
        cls.cropped=Backend(UPSTREAM,cls.cameras)

    def test_rig_rest_and_reflection(self):
        b=self.raw
        zero=np.zeros(22)
        left=b.mesh(zero,np.eye(4),0)
        self.assertTrue(np.allclose(left,b.hand.mesh_vertices.numpy(),atol=1e-4))
        right=b.mesh(zero,np.eye(4),1)
        self.assertTrue(np.allclose(right,left*np.array([-1,1,1]),atol=1e-4))
        self.assertEqual(left.shape,(788,3))
        self.assertEqual(b.triangles.shape,(1544,3))
        self.assertGreaterEqual(b.triangles.min(),0);self.assertLess(b.triangles.max(),788)
        head=np.eye(4);head[:3,3]=[100,200,300]
        wire=packet(dict(hands=[dict(side=0,source_id=1,vertices_world_mm=left.tolist())]),b.triangles,head,1)
        self.assertTrue(np.allclose(wire['hands'][0]['vertices_head_m'],(left-head[:3,3])*.001))
        self.assertTrue(np.isfinite(wire['hands'][0]['normals_head']).all())

    def test_head_crops_equivalent_to_world_crops(self):
        poses=capture_poses(self.session)
        audits=read_audit(self.session/'frame/live-views.txt')
        maximum=0.;seen=0
        for _,stamp,images in Recording(self.session/'frame/sets.bin').frames(50,51):
            observation=audits.get(stamp)
            if not observation or observation['mono_ns'] not in poses:continue
            prepared={}
            for v in observation['views']:
                if v['camera'] not in self.cameras or v['presence']<.5:continue
                c=self.cameras[v['camera']]
                crop=crop_camera(c,v['points'],0,np.eye(4))
                prepared[v['camera'],v['id']]=(warp(c,crop,images[v['camera']][0]),dict(intrinsics=crop.uv_to_window_matrix().tolist(),camera_to_head_mm=crop.camera_to_world_xf.tolist()))
            world=poses[observation['mono_ns']]
            a=self.raw.predict(observation,{k:v[0] for k,v in images.items()},world)
            b=self.cropped.predict(observation,{},world,prepared)
            self.assertEqual([(h['side'],h['source_id']) for h in a['hands']],[(h['side'],h['source_id']) for h in b['hands']])
            for ha,hb in zip(a['hands'],b['hands']):
                maximum=max(maximum,float(np.max(np.abs(np.array(ha['vertices_world_mm'])-hb['vertices_world_mm']))))
                seen+=1
        print(f'Head/world crop comparison: {seen} hands; maximum vertex difference {maximum:.6f} mm')
        self.assertGreater(seen,20)
        self.assertLess(maximum,.1)

if __name__=='__main__':unittest.main()
