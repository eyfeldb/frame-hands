"""Recording corruption, exact clock-envelope preservation and offline replay checks."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from recording import SET,CAMERA,TRACE_SIZE,RELAY,RELAY_SIZE,inspect_camera,inspect_fixed
from relay import convert


def trace(stamp=1000, present=False):
    data=bytearray(TRACE_SIZE)
    struct.pack_into('<8sIIQIIII',data,0,b'FHTRACE1',1,TRACE_SIZE,stamp,6,3,3,1)
    for offset in (40,88):struct.pack_into('<12f',data,offset,1,0,0,.1,0,1,0,.2,0,0,1,.3)
    for offset in (136,704):
        struct.pack_into('<8sIIQQQII',data,offset,b'FHPCVR1\0',1,568,stamp,stamp,stamp,0,0)
        if present:
            # Anatomically plausible, synthetic open-hand landmarks.
            points=[(0.,0.,-.4)]
            for finger,x in enumerate((-.05,-.03,0,.025,.05)):
                for joint in range(4):points.append((x, .03+joint*.025 if finger==0 else .065+joint*.025,-.4))
            struct.pack_into('<II63f',data,offset+48,7,3,*[v for p in points for v in p])
    struct.pack_into('<8sQ',data,1272,b'FHVIEW2\0',stamp)
    if present:struct.pack_into('<II',data,1288,7,1)
    return data


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def camera(self,stamp=100):
        metadata=CAMERA.pack(b'slam_left',2,2,stamp,stamp+1)+CAMERA.pack(b'slam_right',2,2,stamp+1,stamp+2)
        return SET.pack(b'FHSET01\0',2,SET.size+len(metadata)+8)+metadata+bytes(8)
    def test_camera_matching_and_missing(self):
        (self.root/'sets.bin').write_bytes(self.camera())
        (self.root/'live-views.txt').write_text('set 100 200 0\nend\n')
        self.assertTrue(inspect_camera(self.root)['all_audited_images_saved'])
        (self.root/'live-views.txt').write_text('set 99 199 0\nend\n')
        self.assertEqual(inspect_camera(self.root)['audited_sets_missing_images'],1)
    def test_camera_truncation_and_audit_truncation(self):
        (self.root/'sets.bin').write_bytes(self.camera()[:-1])
        (self.root/'live-views.txt').write_text('set 100 200 0\nend\n')
        with self.assertRaises(ValueError):inspect_camera(self.root)
        (self.root/'sets.bin').write_bytes(self.camera())
        (self.root/'live-views.txt').write_text('set 100 200 1\n')
        with self.assertRaises(ValueError):inspect_camera(self.root)
    def test_camera_bad_length_and_repeated_time(self):
        raw=bytearray(self.camera());struct.pack_into('<I',raw,12,999)
        (self.root/'sets.bin').write_bytes(raw)
        with self.assertRaises(ValueError):inspect_camera(self.root)
        (self.root/'sets.bin').write_bytes(self.camera()*2)
        with self.assertRaises(ValueError):inspect_camera(self.root)
    def test_fixed_traces(self):
        path=self.root/'trace.bin';path.write_bytes(trace()+trace(1010))
        self.assertEqual(inspect_fixed(path,'trace')['records'],2)
        path.write_bytes(trace()+trace(1010)[:-1])
        with self.assertRaises(ValueError):inspect_fixed(path,'trace')
        path.write_bytes(trace(1010)+trace())
        with self.assertRaises(ValueError):inspect_fixed(path,'trace')
    def test_raw_transport_preserves_clock_mapping(self):
        source_now=1_000_000_000
        raw=bytearray(2656)
        struct.pack_into('<8sIIQQQII',raw,0,b'FHHANDS1',1,2656,2,source_now-10_000_000,source_now-5_000_000,0,0)
        export=b'FHREAD1\0'+struct.pack('<Q',source_now)+raw
        packet,_=convert(export,5000,5008,8)
        data=RELAY.pack(b'FHRELAY1',1,RELAY_SIZE,5000,5008,0,1)+export+packet
        path=self.root/'relay.bin';path.write_bytes(data)
        self.assertEqual(inspect_fixed(path,'relay')['records'],1)
        self.assertEqual(data[RELAY.size:RELAY.size+2672],export)
        self.assertEqual(struct.unpack_from('<Q',data,RELAY.size+2672+32)[0],4994)
    def test_actual_cpp_offline_replay_roundtrip(self):
        exe=Path(__file__).parent/'build/replay-trace.exe'
        self.assertTrue(exe.is_file(),'Build the offline replayer first')
        source=self.root/'input.bin'
        # Tracking -> short loss -> expiry -> reacquisition -> HMD loss.
        frames=[trace(1000,True),trace(1020,True),trace(1040),trace(1130),trace(1160,True),trace(1180,True)]
        struct.pack_into('<I',frames[-1],28,0)
        source.write_bytes(b''.join(frames))
        first=self.root/'first.bin';second=self.root/'second.bin'
        subprocess.run([str(exe),str(source),str(first)],check=True,capture_output=True)
        run=subprocess.run([str(exe),str(first),str(second)],check=True,capture_output=True,text=True)
        self.assertIn('changed_candidate_hand_frames=0',run.stdout)
        self.assertEqual(first.read_bytes(),second.read_bytes())
        data=first.read_bytes()
        flags=[struct.unpack_from('<I',data,i*TRACE_SIZE+1292)[0] for i in range(6)]
        self.assertEqual([bool(f&4) for f in flags],[True,True,True,False,True,False])
        self.assertTrue(flags[2]&8)
        self.assertEqual(inspect_fixed(first,'trace')['records'],6)
        self.assertNotEqual(subprocess.run([str(exe),str(first),str(second)],capture_output=True).returncode,0)
        source.write_bytes(source.read_bytes()[:-1])
        self.assertNotEqual(subprocess.run([str(exe),str(source),str(self.root/'partial.bin')],capture_output=True).returncode,0)
    def test_viewer_records_by_default_and_copies_after_shutdown(self):
        import run_viewer
        root=self.root/'workspace/apps/frame-hands/pc-input'
        root.mkdir(parents=True)
        (root/'build').mkdir();(root/'build/sample-viewer.exe').touch()
        godot=self.root/'workspace/.local/deps/godot-4.7.2/Godot_v4.7.2-stable_win64.exe'
        godot.parent.mkdir(parents=True);godot.touch()
        (root/'deployment.json').write_text(json.dumps(dict(bundle='synthetic-driver')))
        processes=[];commands=[]
        class Process:
            def __init__(self,source=False):self.returncode=None;self.source=source;self.polls=0
            def poll(self):
                self.polls+=1
                if self.source and self.polls>=4:self.returncode=0
                return self.returncode
            def wait(self,timeout=None):self.returncode=0;return 0
            def terminate(self):self.returncode=0
        def launch(command,**kwargs):
            commands.append((command,kwargs));source=any(str(part).endswith('run_session.py') for part in command)
            if source:Path(command[command.index('--source-ready-file')+1]).touch()
            process=Process(source);processes.append(process);return process
        def collect(*args):self.assertTrue(all(p.returncode is not None for p in processes))
        with patch.object(run_viewer,'__file__',str(root/'run_viewer.py')),patch.object(sys,'argv',['run_viewer.py']),\
             patch.object(run_viewer,'prepare') as prepare,patch.object(run_viewer,'collect',side_effect=collect) as copied,\
             patch.object(run_viewer,'verify',return_value={'complete':True}),\
             patch.object(run_viewer.subprocess,'Popen',side_effect=launch),patch.object(run_viewer.time,'sleep'):
            self.assertEqual(run_viewer.main(),0)
        prepare.assert_called_once();copied.assert_called_once()
        self.assertIn('--record-dir',commands[0][0])
        self.assertTrue(commands[1][1]['env']['FRAME_HANDS_TRACE'].endswith('sampler-trace.bin'))


if __name__=='__main__':unittest.main()
