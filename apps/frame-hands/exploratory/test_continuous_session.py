"""Linux source lifecycle and sparse-spool tests. No camera or XR access."""
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from continuous_session import SparseSpool, prepare

ORIGINAL = Path(os.environ.get('FRAME_SOURCE_FIXTURE', '/nonexistent'))

class SpoolTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform=='linux','Linux fallocate')
    def test_releases_consumed_blocks_preserves_offset_tail_and_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'sets.bin'
            with p.open('wb') as writer:
                writer.write(b'A'*(8*1024*1024)+b'B'*(4*1024*1024));writer.flush()
                spool=SparseSpool();spool.release(p,8*1024*1024+127)
                self.assertEqual(p.stat().st_size,12*1024*1024)
                self.assertLess(p.stat().st_blocks*512,6*1024*1024)
                with p.open('rb') as reader:
                    reader.seek(8*1024*1024);self.assertEqual(reader.read(),b'B'*(4*1024*1024))
                writer.write(b'C');writer.flush()
                self.assertEqual(p.stat().st_size,12*1024*1024+1)
                with self.assertRaises(ValueError):spool.release(p,24*1024*1024)

    @unittest.skipUnless(sys.platform=='linux','Linux fallocate')
    def test_live_tail_keeps_latest_frames_without_restarting(self):
        from export_live import Tail
        from frame_data import SET,CAM
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);tail=Tail(root,continuous=True)
            with (root/'sets.bin').open('wb') as stream,(root/'live-views.txt').open('w') as audit:
                for stamp in range(1,161):
                    pixels=bytes([stamp%256])*(512*512)
                    stream.write(SET.pack(b'FHSET01\0',2,SET.size+2*CAM.size+2*len(pixels)))
                    for name in (b'slam_left',b'slam_right'):stream.write(CAM.pack(name,512,512,stamp,stamp))
                    stream.write(pixels*2);stream.flush()
                    audit.write(f'set {stamp} {stamp} 0\nend\n');audit.flush()
                    tail.update()
                self.assertEqual(len(tail.records),120)
                self.assertEqual(max(tail.records),160)
                offset,rows=tail.records[160]
                with (root/'sets.bin').open('rb') as reader:
                    reader.seek(offset);self.assertEqual(reader.read(2*len(pixels)),pixels*2)
                self.assertLess((root/'sets.bin').stat().st_blocks*512,70*1024*1024)

@unittest.skipUnless(ORIGINAL.is_dir() and sys.platform=='linux','Installed launcher fixture on Linux')
class SourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        prepare(ORIGINAL,self.root/'launchers')
        source=self.root/'launchers/pc-input/run_source.py'
        # Keep lifecycle tests independent of real workflow locks.
        source.write_text(source.read_text().replace('/home/steamos/devkit-game/FrameEyes/workflow.lock',str(self.root/'absent-eye.lock')))
        spec=importlib.util.spec_from_file_location('test_owned_source',source)
        self.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)
        self.child=types.SimpleNamespace(returncode=None)
        self.child.poll=lambda:self.child.returncode
        self.child.terminate=lambda:setattr(self.child,'returncode',0)
        self.child.wait=lambda **kw:self.child.returncode
        self.now=0.0
    def tearDown(self):self.tmp.cleanup()
    def run_source(self, continuous, steps, commands):
        def select(*args):
            self.now+=next(steps)
            return ([True],[],[])
        argv=['source','--seconds','5']+(['--continuous'] if continuous else [])
        with patch.object(sys,'argv',argv),patch.object(sys,'stdin',types.SimpleNamespace(buffer=io.BytesIO(commands))),patch.object(self.module.subprocess,'Popen',return_value=self.child) as spawn,patch.object(self.module.time,'monotonic',lambda:self.now),patch.object(self.module.select,'select',select),patch.object(self.module.signal,'signal'):
            result=self.module.main()
            return result,spawn.call_args.args[0]
    def test_heartbeat_keeps_source_beyond_former_five_minute_limit(self):
        result,command=self.run_source(True,iter([10]*361),b'H'*360+b'Q')
        self.assertEqual(result,0);self.assertGreater(self.now,3600)
        self.assertIn('--continuous',command)
    def test_eof_stops_owned_worker(self):
        self.assertEqual(self.run_source(True,iter([1]),b'')[0],0)
        self.assertEqual(self.child.returncode,0)
    def test_missing_heartbeat_expires_lease_and_cleans_up(self):
        with self.assertRaisesRegex(TimeoutError,'lease expired'):
            self.run_source(True,iter([46]),b'x')
        self.assertEqual(self.child.returncode,0)
    def test_bounded_diagnostic_deadline_is_preserved(self):
        with self.assertRaisesRegex(TimeoutError,'startup/session timeout'):
            self.run_source(False,iter([100]),b'H')
        self.assertEqual(self.child.returncode,0)
    def test_transform_preserves_pinned_binary_and_geometry_checks(self):
        original=(ORIGINAL/'trial.py').read_text()
        changed=(self.root/'launchers/trial.py').read_text()
        import re
        self.assertEqual(re.findall(r"[0-9a-f]{64}",original),re.findall(r"[0-9a-f]{64}",changed))
        self.assertIn('ring_ready(RUNTIME',changed)
        self.assertIn('Camera geometry did not confirm',changed)
        self.assertIn('fcntl.LOCK_EX | fcntl.LOCK_NB',changed)
        self.assertIn("str(0 if args.continuous else args.seconds+8)",changed)
        self.assertIn("('inf' if args.continuous else str(args.seconds+5))",changed)

if __name__=='__main__':unittest.main()
