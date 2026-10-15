import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from rig_pose_subscriber import Subscriber, PoseProcess, atomic_json, decode, run

IDENTITY=[1,0,0,0,0,1,0,0,0,0,1,0]
LIMITS=[[(-2,2)]*20]*2

def snapshot(capture=1_000_000_000, sequence=1):
    return dict(version=3,clock='CLOCK_MONOTONIC',space='capture_head_mm',sequence=sequence,
        capture_raw_ns=capture+8_000_000_000,capture_mono_ns=capture,
        published_mono_ns=capture+20_000_000,hands=[dict(source_id=7,side=0,status='fitted',
        wrist_head_mm=[1,0,0,0,100,200,-500],joint_angles=[0.]*20)])

class SubscriberTests(unittest.TestCase):
    def setUp(self):
        self.state=Subscriber(LIMITS);self.now=1_030_000_000;self.queries=[]
    def query(self,capture):
        self.queries.append(capture)
        return dict(valid=True,capture_mono_ns=capture,queried_mono_ns=self.now,
                    capture=IDENTITY,current=IDENTITY)
    def consume(self,value):
        return self.state.consume(json.dumps(value).encode(),self.query,lambda:self.now,lambda:12345.)[0]
    def test_valid_duplicate_and_stall_do_not_renew(self):
        out=self.consume(snapshot());self.assertTrue(out['context_valid'])
        self.assertEqual(out['hands'][0]['wrist_world_m'][2][3],-.5)
        self.now+=100_000_000;self.assertIsNone(self.consume(snapshot()))
        self.assertEqual(self.queries,[1_000_000_000])
        self.assertIsNone(self.state.expire(1_250_000_000,12345.))
        expired=self.state.expire(1_250_000_001,12345.)
        self.assertFalse(expired['context_valid']);self.assertEqual(expired['reason'],'tracker_stalled')
        self.assertGreater(expired['sequence'],out['sequence'])
        self.assertIsNone(self.state.expire(1_300_000_000,12345.))
    def test_stale_and_future_do_not_query(self):
        for now in (1_010_000_000,1_260_000_000):
            self.setUp();self.now=now
            self.assertFalse(self.consume(snapshot())['context_valid']);self.assertEqual(self.queries,[])
    def test_restart_is_latched_until_new_session(self):
        self.consume(snapshot(sequence=5));self.now+=40_000_000
        out=self.consume(snapshot(capture=1_040_000_000,sequence=1))
        self.assertFalse(out['context_valid']);self.assertTrue(self.state.restart_required)
        self.assertIsNone(self.consume(snapshot(capture=1_050_000_000,sequence=6)))
        self.assertEqual(len(self.queries),1)
    def test_malformed_geometry_invalidates_and_next_frame_recovers(self):
        self.consume(snapshot());bad=snapshot(capture=1_005_000_000,sequence=2)
        bad['hands'][0]['wrist_head_mm'][0]=3
        self.assertFalse(self.consume(bad)['context_valid'])
        self.assertTrue(self.consume(snapshot(capture=1_006_000_000,sequence=3))['context_valid'])
    def test_monocular_or_empty_retains_explicit_unavailability(self):
        value=snapshot();value['hands'][0]['status']='monocular'
        out=self.consume(value);self.assertTrue(out['context_valid']);self.assertEqual(out['hands'],[])
        self.assertEqual(out['unavailable_sides'][0]['reason'],'monocular')
    def test_decoder_and_atomic_replacement(self):
        for value in (b'[]',b'{"x":NaN}',b'{"x":Infinity}',b'x'*65537):
            with self.assertRaises(ValueError):decode(value)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'pose.json';atomic_json(path,dict(sequence=1));atomic_json(path,dict(sequence=2))
            self.assertEqual(decode(path.read_bytes())['sequence'],2)
            self.assertFalse(path.with_name('pose.json.tmp').exists())

HELPER = """import sys,json,time
for line in sys.stdin:
 if line.strip()=='Q':break
 stamp=int(line);mode=sys.argv[1]
 if stamp==0:
  print(json.dumps(dict(valid=False,reason='invalid_or_stale_capture_mono_ns')),flush=True);continue
 if mode=='timeout':time.sleep(10)
 if mode=='eof':break
 if mode=='wrong':stamp+=1
 print(json.dumps(dict(valid=True,capture_mono_ns=stamp,queried_mono_ns=time.monotonic_ns(),capture=[1,0,0,0,0,1,0,0,0,0,1,0],current=[1,0,0,0,0,1,0,0,0,0,1,0])),flush=True)
"""
class PipeTests(unittest.TestCase):
    def helper(self,mode):
        return PoseProcess([sys.executable,'-u','-c',HELPER,mode],subprocess.DEVNULL,timeout=.5)
    def test_actual_pipe_multiple_queries(self):
        helper=self.helper('ok')
        try:
            for stamp in (123,456):self.assertEqual(helper.query(stamp)['capture_mono_ns'],stamp)
        finally:helper.close()
        self.assertIsNotNone(helper.process.poll())
    def test_timeout_eof_and_wrong_stamp_poison_transport(self):
        for mode in ('timeout','eof','wrong'):
            with self.subTest(mode=mode):
                helper=self.helper(mode)
                try:
                    with self.assertRaises(RuntimeError):helper.query(123)
                    with self.assertRaises(RuntimeError):helper.query(456)
                finally:helper.close()
                self.assertIsNotNone(helper.process.poll())
    def test_helper_failure_clears_output_and_records_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'snapshot.json';out=root/'out';out.mkdir()
            helper=self.helper('wrong')
            atomic_json(source,snapshot(time.monotonic_ns()-30_000_000))
            with self.assertRaises(RuntimeError):run(source,out,Subscriber(LIMITS),helper,.2)
            rows=[json.loads(line) for line in (out/'trace.jsonl').read_text().splitlines()]
            failed=[row for row in rows if row['packet'].get('reason')=='pose_helper_failed']
            self.assertEqual(len(failed),1);self.assertIsNotNone(failed[0]['snapshot_utf8'])
            self.assertFalse(decode((out/'pose.json').read_bytes())['context_valid'])
            self.assertIsNotNone(helper.process.poll())
    def test_real_file_pipe_publication_lifecycle(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'snapshot.json';out=root/'out';out.mkdir()
            helper=self.helper('ok')
            # Ensure startup is complete before giving the fixture a fresh timestamp.
            helper.query(time.monotonic_ns())
            capture=time.monotonic_ns()-30_000_000
            atomic_json(source,snapshot(capture))
            run(source,out,Subscriber(LIMITS),helper,.35)
            rows=[json.loads(line) for line in (out/'trace.jsonl').read_text().splitlines()]
            self.assertEqual(rows[0]['packet']['reason'],'starting')
            valid=[row for row in rows if row['packet']['context_valid']]
            self.assertEqual(len(valid),1)
            self.assertEqual(valid[0]['head']['capture_mono_ns'],capture)
            self.assertEqual(json.loads(valid[0]['snapshot_utf8'])['capture_raw_ns'],capture+8_000_000_000)
            self.assertIn('tracker_stalled',[row['packet'].get('reason') for row in rows])
            self.assertEqual(rows[-1]['packet']['reason'],'duration_complete')
            self.assertEqual(decode((out/'pose.json').read_bytes()),rows[-1]['packet'])
            self.assertEqual([row['packet']['sequence'] for row in rows],list(range(1,len(rows)+1)))
            self.assertIsNotNone(helper.process.poll())

if __name__=='__main__':unittest.main()
