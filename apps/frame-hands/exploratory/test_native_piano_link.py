import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from native_piano_link import receive_packet, expired

class ReceiverTests(unittest.TestCase):
    def setUp(self):
        self.now=2_000_000_000
        self.head=dict(valid=True,queried_mono_ns=self.now-1_000_000,current=[1,0,0,.25,0,1,0,1.5,0,0,1,-.3])
        self.packet=dict(version=2,sequence=1,capture_ns=self.now-60_000_000,published_wall_s=-9999,age_ms=-999,context_valid=True,hands=[dict(side=0,source_id=7,joint_angles=[0.0]*22,wrist_world_m=[[1,0,0,.1],[0,1,0,1],[0,0,1,-.4],[0,0,0,1]])])
    def result(self):return receive_packet(self.packet,self.head,self.now,12345,0)
    def test_age_and_wall_time_are_local_to_frame(self):
        result=self.result();self.assertEqual(result['age_ms'],60);self.assertEqual(result['published_wall_s'],12345)
        self.assertEqual(result['capture_ns'],self.packet['capture_ns'])
    def test_current_native_head_and_world_hands_keep_meter_units(self):
        result=self.result();self.assertEqual(result['head_world_m'][1][3],1.5)
        self.assertEqual(result['hands'],self.packet['hands'])
    def test_reordered_capture_rejected(self):
        self.assertIsNone(receive_packet(self.packet,self.head,self.now,12345,self.packet['capture_ns']))
    def test_stale_capture_rejected(self):
        self.packet['capture_ns']=self.now-250_000_000;self.assertIsNone(self.result())
    def test_future_capture_rejected(self):
        self.packet['capture_ns']=self.now+1;self.assertIsNone(self.result())
    def test_invalid_or_stale_head_clears_feed(self):
        self.head['valid']=False;self.assertIsNone(self.result())
        self.head['valid']=True;self.head['queried_mono_ns']=self.now-51_000_000;self.assertIsNone(self.result())
    def test_duplicate_sides_rejected(self):
        second=copy.deepcopy(self.packet['hands'][0]);second['source_id']=8
        self.packet['hands'].append(second);self.assertIsNone(self.result())
    def test_no_synthetic_hands_on_shutdown(self):
        packet=expired(self.now);self.assertFalse(packet['context_valid']);self.assertEqual(packet['hands'],[])
    def test_model_packet_keeps_all_22_angles(self):
        self.packet['hands'][0]['joint_angles']=list(range(22))
        self.assertEqual(self.result()['hands'][0]['joint_angles'],list(range(22)))
    def test_wrong_angle_count_rejected(self):
        for size in (0,19,20,21,23):
            self.packet['hands'][0]['joint_angles']=[0.0]*size
            self.assertIsNone(self.result())
    def test_nonfinite_angles_rejected(self):
        self.packet['hands'][0]['joint_angles'][0]=float('nan');self.assertIsNone(self.result())

@unittest.skipIf(os.name=='nt','Frame receiver uses POSIX file locks')
class ReceiverPipeTests(unittest.TestCase):
    def test_hand_heartbeat_resume_and_eof(self):
        # Isolated fake head context and temporary feed: never publish test hands
        # to Piano or open cameras/XR. Exercise the actual receiver pipe loop.
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'pose.json'
            code="""import time
from pathlib import Path
import native_piano_link as link
class Head:
 def get(self,capture):return dict(valid=True,queried_mono_ns=time.monotonic_ns(),current=[1,0,0,0,0,1,0,0,0,0,1,0])
 def close(self):pass
link.NativePose=Head
link.FEED=Path(%r)
link.main()
""" % str(path)
            child=subprocess.Popen([sys.executable,'-u','-c',code],cwd=Path(__file__).parent,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            def send(packet):
                child.stdin.write(json.dumps(packet)+'\n');child.stdin.flush()
                return json.loads(child.stdout.readline())
            try:
                for index in range(2):
                    packet=dict(version=2,capture_ns=time.monotonic_ns()-10_000_000,context_valid=True,hands=[dict(side=0,source_id=1,joint_angles=[0.0]*22,wrist_world_m=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])])
                    self.assertTrue(send(packet)['accepted'])
                    self.assertTrue(json.loads(path.read_text())['context_valid'])
                    self.assertTrue(send({'kind':'heartbeat'})['waiting'])
                    self.assertFalse(json.loads(path.read_text())['context_valid'])
                    self.assertIsNone(child.poll())
                child.stdin.close();child.wait(timeout=3)
                self.assertEqual(child.returncode,0,child.stderr.read())
                self.assertEqual(json.loads(path.read_text())['hands'],[])
            finally:
                if child.poll() is None:child.kill();child.wait()
                child.stdout.close();child.stderr.close()

if __name__=='__main__':unittest.main()
