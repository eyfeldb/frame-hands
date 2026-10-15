import hashlib,json,tempfile,unittest
from pathlib import Path
from run_tracker_rig_check import verify_head_evidence,recorded_trace_stats
class EvidenceTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.addCleanup(self.tmp.cleanup);self.recording='a'*64
  self.text='FHHEAD 1 CLOCK_MONOTONIC raw_world_mm '+self.recording+'\n100 90 1 1 0 0 0 0 0 0\n';self.path=self.root/'heads.txt';self.path.write_text(self.text)
  self.meta=dict(version=1,recording_sha256=self.recording,heads_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),frames=1,valid_frames=1,clock='CLOCK_MONOTONIC',space='raw_world_mm');self.save()
 def save(self):(self.root/'manifest.json').write_text(json.dumps(self.meta))
 def test_matching(self):self.assertEqual(verify_head_evidence(self.root,self.recording)['recording_sha256'],self.recording)
 def test_recording_change_and_payload_change(self):
  with self.assertRaises(ValueError):verify_head_evidence(self.root,'b'*64)
  self.path.write_text(self.text+'\n')
  with self.assertRaises(ValueError):verify_head_evidence(self.root,self.recording)
 def test_wrong_clock_empty_unbounded_and_header(self):
  for field,value in [('clock','CLOCK_MONOTONIC_RAW'),('valid_frames',0),('frames',100001)]:
   old=self.meta[field];self.meta[field]=value;self.save()
   with self.assertRaises(ValueError):verify_head_evidence(self.root,self.recording)
   self.meta[field]=old
  self.path.write_text(self.text.replace('raw_world_mm','raw_world_m'));self.meta['heads_sha256']=hashlib.sha256(self.path.read_bytes()).hexdigest();self.save()
  with self.assertRaises(ValueError):verify_head_evidence(self.root,self.recording)
 def test_trace_counts_and_forbidden_runtime_queries(self):
  trace=self.root/'rig.txt';text='head_recorded 100 90 0 0 1 1 1 0 0 0 0 0 0\nwhole 1 1 1 2 1\n'
  trace.write_text(text);self.assertEqual(recorded_trace_stats(trace),dict(recorded_head_frames=1,valid_head_frames=1,whole_eligible=1,whole_selected=1))
  for altered in [text.replace('head_recorded','head'),text.replace('90 0 0','90 95 96'),text.replace('whole 1 1 1','whole 1 0 1')]:
   trace.write_text(altered)
   with self.assertRaises(ValueError):recorded_trace_stats(trace)
if __name__=='__main__':unittest.main()
