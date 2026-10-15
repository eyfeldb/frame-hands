import copy,math,unittest
from rig_pose_bridge import RigPoseBridge,head_matrix

class BridgeTests(unittest.TestCase):
 def setUp(self):
  self.bridge=RigPoseBridge([[(-2.,2.)]*20]*2)
  self.snapshot=dict(version=3,clock='CLOCK_MONOTONIC',space='capture_head_mm',sequence=1,capture_raw_ns=9000000000,capture_mono_ns=1000000000,published_mono_ns=1020000000,hands=[dict(source_id=7,side=0,status='fitted',wrist_head_mm=[1,0,0,0,100,200,-500],joint_angles=[0.]*20)])
  # Capture head is90deg aboutZ at(1,2,3); current head has moved independently.
  self.head=dict(valid=True,capture_mono_ns=1000000000,queried_mono_ns=1020000000,capture=[0,-1,0,1,1,0,0,2,0,0,1,3],current=[1,0,0,4,0,1,0,5,0,0,1,6])
 def call(self,s=None,h=None,now=1030000000):return self.bridge.convert(s or self.snapshot,h or self.head,now,12345.)
 def test_capture_transform_units_and_current_mapping(self):
  out=self.call();self.assertTrue(out['context_valid']);m=out['hands'][0]['wrist_world_m'];self.assertAlmostEqual(m[0][3],.8);self.assertAlmostEqual(m[1][3],2.1);self.assertAlmostEqual(m[2][3],2.5)
  self.assertEqual([r[3] for r in out['head_world_m'][:3]],[4.,5.,6.]);self.assertEqual(out['capture_ns'],1000000000);self.assertEqual(out['age_ms'],30.)
 def test_raw_clock_cannot_be_used_for_head_query(self):
  self.head['capture_mono_ns']=self.snapshot['capture_raw_ns'];self.assertFalse(self.call()['context_valid'])
 def test_invalid_context_stale_and_future_clear(self):
  for modification,now in [('valid',1030000000),('old',1300000000),('future',1010000000),('query_old',1080000000)]:
   with self.subTest(modification=modification):
    self.setUp()
    if modification=='valid':self.head['valid']=False
    out=self.call(now=now);self.assertFalse(out['context_valid']);self.assertEqual(out['hands'],[])
 def test_duplicates_do_not_renew_capture(self):
  self.assertIsNotNone(self.call());self.assertIsNone(self.call(now=1040000000));self.snapshot['sequence']=2;self.assertIsNone(self.call(now=1040000000))
 def test_source_conflict_preserves_other_side(self):
  other=copy.deepcopy(self.snapshot['hands'][0]);other.update(source_id=8,side=1);self.snapshot['hands'].append(other)
  conflict=copy.deepcopy(self.snapshot['hands'][0]);conflict['source_id']=9;self.snapshot['hands'].append(conflict)
  out=self.call();self.assertEqual([h['side'] for h in out['hands']],[1]);self.assertEqual(out['unavailable_sides'],[dict(side=0,reason='ambiguous_side')])
 def test_nonfit_and_empty_do_not_invent_pose(self):
  for status in ('monocular','missing_or_ambiguous_views','insufficient'):
   self.setUp();self.snapshot['hands'][0]['status']=status;out=self.call();self.assertTrue(out['context_valid']);self.assertEqual(out['hands'],[])
  self.setUp();self.snapshot['hands']=[];self.assertEqual(self.call()['hands'],[])
 def test_bad_geometry_and_contract_rejected(self):
  for modification in ('nan','quaternion','angle','source','clock','reflection'):
   with self.subTest(modification=modification):
    self.setUp()
    if modification=='nan':self.snapshot['hands'][0]['wrist_head_mm'][4]=float('nan')
    if modification=='quaternion':self.snapshot['hands'][0]['wrist_head_mm'][0]=2
    if modification=='angle':self.snapshot['hands'][0]['joint_angles'][0]=3
    if modification=='source':self.snapshot['hands'].append(copy.deepcopy(self.snapshot['hands'][0]))
    if modification=='clock':self.snapshot['clock']='CLOCK_MONOTONIC_RAW'
    if modification=='reflection':self.head['current'][0]=-1
    with self.assertRaises(ValueError):self.call()
if __name__=='__main__':unittest.main()
