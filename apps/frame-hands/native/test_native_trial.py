import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from run_native_trial import BASE, SOURCES, Children, calibration_matches, commands, configuration, run_session, ring_ready, sha

WORKER = """import json,os,sys,time
from pathlib import Path
role,root,mode=sys.argv[1:];root=Path(root)
(root/(role+'.pid')).write_text(str(os.getpid()))
if role=='tracker':
 (root/'recording/sets.bin').write_bytes(b'fixture only')
if role=='subscriber':(root/'subscriber').mkdir()
start=time.monotonic()
while True:
 now=time.monotonic_ns()
 if role=='tracker':
  path=root/'rig.json';value=dict(version=3,capture_mono_ns=now)
 elif role=='subscriber':
  path=root/'subscriber/pose.json';value=dict(version=2,sequence=1,context_valid=True,capture_ns=now,hands=[])
 else:path=None
 if path:
  tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value));os.replace(tmp,path)
 if mode=='exit' and time.monotonic()-start>.15:sys.exit(9 if role=='tracker' else 0)
 time.sleep(.005)
"""

@unittest.skipUnless(sys.platform=='linux','Real process groups require Linux')
class LifecycleTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
  self.plan={name:([sys.executable,'-u','-c',WORKER,name,str(self.root),'normal'],os.environ.copy()) for name in ('camera','tracker','subscriber','viewer')}
 def check_recording(self,path):return dict(structurally_complete=True,all_audited_images_saved=True,scope='synthetic lifecycle fixture, not camera data')
 def run_trial(self,scene=lambda:None,geometry=lambda:None,seconds=.3,recording=None):
  return run_session(self.plan,self.root,seconds,lambda pid,stamp:True,geometry,scene,recording or self.check_recording)
 def assert_stopped(self):
  result=json.loads((self.root/'result.json').read_text())
  for row in result['processes'].values():
   self.assertIsNotNone(row['exit'])
   with self.assertRaises(ProcessLookupError):os.kill(row['pid'],0)
  return result
 def test_duration_stops_all_and_clears_pose(self):
  out=self.run_trial();self.assertEqual(out['reason'],'duration_complete');self.assertTrue(out['recording_ok']);self.assertEqual(len(out['processes']),4)
  self.assertFalse(json.loads((self.root/'subscriber/pose.json').read_text())['context_valid']);self.assert_stopped()
 def test_busy_scene_starts_nothing(self):
  def occupied():raise RuntimeError('occupied scene')
  with self.assertRaisesRegex(RuntimeError,'occupied scene'):self.run_trial(scene=occupied)
  self.assertEqual(self.assert_stopped()['processes'],{})
 def test_geometry_failure_stops_camera_before_tracker(self):
  def invalid():raise RuntimeError('wrong camera ordering')
  with self.assertRaisesRegex(RuntimeError,'wrong camera'):self.run_trial(geometry=invalid)
  self.assertEqual(set(self.assert_stopped()['processes']),{'camera'})
 def test_scene_taken_during_startup_does_not_launch_viewer(self):
  calls=[]
  def scene():
   calls.append(1)
   if len(calls)==2:raise RuntimeError('another app launched')
  with self.assertRaisesRegex(RuntimeError,'another app'):self.run_trial(scene=scene)
  self.assertEqual(set(self.assert_stopped()['processes']),{'camera','tracker','subscriber'})
 def test_tracker_early_exit_cleans_other_roles(self):
  self.plan['tracker'][0][-1]='exit'
  with self.assertRaisesRegex(RuntimeError,'tracker exited early'):self.run_trial(seconds=.5)
  self.assert_stopped()
 def test_viewer_close_is_normal_and_stops_capture(self):
  self.plan['viewer'][0][-1]='exit';out=self.run_trial(seconds=.7)
  self.assertEqual(out['reason'],'viewer_closed');self.assert_stopped()
 def test_incomplete_recording_is_not_reported_success(self):
  with self.assertRaisesRegex(RuntimeError,'recording is incomplete'):
   self.run_trial(recording=lambda _:dict(structurally_complete=True,all_audited_images_saved=False))
  self.assertFalse(self.assert_stopped()['recording_ok'])
 def test_kills_owned_descendant_but_preserves_unrelated_process(self):
  unrelated=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'],start_new_session=True)
  children=Children(self.root)
  code="import subprocess,sys,time; from pathlib import Path; p=subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(20)']); Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(20)"
  try:
   children.start('parent',[sys.executable,'-c',code,str(self.root/'descendant')],os.environ.copy())
   deadline=time.monotonic()+2
   while not (self.root/'descendant').exists() and time.monotonic()<deadline:time.sleep(.01)
   child=int((self.root/'descendant').read_text());time.sleep(.1);children.stop()
   deadline=time.monotonic()+1
   while Path(f'/proc/{child}/stat').exists() and time.monotonic()<deadline:
    if Path(f'/proc/{child}/stat').read_text().split()[2]=='Z':break
    time.sleep(.01)
   self.assertTrue(not Path(f'/proc/{child}/stat').exists() or Path(f'/proc/{child}/stat').read_text().split()[2]=='Z')
   self.assertIsNone(unrelated.poll())
  finally:
   unrelated.terminate();unrelated.wait(timeout=2)

class PlanTests(unittest.TestCase):
 def test_command_contract_and_no_environment_leak(self):
  files={name:Path('/fixture')/name for name in ('camera','godot','pose_service','openvr','xr_runtime')}
  with patch.dict(os.environ,{'FRAME_UME_RIG':'old','FRAME_HANDS_BAD':'old','FRAME_JOB_DEVICE_ROOT':'old'}):
   plan=commands(dict(files=files,build=Path('/build'),viewer=Path('/viewer')),Path('/private'),90)
  self.assertIn('--no-publish',plan['tracker'][0]);self.assertIn('--record',plan['tracker'][0])
  self.assertEqual(plan['tracker'][0][-1],'125')
  self.assertEqual(plan['tracker'][0][plan['tracker'][0].index('--seconds')+1],'125')
  self.assertEqual(plan['tracker'][1]['FT_HANDS_SWAP_SIDES'],'0')
  self.assertNotIn('FRAME_HANDS_BAD',plan['tracker'][1]);self.assertNotIn('FRAME_UME_RIG',plan['viewer'][1])
  self.assertEqual(plan['viewer'][1]['FRAME_UME_VIEW'],'/private/subscriber/pose.json')
  self.assertEqual(plan['subscriber'][0][-1],'105')
 def test_whole_pose_command_requires_explicit_configuration(self):
  files={name:Path('/fixture')/name for name in ('camera','godot','pose_service','openvr','xr_runtime')}
  spec=dict(files=files,build=Path('/build'),viewer=Path('/viewer'),config={'whole_pose':True})
  env=commands(spec,Path('/private'),90)['tracker'][1]
  self.assertEqual(env['FRAME_HANDS_RIG_WHOLE_POSE'],'1')
  self.assertEqual(env['FRAME_HANDS_RIG_OPENVR'],'/fixture/openvr')
  self.assertNotIn('FRAME_HANDS_RIG_REPLAY_HEADS',env)
 def test_calibration_subset_checks_values_and_array_order(self):
  reference=dict(cameras=[dict(sourceCamera='slam_left',fx=1)],axis=[1,2,3])
  actual=dict(cameras=[dict(sourceCamera='other',fx=2),dict(sourceCamera='slam_left',fx=1,extra=0)],axis=[1,2,3],unrelated=1)
  self.assertTrue(calibration_matches(reference,actual))
  actual['axis']=[3,2,1];self.assertFalse(calibration_matches(reference,actual))
  actual['axis']=[1,2,3];actual['cameras'][1]['fx']=2;self.assertFalse(calibration_matches(reference,actual))
 @unittest.skipUnless(sys.platform=='linux','pread readiness on Linux')
 def test_ring_requires_own_fresh_forward_pair(self):
  with tempfile.TemporaryDirectory() as directory:
   path=Path(directory)/'ring';now=time.monotonic_ns();data=bytearray(1344)
   struct.pack_into('<8sIIIIQqQ16x',data,0,b'FHRING01',1,1344,2,0,1344,123,now)
   for i in range(2):
    struct.pack_into('<II',data,64+160*i+72,1056,1024);struct.pack_into('<Q',data,64+160*i+104,1)
   path.write_bytes(data);self.assertTrue(ring_ready(path,123,now-1));self.assertFalse(ring_ready(path,124,now-1))
   struct.pack_into('<Q',data,64+160+104,0);path.write_bytes(data);self.assertFalse(ring_ready(path,123,now-1))

class ConfigurationTests(unittest.TestCase):
 def test_pinned_build_assets_profiles_and_launcher_must_match(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);build=root/'build';source=build/'source';profiles=source/'profiles';viewer=root/'viewer'
   profiles.mkdir(parents=True);viewer.mkdir()
   index=dict(version=2,profiles=[])
   for side in (0,1):
    profile=profiles/f'side{side}.fhrig'
    profile.write_text(f"FHRIG 1 mm left experimental {side} model calibration 20 21 17\n"+"0 0 1 0 0 0 -2 2\n"*20)
    mesh=viewer/f'side{side}.json';mesh.write_text('{}')
    index['profiles'].append(dict(side=side,file=mesh.name,sha256=sha(mesh),profile_sha256=sha(profile)))
   (viewer/'profile-rigs.json').write_text(json.dumps(index))
   (viewer/'manifest.json').write_text(json.dumps(dict(sha256={p.name:sha(p) for p in viewer.iterdir()})))
   (source/'manifest.json').write_text(json.dumps(dict(sha256={str(p.relative_to(source)):sha(p) for p in profiles.iterdir()})))
   binary=build/'ft-hands-native-rig';binary.write_bytes(b'fixture executable')
   (build/'build.json').write_text(json.dumps(dict(binary_sha256=sha(binary),provenance=dict(bundle_manifest_sha256=sha(source/'manifest.json')))))
   config=dict(version=1,build=str(build),viewer=str(viewer),build_sha256=sha(build/'build.json'),viewer_manifest_sha256=sha(viewer/'manifest.json'),source_sha256={name:sha(BASE/name) for name in SOURCES},files={})
   for name in ('camera','godot','pose_service','openvr','scene_check','geometry','xr_runtime','recording_tool'):
    path=root/name;path.write_bytes(b'fixture artifact');config['files'][name]=dict(path=str(path),sha256=sha(path))
   path=root/'config.json';path.write_text(json.dumps(config))
   with patch('run_native_trial.CAMERA_SHA',config['files']['camera']['sha256']):
    self.assertEqual(configuration(path)['build'],build)
    config['whole_pose']=True;path.write_text(json.dumps(config))
    with self.assertRaisesRegex(ValueError,'whole-pose replay'):configuration(path)
    config.pop('whole_pose');path.write_text(json.dumps(config))
    for target in (binary,profiles/'side0.fhrig',viewer/'side1.json',root/'pose_service'):
     with self.subTest(changed=target.name):
      original=target.read_bytes();target.write_bytes(b'changed')
      with self.assertRaises(ValueError):configuration(path)
      target.write_bytes(original)
    original=config['source_sha256']['rig_pose_bridge.py'];config['source_sha256']['rig_pose_bridge.py']='wrong';path.write_text(json.dumps(config))
    with self.assertRaisesRegex(ValueError,'launcher source'):configuration(path)
    config['source_sha256']['rig_pose_bridge.py']=original;path.write_text(json.dumps(config))
   with self.assertRaisesRegex(ValueError,'exact-v3'):configuration(path)

if __name__=='__main__':unittest.main()
