import unittest
from workflow_guard import conflicts
class GuardTests(unittest.TestCase):
 def test_direct_eye_reader_is_conflict_without_service(self):
  self.assertEqual(conflicts([(14,'fe-eyegrab'),(20,'python3')],20),[(14,'fe-eyegrab')])
 def test_eye_collector_is_conflict_without_wrapper(self):
  self.assertEqual(conflicts([(15,'fe-collector')],20),[(15,'fe-collector')])
 def test_native_capture_and_tracker_are_conflicts(self):
  self.assertEqual(len(conflicts([(1,'frame-camera-re'),(2,'ft-hands-ume-ro'),(3,'vrserver')],20)),2)
 def test_original_camera_and_ring_replay_are_conflicts(self):
  self.assertEqual(conflicts([(30,'ft-camd'),(31,'ft-ringplay'),(32,'ft-handreplay')],20),[(30,'ft-camd'),(31,'ft-ringplay')])
 def test_own_pid_and_unrelated_runtime_not_conflicts(self):
  self.assertEqual(conflicts([(20,'ft-hands'),(4,'vrcompositor'),(5,'python3')],20),[])
# The kernel-lock tests run on Linux; Windows still runs the process-name controls.
import os, subprocess, sys, tempfile
from pathlib import Path
from unittest.mock import patch
from workflow_guard import native_workflow

@unittest.skipUnless(sys.platform == 'linux', 'Requires Linux flock')
class KernelLockTests(unittest.TestCase):
 def test_each_shared_or_legacy_lock_blocks_and_releases_prior_locks(self):
  import fcntl, select
  for relative in ('FrameEyes/workflow.lock','FrameHands/workflow.lock','FrameHands/trial.lock'):
   with self.subTest(lock=relative), tempfile.TemporaryDirectory() as directory:
    root=Path(directory)
    for name in ('FrameEyes','FrameHands'):(root/name).mkdir()
    peer=subprocess.Popen([sys.executable,'-u','-c',"import fcntl,sys; f=open(sys.argv[1],'a'); fcntl.flock(f,fcntl.LOCK_EX); print('locked',flush=True); sys.stdin.read()",str(root/relative)],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    try:
     if not select.select([peer.stdout],[],[],3)[0]:self.fail('Peer did not acquire lock')
     self.assertEqual(peer.stdout.readline(),b'locked\n')
     with patch('workflow_guard.subprocess.run') as service, patch('workflow_guard.require_no_optical_processes') as optical:
      with self.assertRaisesRegex(RuntimeError,relative):
       with native_workflow(root):self.fail('Entered occupied workflow')
      service.assert_not_called();optical.assert_not_called()
     for other in ('FrameEyes/workflow.lock','FrameHands/workflow.lock','FrameHands/trial.lock'):
      if other==relative:continue
      with (root/other).open('a') as handle:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    finally:
     peer.stdin.close();peer.wait(timeout=3);peer.stdout.close()
    with patch('workflow_guard.subprocess.run',return_value=subprocess.CompletedProcess([],3)), patch('workflow_guard.require_no_optical_processes'):
     with native_workflow(root):pass
 def test_service_and_process_failure_release_every_lock(self):
  import fcntl
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory)
   for name in ('FrameEyes','FrameHands'):(root/name).mkdir()
   for active in (True,False):
    with patch('workflow_guard.subprocess.run',return_value=subprocess.CompletedProcess([],0 if active else 3)), patch('workflow_guard.require_no_optical_processes',side_effect=RuntimeError('Optical process active')):
     with self.assertRaises(RuntimeError):
      with native_workflow(root):self.fail('Entered conflicting workflow')
    for relative in ('FrameEyes/workflow.lock','FrameHands/workflow.lock','FrameHands/trial.lock'):
     with (root/relative).open('a') as handle:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)

if __name__=='__main__':unittest.main()
