import threading
import unittest
from pathlib import Path
from ready_panel import ManualGate, trial_command, require_pc_vr

class ManualGateTests(unittest.TestCase):
    def test_idle_has_no_effects(self):
        calls=[]
        gate=ManualGate(lambda:calls.append('check'),lambda:calls.append('launch'))
        self.assertEqual(calls,[])
        self.assertFalse(gate.busy)

    def test_failed_check_never_launches_or_retries(self):
        calls=[]
        def fail(): calls.append('check'); raise RuntimeError('busy')
        gate=ManualGate(fail,lambda:calls.append('launch'))
        with self.assertRaisesRegex(RuntimeError,'busy'):gate.start_clicked()
        self.assertEqual(calls,['check'])
        self.assertFalse(gate.busy)

    def test_duplicate_click_does_not_launch_twice(self):
        entered=threading.Event();release=threading.Event();calls=[]
        def launch(): calls.append('launch');entered.set();release.wait(2)
        gate=ManualGate(lambda:calls.append('check'),launch)
        worker=threading.Thread(target=gate.start_clicked);worker.start()
        self.assertTrue(entered.wait(1))
        self.assertIsNone(gate.start_clicked())
        release.set();worker.join(2)
        self.assertEqual(calls,['check','launch'])
        self.assertFalse(gate.busy)

    def test_exact_pc_candidate(self):
        command=trial_command(Path('C:/workspace with spaces'))
        for flag,value in [('--continuity','native'),('--identity','image-cuda'),('--camera-reader','exact-v3'),('--presentation','predict'),('--seconds','90')]:
            self.assertEqual(command[command.index(flag)+1],value)
        self.assertIn('--ready',command)
        self.assertEqual(command[command.index('--workspace')+1],str(Path('C:/workspace with spaces')))

    def test_piano_is_explicit_bridge_with_owned_stop_file(self):
        command=trial_command(Path('C:/workspace'),Path('C:/Piano/Frame Piano.exe'),Path('C:/owned.stop'))
        self.assertEqual(command[command.index('--continuity')+1],'bridge')
        self.assertIn('--continuous',command)
        self.assertEqual(command[command.index('--stop-file')+1],'C:\\owned.stop')
        self.assertIn('--piano',command)

class NativeCommandTests(unittest.TestCase):
    def test_native_mode_never_requests_pc_piano_or_a_recorded_trial(self):
        command=trial_command(Path('C:/workspace'),stop_file=Path('C:/stop'),native=True)
        self.assertIn('--native-piano',command);self.assertIn('--continuous',command)
        self.assertNotIn('--piano',command)
        self.assertEqual(command[command.index('--continuity')+1],'bridge')

class SteamCommandTests(unittest.TestCase):
    def test_steamvr_uses_continuous_model_bridge_without_a_scene(self):
        command=trial_command(Path('C:/workspace'),stop_file=Path('C:/stop'),steamvr=Path('C:/deployment.json'))
        self.assertIn('--steamvr',command);self.assertIn('--continuous',command)
        self.assertNotIn('--native-piano',command);self.assertNotIn('--piano',command)
        self.assertEqual(command[command.index('--continuity')+1],'bridge')

class ReadinessTests(unittest.TestCase):
    def test_live_pose_passes_without_process_name_heuristic(self):
        require_pc_vr(dict(connected=True,frame_hmd=True,pose_valid=True))
    def test_connected_but_untracked_gets_wake_guidance(self):
        with self.assertRaisesRegex(RuntimeError,'head tracking is not ready'):
            require_pc_vr(dict(connected=True,frame_hmd=True,pose_valid=False))
    def test_disconnected_has_distinct_message(self):
        with self.assertRaisesRegex(RuntimeError,'headset disconnected'):
            require_pc_vr(dict(connected=False,frame_hmd=True,pose_valid=False))
    def test_other_headset_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'different headset'):
            require_pc_vr(dict(connected=True,frame_hmd=False,pose_valid=True))
    def test_runtime_failure_is_not_called_stream_disconnection(self):
        with self.assertRaisesRegex(RuntimeError,'SteamVR is not ready'):
            require_pc_vr(dict(runtime_error=108))

if __name__=='__main__':unittest.main()
