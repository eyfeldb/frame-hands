"""Offline integration: never writes the live driver mailbox or starts cameras."""
import json,os,subprocess,sys,tempfile,time,unittest,struct
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from wire import pack
from native_service import service_plan,require_awake,require_camera_available,require_camera_permissions
from unittest.mock import patch,MagicMock
from native.run_native_trial import configuration
class Checks(unittest.TestCase):
    def test_continuous_no_recording(self):
        spec=configuration(BASE/'accepted-config.json');plan=service_plan(spec,Path('/tmp/not-started'))
        self.assertEqual(set(plan),{'camera','tracker'})
        argv,env=plan['tracker'];self.assertEqual(argv[argv.index('--seconds')+1],'0')
        self.assertFalse(any(x.startswith('--record') for x in argv))
        self.assertNotIn('FRAME_HANDS_AUDIT',env);self.assertEqual(env['FRAME_HANDS_RIG_TRACE'],'/dev/null')
        self.assertEqual(Path(argv[0]).name,'tracker-no-record')
        self.assertEqual(env['FRAME_HANDS_RIG_WHOLE_POSE'],'1')
    def test_real_trace_startup(self):
        spec=configuration(BASE/'accepted-config.json')
        with tempfile.TemporaryDirectory(prefix='fh-trace-startup-') as td:
            output=Path(td);_,env=service_plan(spec,output)['tracker']
            existing=output/'existing.txt';existing.write_text('preserve me')
            for sink,code,reason in [('/dev/null',0,'initialized'),(None,1,'requires a trace path'),(str(existing),1,'existing trace')]:
                configured=dict(env)
                if sink is None:configured.pop('FRAME_HANDS_RIG_TRACE')
                else:configured['FRAME_HANDS_RIG_TRACE']=sink
                result=subprocess.run([str(BASE/'test-trace-startup')],env=configured,capture_output=True,text=True,timeout=15)
                self.assertEqual(result.returncode,code,result.stderr)
                self.assertIn(reason,result.stdout+result.stderr)
            self.assertEqual(existing.read_text(),'preserve me')
    def test_wire_expiry(self):
        v=dict(source_ms=1000,wall_s=1.0,hands=[])
        self.assertEqual(len(pack(v,1,1000,1.0)),592)
        self.assertIsNone(pack(v,1,1080,1.081))
        self.assertIsNone(pack(v,1,1501,1.0))
        self.assertIsNone(pack(v,1,999,1.0))
    def test_standby_before_camera(self):
        for valid in (False,True):
            with tempfile.TemporaryDirectory() as td, patch('native_service.PoseProcess') as factory:
                helper=factory.return_value;helper.query.return_value={'valid':valid}
                if valid:require_awake({'files':{'openvr':'/unused'}},Path(td))
                else:
                    with self.assertRaisesRegex(RuntimeError,'wake'):require_awake({'files':{'openvr':'/unused'}},Path(td))
                helper.close.assert_called_once()
    def test_streaming_allowed_but_eyes_blocked(self):
        with patch('native_service.process_names',return_value=[(20,'vrlink'),(21,'run_vrlink.sh')]):require_camera_available()
        with patch('native_service.process_names',return_value=[(20,'frame-eyes-read')]):
            with self.assertRaisesRegex(RuntimeError,'Eye'):require_camera_available()
    def test_camera_permissions(self):
        for caps in ('', 'cap_sys_ptrace=ep', 'cap_dac_read_search,cap_sys_ptrace,cap_perfmon=p', 'cap_dac_read_search,cap_sys_ptrace,cap_perfmon=ep'):
            with patch('native_service.subprocess.run') as run:
                run.return_value.stdout='/reader '+caps if caps else ''
                if caps.endswith('perfmon=ep'):require_camera_permissions(Path('/reader'))
                else:
                    with self.assertRaisesRegex(RuntimeError,'permissions are missing'):require_camera_permissions(Path('/reader'))

    def test_clock(self):
        before=time.monotonic_ns()//1000000
        tick=int(subprocess.check_output([str(BASE/'test-model'),'--clock'],text=True))
        self.assertLessEqual(before,tick);self.assertLessEqual(tick,time.monotonic_ns()//1000000)

def main():
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():raise SystemExit(1)
    with tempfile.TemporaryDirectory(prefix='native-hands-offline-') as td:
        output=Path(td);config=json.loads((BASE/'accepted-config.json').read_text());godot=config['files']['godot']['path']
        subprocess.run([godot,'--headless','--xr-mode','off','--path',str(BASE/'processor'),'--script','res://test_fk.gd','--',str(output/'joints.json')],check=True,timeout=30)
        packets=json.loads((output/'joints.json').read_text())
        (output/'packets.bin').write_bytes(b''.join(pack(v,i+1,v['source_ms'],1.0) for i,v in enumerate(packets)))
        subprocess.run([str(BASE/'test-model'),str(output/'packets.bin')],check=True,timeout=10)
        # Feed actual accepted native poses to the full background bridge. Redirect
        # its mailbox to this temporary directory: nothing reaches vrserver.
        import native_service as service
        service.RUNTIME=output
        spec=configuration(BASE/'accepted-config.json')
        poses=json.loads((BASE/'pose-fixture.json').read_text())
        service.atomic_json(output/'pose.json',dict(sequence=0,context_valid=False))
        renderer=service.Renderer(spec,output)
        try:
            time.sleep(.35)
            for i,original in enumerate(poses):
                packet=dict(original,sequence=i+1,capture_ns=time.monotonic_ns()-20000000,published_wall_s=time.time(),age_ms=20.)
                service.atomic_json(output/'pose.json',packet)
                time.sleep(1/30)
            deadline=time.monotonic()+.08
            while renderer.hands==0 and time.monotonic()<deadline:time.sleep(.005)
            if renderer.error:raise renderer.error
            if not renderer.hands:raise RuntimeError('Recorded native poses never produced driver hands')
            print('Recorded-pose bridge:',renderer.count,'publications,',renderer.hands,'hand outputs')
            # Source expiry cannot be renewed by render ticks.
            before=(output/'input.bin').read_bytes();source=struct.unpack_from('<Q',before,32)[0]
            time.sleep(.23)
            after=(output/'input.bin').read_bytes()
            if struct.unpack_from('<Q',after,32)[0]!=source:raise RuntimeError('Render renewed capture context')
            if struct.unpack_from('<I',after,44)[0] or struct.unpack_from('<I',after,320)[0]:raise RuntimeError('Expired hands still visible')
            packet=dict(poses[-1],sequence=len(poses)+1,capture_ns=time.monotonic_ns()-20000000,published_wall_s=time.time(),age_ms=20.)
            service.atomic_json(output/'pose.json',packet);time.sleep(.06)
            data=(output/'input.bin').read_bytes()
            if not (struct.unpack_from('<I',data,44)[0] or struct.unpack_from('<I',data,320)[0]):raise RuntimeError('Reacquisition failed')
            service.atomic_json(output/'pose.json',dict(sequence=len(poses)+2,context_valid=False));time.sleep(.04)
            data=(output/'input.bin').read_bytes()
            if struct.unpack_from('<I',data,44)[0] or struct.unpack_from('<I',data,320)[0]:raise RuntimeError('Invalid head context did not clear')
        finally:renderer.close()
        if (output/'input.bin').exists():raise RuntimeError('Stop left a live mailbox')
    print('PASS native service startup plan, clock, fixed rig, actual driver reconstruction and headless processor')
if __name__=='__main__':main()
