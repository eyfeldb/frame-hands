"""Targeted mode delivery tests: no OpenVR runtime, camera, or live mailboxes."""
import importlib.util,sys,tempfile,types,unittest,struct
from pathlib import Path
from unittest.mock import patch
BASE=Path(__file__).resolve().parent
class Tests(unittest.TestCase):
    def test_selection_without_pose_or_head(self):
        stub=types.ModuleType('rig_pose_subscriber');stub.PoseProcess=object
        with patch.dict(sys.modules,{'rig_pose_subscriber':stub}), patch('os.getuid',return_value=1000,create=True):
            spec=importlib.util.spec_from_file_location('tested_export',BASE/'export_pc.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        with tempfile.TemporaryDirectory() as td:
            m.ROOT=Path(td)
            for mode in (1,2):
                (m.ROOT/'mode.bin').write_bytes(struct.pack('<8sQII',b'FHMODE1\0',42,mode,0))
                result=m.reply(None)
                self.assertEqual(result['state'],'waiting');self.assertEqual(result['mode'],[42,mode])
            (m.ROOT/'mode.bin').write_bytes(b'bad')
            self.assertNotIn('mode',m.reply(None))
            (m.ROOT/'mode.bin').write_bytes(struct.pack('<8sQII',b'FHMODE1\0',42,2,1))
            self.assertIsNone(m.read_mode())
if __name__=='__main__':unittest.main()
