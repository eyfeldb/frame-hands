import json
from pathlib import Path
import tempfile
import unittest
from build_native_tracker import verify_inputs, build_command, sha

class BuildInputTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.bundle=self.root/'bundle';self.replay=self.root/'replay';self.ncnn=self.root/'ncnn'
        for folder in (self.bundle,self.replay,self.ncnn/'lib'):folder.mkdir(parents=True)
        (self.bundle/'source.cpp').write_text('fixture only')
        (self.bundle/'manifest.json').write_text(json.dumps(dict(sha256={'source.cpp':sha(self.bundle/'source.cpp')})))
        (self.ncnn/'lib/libncnn.a').write_bytes(b'fixture only')
        (self.replay/'ft-handreplay-rig').write_bytes(b'fixture only')
        rows=[]
        for index,variant in enumerate(('baseline','candidate','candidate','baseline')):
            name=f'{index}-{variant}';folder=self.replay/name;folder.mkdir()
            for filename in ('poses.txt','views.txt','times.txt'):(folder/filename).write_text('one row\n')
            if variant=='candidate':(folder/'rig.txt').write_text('frame 1 1\nrig 1 0 fitted 1\n')
            rows.append(dict(name=name,variant=variant,processed_frames=1,poses_sha256=sha(folder/'poses.txt'),views_sha256=sha(folder/'views.txt')))
        self.result=dict(scheduling='all saved sets',original_detector_outputs_identical=True,environment=dict(machine='aarch64'),results=rows,provenance=dict(manifest_sha256=sha(self.bundle/'manifest.json'),ncnn_library_sha256=sha(self.ncnn/'lib/libncnn.a'),executable_sha256=sha(self.replay/'ft-handreplay-rig')))
        self.save()
    def save(self):(self.replay/'result.json').write_text(json.dumps(self.result))
    def check(self):return verify_inputs(self.bundle,self.replay,self.ncnn)
    def test_consistent_fixture_and_live_sources(self):
        self.assertEqual(self.check()['fitted_outputs'],[1,1])
        command=build_command(Path('source'),Path('lib'),Path('include'),Path('out'))
        self.assertIn(str(Path('source/track/main.cpp')),command)
        self.assertNotIn(str(Path('source/track/replay.cpp')),command)
    def test_changed_source_library_binary_and_recorded_outputs_rejected(self):
        for path in (self.bundle/'source.cpp',self.ncnn/'lib/libncnn.a',self.replay/'ft-handreplay-rig',self.replay/'1-candidate/poses.txt',self.replay/'3-baseline/views.txt'):
            with self.subTest(path=path):
                original=path.read_bytes();path.write_bytes(b'changed')
                with self.assertRaises(ValueError):self.check()
                path.write_bytes(original)
    def test_whole_pose_build_requires_matching_evidence_and_actual_trace_counts(self):
        from run_tracker_rig_check import verify_head_evidence,recorded_trace_stats
        evidence=self.root/'heads';evidence.mkdir();recording='a'*64
        (evidence/'heads.txt').write_text('FHHEAD 1 CLOCK_MONOTONIC raw_world_mm '+recording+'\n100 90 1 1 0 0 0 0 0 0\n')
        (evidence/'manifest.json').write_text(json.dumps(dict(version=1,recording_sha256=recording,heads_sha256=sha(evidence/'heads.txt'),frames=1,valid_frames=1,clock='CLOCK_MONOTONIC',space='raw_world_mm')))
        self.result['comparison']='fixed_rig_vs_whole_pose'
        self.result['provenance'].update(recording_sha256=recording,head_evidence=verify_head_evidence(evidence,recording))
        for row in self.result['results']:
            path=self.replay/row['name']/'rig.txt';text='frame 100 1\nrig 1 0 fitted 1\n'
            if row['variant']=='candidate':text='head_recorded 100 90 0 0 1 1 1 0 0 0 0 0 0\n'+text+'whole 1 1 0 1 2\n'
            path.write_text(text);row.update(rig_sha256=sha(path),**recorded_trace_stats(path))
        self.save();self.assertEqual(self.check()['fitted_outputs'],[1,1])
        self.result['results'][1]['whole_eligible']=2;self.save()
        with self.assertRaises(ValueError):self.check()
        self.result['results'][1]['whole_eligible']=1;self.save()
        (evidence/'heads.txt').write_text('changed')
        with self.assertRaises(ValueError):self.check()

    def test_host_scheduled_failed_incomplete_and_unexercised_replays_rejected(self):
        for mutation in ('host','scheduled','failed','partial','unfitted','missing_times'):
            with self.subTest(mutation=mutation):
                original=json.loads(json.dumps(self.result))
                if mutation=='host':self.result['environment']['machine']='x86_64'
                if mutation=='scheduled':self.result['scheduling']='measured busy scheduling'
                if mutation=='failed':self.result['original_detector_outputs_identical']=False
                if mutation=='partial':self.result['results']=self.result['results'][:3]
                path=None
                if mutation=='unfitted':path=self.replay/'1-candidate/rig.txt'
                if mutation=='missing_times':path=self.replay/'1-candidate/times.txt'
                if path:old=path.read_bytes();path.write_bytes(b'')
                self.save()
                with self.assertRaises(ValueError):self.check()
                self.result=original;self.save()
                if path:path.write_bytes(old)

class CaptureBuildTests(unittest.TestCase):
    def test_local_capture_support_is_explicit(self):
        from build_native_tracker import build_command
        default=build_command(Path('src'),Path('lib'),Path('include'),Path('out'))
        enabled=build_command(Path('src'),Path('lib'),Path('include'),Path('out'),True)
        self.assertNotIn('-DFRAME_HANDS_WITH_OPENVR',default)
        self.assertNotIn('-ldl',default)
        self.assertIn('-DFRAME_HANDS_WITH_OPENVR',enabled)
        self.assertIn('-ldl',enabled)
        self.assertEqual(default,[x for x in enabled if x not in ('-DFRAME_HANDS_WITH_OPENVR','-ldl')])

if __name__=='__main__':unittest.main()
