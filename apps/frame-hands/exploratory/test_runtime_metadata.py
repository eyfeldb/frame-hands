import tempfile
from pathlib import Path
import unittest
from runtime_metadata import installed_revision

class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.git=self.root/'.git';self.git.mkdir();self.sha='a'*40
    def tearDown(self):self.tmp.cleanup()
    def test_detached_without_git_executable(self):
        (self.git/'HEAD').write_text(self.sha)
        self.assertEqual(installed_revision(self.root),self.sha)
    def test_loose_ref_without_git_executable(self):
        (self.git/'HEAD').write_text('ref: refs/heads/main')
        (self.git/'refs/heads').mkdir(parents=True)
        (self.git/'refs/heads/main').write_text(self.sha)
        self.assertEqual(installed_revision(self.root),self.sha)
    def test_packed_ref(self):
        (self.git/'HEAD').write_text('ref: refs/heads/main')
        (self.git/'packed-refs').write_text(self.sha+' refs/heads/main\n')
        self.assertEqual(installed_revision(self.root),self.sha)
    def test_missing_metadata_never_blocks_tracking(self):
        self.assertEqual(installed_revision(self.root),'unavailable')

if __name__=='__main__':unittest.main()
