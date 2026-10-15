import unittest
from unittest.mock import patch
from litert_benchmark_core import require_native
class NativeAbiTests(unittest.TestCase):
    def check(self, machine, libc, accepted):
        with patch('litert_benchmark_core.sys.platform','linux'), patch('litert_benchmark_core.sys.version_info',(3,12,3)), patch('litert_benchmark_core.platform.machine',return_value=machine), patch('litert_benchmark_core.platform.libc_ver',return_value=libc):
            if accepted: require_native()
            else:
                with self.assertRaises(RuntimeError): require_native()
    def test_wrong_cpu(self): self.check('x86_64',('glibc','2.39'),False)
    def test_package_newer_than_wheel_tag(self): self.check('aarch64',('glibc','2.27'),False)
    def test_other_libc(self): self.check('aarch64',('musl','2.39'),False)
    def test_audited_minimum(self): self.check('aarch64',('glibc','2.34'),True)
    def test_recorded_frame_version(self): self.check('aarch64',('glibc','2.39'),True)
if __name__=='__main__': unittest.main()
