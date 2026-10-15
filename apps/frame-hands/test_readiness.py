"""Linux integration checks against real temporary files, not camera hardware."""
import os
from pathlib import Path
import struct
import tempfile
import time
import unittest
from readiness import ring_ready, hands_ready, RING_HEADER, RING_BYTES
from test_measure import packet


@unittest.skipUnless(hasattr(os,'pread'),'Linux positioned reads required')
class TrialReadiness(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path=Path(self.directory.name)/'ring'
        self.started=time.monotonic_ns()-1000000

    def ring(self,pid=123,heartbeat=None,published=2):
        data=bytearray(RING_BYTES)
        RING_HEADER.pack_into(data,0,b'FHRING01',1,RING_BYTES,4,0,RING_BYTES,pid,
                              heartbeat if heartbeat is not None else time.monotonic_ns())
        for n in range(published):struct.pack_into('<Q',data,64+160*n+104,1)
        self.path.write_bytes(data)

    def test_current_publisher_and_two_streams(self):
        self.ring();self.assertTrue(ring_ready(self.path,123,self.started))

    def test_upper_pair_cannot_satisfy_forward_startup(self):
        self.ring()
        self.assertFalse(ring_ready(self.path,123,self.started,require_forward=True))
        data=bytearray(self.path.read_bytes())
        for n in range(2):struct.pack_into('<II',data,64+160*n+72,1056,1024)
        self.path.write_bytes(data)
        self.assertTrue(ring_ready(self.path,123,self.started,require_forward=True))

    def test_old_ring_or_other_writer_is_not_ready(self):
        self.ring(pid=456);self.assertFalse(ring_ready(self.path,123,self.started))
        self.ring(heartbeat=self.started-1);self.assertFalse(ring_ready(self.path,123,self.started))

    def test_allocated_ring_with_no_frames_is_not_ready(self):
        for n in [0,1]:
            self.ring(published=n);self.assertFalse(ring_ready(self.path,123,self.started))

    def test_wrong_layout_and_symlink_are_not_ready(self):
        self.path.write_bytes(b'bad');self.assertFalse(ring_ready(self.path,123,self.started))
        self.ring();link=self.path.with_name('link');link.symlink_to(self.path)
        self.assertFalse(ring_ready(link,123,self.started))

    def test_previous_hand_observations_are_not_ready(self):
        self.path.write_bytes(packet(capture=self.started-100,publish=self.started-1))
        self.assertFalse(hands_ready(self.path,self.started))

    def test_fresh_empty_observation_is_transport_ready(self):
        now=time.monotonic_ns()
        self.path.write_bytes(packet(capture=now,publish=now,count=0))
        self.assertTrue(hands_ready(self.path,self.started))


if __name__=='__main__':unittest.main()
