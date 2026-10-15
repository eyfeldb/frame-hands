"""Control-boundary tests: no camera, runtime, or systemd changes."""
import asyncio
import json
from pathlib import Path
import struct
import tempfile
import unittest
from service import HandsControl, MODE, UNIT, read_mode, write_mode


class FakeSystemd:
    def __init__(self, state='inactive'):
        self.state, self.calls, self.failure = state, [], None

    async def __call__(self, *args):
        self.calls.append(args)
        if args[0] == 'show':
            return self.state
        if self.failure:
            raise RuntimeError(self.failure)
        self.state = 'activating' if args[0] == 'start' else 'deactivating'
        return ''


class Lifecycle(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.backend = FakeSystemd()
        self.control = HandsControl(self.root, self.backend)

    def tearDown(self):
        self.temp.cleanup()

    async def test_idle_never_starts(self):
        self.assertFalse((await self.control.poll())['enabled'])
        self.assertTrue(all(call[0] == 'show' for call in self.backend.calls))

    async def test_start_requires_service_acknowledgement(self):
        await self.control.request(True)
        self.assertEqual(read_mode(self.root)[1], 1)
        self.assertEqual(self.backend.calls[-1], ('start', '--no-block', UNIT))
        state = await self.control.poll()
        self.assertTrue(state['busy'])
        self.assertFalse(state['enabled'])
        self.backend.state = 'active'
        state = await self.control.poll()
        self.assertTrue(state['enabled'])
        self.assertFalse(state['busy'])

    async def test_standby_and_missing_hands_preserve_selection(self):
        self.backend.state = 'active'
        write_mode(self.root, True)
        for status in ('paused', 'waiting_for_hands', 'tracking'):
            (self.root / 'status.json').write_text(json.dumps({'state': status}))
            self.assertTrue((await self.control.poll())['enabled'])
        self.assertFalse(any(call[0] == 'stop' for call in self.backend.calls))

    async def test_controller_takeover_stops_capture_once(self):
        self.backend.state = 'active'
        write_mode(self.root, True)
        self.assertTrue((await self.control.poll())['enabled'])
        write_mode(self.root, False)
        state = await self.control.poll()
        self.assertFalse(state['enabled'])
        self.assertTrue(state['busy'])
        await self.control.poll()
        self.assertEqual(sum(call[0] == 'stop' for call in self.backend.calls), 1)
        self.backend.state = 'inactive'
        self.assertFalse((await self.control.poll())['busy'])

    async def test_click_off_selects_controllers_before_stop(self):
        self.backend.state = 'active'
        write_mode(self.root, True)
        await self.control.request(False)
        self.assertEqual(read_mode(self.root)[1], 2)
        self.assertEqual(self.backend.calls[-1], ('stop', '--no-block', UNIT))

    async def test_failed_start_clears_mode_and_surfaces_reason(self):
        await self.control.request(True)
        self.backend.state = 'failed'
        (self.root / 'status.json').write_text(json.dumps({'error': 'Camera alignment failed'}))
        state = await self.control.poll()
        self.assertFalse(state['enabled'])
        self.assertFalse(state['busy'])
        self.assertEqual(state['error'], 'Camera alignment failed')
        self.assertEqual(read_mode(self.root)[1], 2)

    async def test_command_failure_does_not_claim_on(self):
        self.backend.failure = 'Unit unavailable'
        await self.control.request(True)
        self.assertEqual(read_mode(self.root)[1], 2)
        self.assertEqual(self.control.error, 'Unit unavailable')

    async def test_cancel_start_last_request_wins(self):
        await self.control.request(True)
        await self.control.request(False)
        self.assertEqual(read_mode(self.root)[1], 2)
        self.backend.state = 'inactive'
        self.assertFalse((await self.control.poll())['enabled'])

    async def test_takeover_during_start_cancels_start(self):
        await self.control.request(True)
        write_mode(self.root, False)
        state = await self.control.poll()
        self.assertFalse(state['enabled'])
        self.assertFalse(self.control.starting)
        self.assertEqual(self.backend.calls[-1][0], 'stop')

    async def test_existing_launcher_may_select_hands_after_startup(self):
        # The existing Library launcher writes Hands only after its readiness gate.
        write_mode(self.root, False)
        self.backend.state = 'activating'
        self.assertFalse((await self.control.poll())['enabled'])
        self.assertFalse(any(call[0] == 'stop' for call in self.backend.calls))
        write_mode(self.root, True)
        self.backend.state = 'active'
        self.assertTrue((await self.control.poll())['enabled'])

    async def test_external_success_clears_previous_error(self):
        self.control.error = 'Headset asleep'
        self.backend.state = 'active'
        write_mode(self.root, True)
        state = await self.control.poll()
        self.assertTrue(state['enabled'])
        self.assertEqual(state['error'], '')

    async def test_reject_non_boolean_command(self):
        for value in ('true', 1, None, {}):
            with self.assertRaises(ValueError):
                await self.control.request(value)
        self.assertEqual(self.backend.calls, [])

    async def test_corrupt_mode_defaults_to_controllers(self):
        for value in (b'', b'invalid', MODE.pack(b'FHMODE1\0', 1, 7, 0), MODE.pack(b'FHMODE1\0', 1, 1, 1)):
            (self.root / 'mode.bin').write_bytes(value)
            self.assertEqual(read_mode(self.root), (0, 2))


if __name__ == '__main__':
    unittest.main()
