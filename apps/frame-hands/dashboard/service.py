"""On-Frame native dashboard toggle. Never starts hands except on an explicit click."""
import asyncio
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import struct
import time
import urllib.request
from urllib.parse import parse_qs, urlsplit

from cdp import CDP

BASE = Path(__file__).resolve().parent
RUNTIME = Path('/run/user') / str(os.getuid()) / 'frame-hands-native'
UNIT = 'frame-hands-native.service'
MODE = struct.Struct('<8sQII')
EXPECTED = {
    'sp.js': '59c97d98b0b3eaa28e2e677209738a6d075ecaefc13f3f6e13d7371332163956',
    'css/sp.css': '77d3a89e4a2cb27eacf463fa1e5319b78e200cbb9f16dc16d565760cf6dd295f',
}


def log(event, **values):
    print(json.dumps(dict(event=event, **values)), flush=True)


def read_mode(root):
    try:
        tag, epoch, mode, reserved = MODE.unpack((root / 'mode.bin').read_bytes())
        if tag == b'FHMODE1\0' and reserved == 0 and epoch > 0 and mode in (1, 2):
            return epoch, mode
    except (OSError, struct.error):
        pass
    return 0, 2


def write_mode(root, enabled):
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    epoch = max(time.monotonic_ns(), read_mode(root)[0] + 1)
    temporary = root / 'mode.dashboard.tmp'
    temporary.write_bytes(MODE.pack(b'FHMODE1\0', epoch, 1 if enabled else 2, 0))
    os.replace(temporary, root / 'mode.bin')
    return epoch


async def systemctl(*arguments):
    process = await asyncio.create_subprocess_exec('systemctl', '--user', *arguments, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(process.communicate(), 8)
    except BaseException:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        await process.wait()
        raise
    if process.returncode:
        raise RuntimeError(err.decode().strip() or 'Hands service command failed')
    return out.decode().strip()


class HandsControl:
    def __init__(self, root=RUNTIME, command=systemctl):
        self.root, self.command = root, command
        self.lock = asyncio.Lock()
        self.starting = False
        self.stopping = False
        self.error = ''
        self.last_enabled = None

    async def request(self, enabled):
        if type(enabled) is not bool:
            raise ValueError('Toggle requires a boolean')
        async with self.lock:
            self.error = ''
            write_mode(self.root, enabled)
            try:
                await self.command('start' if enabled else 'stop', '--no-block', UNIT)
                self.starting, self.stopping = enabled, not enabled
                log('toggle', enabled=enabled)
            except Exception as error:
                write_mode(self.root, False)
                self.starting = self.stopping = False
                self.error = str(error)[:200]
                log('toggle_failed', error=self.error)

    async def poll(self):
        async with self.lock:
            service = await self.command('show', UNIT, '--property=ActiveState', '--value')
            _, mode = read_mode(self.root)
            running = service in ('active', 'activating', 'reloading')
            if running and (service == 'active' or self.starting) and mode == 2 and not self.stopping:
                # Existing controller-watch owns the physical press/release gate.
                await self.command('stop', '--no-block', UNIT)
                self.stopping, self.starting = True, False
                log('controller_takeover_stop')
            if service == 'active':
                self.starting = False
            elif service in ('inactive', 'failed'):
                if self.starting or (service == 'failed' and not self.stopping):
                    try:
                        reason = json.loads((self.root / 'status.json').read_text()).get('error')
                    except (OSError, ValueError):
                        reason = None
                    self.error = (reason or 'Could not start. Wake Frame and retry.')[:200]
                    if mode == 1:
                        write_mode(self.root, False)
                self.starting = self.stopping = False
            enabled = service == 'active' and mode == 1 and not self.stopping
            if enabled:
                self.error = ''
            if enabled != self.last_enabled:
                log('state', enabled=enabled, service=service, mode=mode)
                self.last_enabled = enabled
            return dict(enabled=enabled, busy=self.starting or self.stopping or service in ('activating', 'deactivating'), ready=True, error=self.error)


def validate_build():
    steamui = Path.home() / '.local/share/Steam/steamui'
    if (steamui / 'changelist.txt').read_text().strip() != '11094443':
        raise RuntimeError('Steam UI updated; dashboard adapter needs revalidation')
    for name, digest in EXPECTED.items():
        if hashlib.sha256((steamui / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Steam UI changed: ' + name)


def targets():
    # No proxy or remote endpoint; this is the already-enabled local Steam debugger.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open('http://127.0.0.1:8080/json/list', timeout=3) as response:
        values = json.load(response)
    return [item for item in values if item.get('type') == 'page'
            and 'valve.steam.gamepadui.bar (' in item.get('title', '')
            and (urlsplit(item.get('url', '')).hostname == 'steamloopback.host'
                 or parse_qs(urlsplit(item.get('url', '')).query).get('vrOverlayKey') == ['valve.steam.gamepadui.bar'])]


async def run():
    RUNTIME.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock = (RUNTIME / 'dashboard.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    control = HandsControl()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    ui = (BASE / 'ui.js').read_text()
    cdp = None
    requests = asyncio.Queue(maxsize=4)
    current_context = None
    last_validation = 0
    last_error = ''

    def on_event(message):
        if message['method'] == 'Runtime.bindingCalled':
            params = message.get('params', {})
            if params.get('name') != 'frameHandsRequest' or params.get('executionContextId') != current_context:
                return
            try:
                value = json.loads(params['payload'])
                if set(value) != {'enabled'} or type(value['enabled']) is not bool:
                    return
                requests.put_nowait(value['enabled'])
            except (KeyError, TypeError, ValueError, asyncio.QueueFull):
                pass

    try:
        while not stop.is_set():
            try:
                while not requests.empty():
                    await control.request(requests.get_nowait())
                state = await control.poll()
                if time.monotonic() - last_validation > 10:
                    await asyncio.to_thread(validate_build)
                    last_validation = time.monotonic()
                if not cdp:
                    available = await asyncio.to_thread(targets)
                    if available:
                        cdp = await CDP.connect(available[0]['webSocketDebuggerUrl'], on_event)
                        if not await cdp.evaluate("location.hostname === 'steamloopback.host' && !!window.opener?.webpackChunksteamui && !!document.querySelector('#VRDashboardBarContainer')"):
                            raise RuntimeError('Native dashboard origin/context not ready')
                        # AddBinding applies only to this exact trusted page; its calling
                        # default execution context is checked again for each request.
                        await cdp.call('Runtime.enable')
                        contexts = []
                        previous_event = cdp.on_event
                        def capture(message):
                            if message['method'] == 'Runtime.executionContextCreated':
                                contexts.append(message['params']['context'])
                            previous_event(message)
                        cdp.on_event = capture
                        await cdp.call('Runtime.disable')
                        await cdp.call('Runtime.enable')
                        current_context = next(c['id'] for c in contexts if c.get('auxData', {}).get('isDefault'))
                        cdp.on_event = on_event
                        await cdp.call('Runtime.addBinding', {'name': 'frameHandsRequest', 'executionContextId': current_context})
                        await cdp.evaluate(ui)
                        log('dashboard_attached', target=available[0]['id'])
                if cdp:
                    # Idempotently remount after a native bar rebuild, then publish actual state.
                    result = await cdp.evaluate('(() => {const a=window.__frameHandsDashboard; if(!a)return null; a.ensureMounted(); a.update(' + json.dumps(state) + '); return a.info();})()')
                    if result is None or not result.get('mounted'):
                        raise ConnectionError('Dashboard context changed')
                last_error = ''
            except Exception as error:
                if str(error) != last_error:
                    log('adapter_waiting', error=str(error)[:300])
                    last_error = str(error)
                if cdp:
                    with contextlib.suppress(Exception):
                        await cdp.evaluate('window.__frameHandsDashboard?.dispose()')
                    await cdp.close()
                    cdp = None
                    current_context = None
            try:
                await asyncio.wait_for(stop.wait(), .5 if cdp else 2)
            except asyncio.TimeoutError:
                pass
    finally:
        if cdp:
            with contextlib.suppress(Exception):
                await cdp.evaluate('window.__frameHandsDashboard?.dispose()')
                await cdp.call('Runtime.removeBinding', {'name': 'frameHandsRequest'})
            await cdp.close()
        lock.close()


if __name__ == '__main__':
    asyncio.run(run())
