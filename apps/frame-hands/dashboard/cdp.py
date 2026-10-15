"""Dependency-free, loopback-only CDP WebSocket client for Frame's Steam UI."""
import asyncio
import base64
import hashlib
import json
import os
import struct
from urllib.parse import urlsplit


class CDP:
    def __init__(self, reader, writer, on_event):
        self.reader, self.writer, self.on_event = reader, writer, on_event
        self.sequence = 0
        self.pending = {}
        self.write_lock = asyncio.Lock()
        self.task = asyncio.create_task(self.receive())

    @classmethod
    async def connect(cls, url, on_event):
        u = urlsplit(url)
        if u.scheme != 'ws' or u.hostname != '127.0.0.1' or u.port != 8080 or not u.path.startswith('/devtools/page/'):
            raise ValueError('Unexpected dashboard debugger endpoint')
        reader, writer = await asyncio.wait_for(asyncio.open_connection(u.hostname, u.port), 5)
        key = base64.b64encode(os.urandom(16)).decode()
        writer.write((f'GET {u.path} HTTP/1.1\r\nHost: 127.0.0.1:8080\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n').encode())
        await writer.drain()
        try:
            header = (await asyncio.wait_for(reader.readuntil(b'\r\n\r\n'), 5)).decode('ascii')
            fields = dict(line.split(': ', 1) for line in header.split('\r\n')[1:] if ': ' in line)
            fields = {k.lower(): v.strip() for k, v in fields.items()}
            expected = base64.b64encode(hashlib.sha1((key + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11').encode()).digest()).decode()
            if not header.startswith('HTTP/1.1 101 ') or fields.get('sec-websocket-accept') != expected:
                raise RuntimeError('WebSocket handshake rejected')
        except BaseException:
            writer.close()
            raise
        return cls(reader, writer, on_event)

    async def send_frame(self, data, opcode=1):
        mask = os.urandom(4)
        size = len(data)
        header = bytes([0x80 | opcode])
        header += bytes([0x80 | size]) if size < 126 else bytes([0x80 | 126]) + struct.pack('!H', size) if size < 65536 else bytes([0x80 | 127]) + struct.pack('!Q', size)
        payload = bytes(value ^ mask[i % 4] for i, value in enumerate(data))
        async with self.write_lock:
            self.writer.write(header + mask + payload)
            await self.writer.drain()

    async def receive(self):
        parts = bytearray()
        try:
            while True:
                first, second = await self.reader.readexactly(2)
                final, opcode, size = first & 0x80, first & 15, second & 127
                if size == 126:
                    size = struct.unpack('!H', await self.reader.readexactly(2))[0]
                elif size == 127:
                    size = struct.unpack('!Q', await self.reader.readexactly(8))[0]
                if size > 8 * 1024 * 1024 or len(parts) + size > 8 * 1024 * 1024:
                    raise RuntimeError('Oversized dashboard message')
                if second & 0x80:
                    raise RuntimeError('Unexpected masked server frame')
                data = await self.reader.readexactly(size)
                if opcode == 8:
                    raise ConnectionError('Dashboard closed')
                if opcode == 9:
                    await self.send_frame(data, 10)
                    continue
                if opcode == 10:
                    continue
                if opcode not in (0, 1):
                    raise RuntimeError('Unexpected WebSocket opcode')
                parts.extend(data)
                if not final:
                    continue
                message = json.loads(parts)
                parts.clear()
                future = self.pending.pop(message.get('id'), None)
                if future and not future.done():
                    if 'error' in message:
                        future.set_exception(RuntimeError(str(message['error'])))
                    else:
                        future.set_result(message.get('result', {}))
                elif 'method' in message:
                    self.on_event(message)
        except (Exception, asyncio.CancelledError) as error:
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(ConnectionError(str(error)))
            self.pending.clear()

    async def call(self, method, params=None):
        if self.task.done():
            raise ConnectionError('Dashboard disconnected')
        self.sequence += 1
        ident = self.sequence
        future = asyncio.get_running_loop().create_future()
        self.pending[ident] = future
        try:
            await self.send_frame(json.dumps({'id': ident, 'method': method, 'params': params or {}}).encode())
            return await asyncio.wait_for(future, 5)
        finally:
            self.pending.pop(ident, None)

    async def evaluate(self, expression):
        result = await self.call('Runtime.evaluate', {'expression': expression, 'returnByValue': True})
        if result.get('exceptionDetails'):
            details = result['exceptionDetails']
            raise RuntimeError(details.get('exception', {}).get('description') or details.get('text', 'Dashboard script failed'))
        return result.get('result', {}).get('value')

    async def close(self):
        self.task.cancel()
        self.writer.close()
        await asyncio.gather(self.task, return_exceptions=True)
