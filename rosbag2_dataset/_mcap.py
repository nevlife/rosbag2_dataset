# Copyright 2026 nevlife
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Read MCAP records and message indexes directly from a memory mapping."""

from collections import OrderedDict
from dataclasses import dataclass
import mmap
import struct

import numpy as np

MAGIC = b'\x89MCAP0\r\n'
_U16 = struct.Struct('<H')
_U32 = struct.Struct('<I')
_U64 = struct.Struct('<Q')
_RECORD = struct.Struct('<BQ')
_MESSAGE = struct.Struct('<HIQ8x')
_READ_BUFFER = 1 << 20
_INDEX_ENTRY = np.dtype([('time', '<u8'), ('offset', '<u8')])


def _string(buffer, pos):
    size = _U32.unpack_from(buffer, pos)[0]
    pos += 4
    end = pos + size
    return bytes(buffer[pos:end]).decode('utf-8'), end


def records(buffer, start=0, end=None):
    """Yield complete record headers, tolerating an unfinished final record."""
    end = len(buffer) if end is None else end
    pos = start
    while pos + 9 <= end:
        opcode, size = _RECORD.unpack_from(buffer, pos)
        stop = pos + 9 + size
        if stop > end:
            break
        yield opcode, pos, pos + 9, stop
        pos = stop


def file_layout(path):
    """Describe how an MCAP file was written from its footer, summary and first records."""
    with path.open('rb') as stream:
        buffer = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
    try:
        if buffer[:8] != MAGIC:
            raise ValueError(f'Not an MCAP file: {path}')
        return _layout(buffer)
    except (struct.error, OverflowError) as error:
        raise ValueError(f'Corrupt MCAP file: {path}') from error
    finally:
        buffer.close()


def _layout(buffer):
    summary_start = summary_end = 0
    footer = len(buffer) - 37
    if footer >= 8 and buffer[-8:] == MAGIC and buffer[footer] == 2:
        summary_start, summary_offsets = struct.unpack_from('<QQ', buffer, footer + 9)
        summary_end = summary_offsets or footer
    for opcode, _, pos, _ in records(buffer, summary_start, summary_end):
        if opcode == 8:
            map_size = _U32.unpack_from(buffer, pos + 32)[0]
            compression = _string(buffer, pos + 44 + map_size)[0]
            return f'chunked {compression}' if compression else 'chunked'
    # Without chunk indexes the first data record tells chunked from unchunked.
    for opcode, _, pos, _ in records(buffer, 8, summary_start or len(buffer)):
        if opcode == 5:
            return 'unchunked (no index, full scan needed)'
        if opcode == 6:
            return 'chunked, no index (full scan needed)'
    return 'no messages'


@dataclass(frozen=True)
class Schema:
    """Describe an embedded serialization schema."""

    name: str
    encoding: str
    text: str


@dataclass(frozen=True)
class Channel:
    """Associate a topic with a schema, message encoding, and channel metadata."""

    schema_id: int
    topic: str
    encoding: str
    metadata: dict


def definition(buffer, opcode, pos, schemas, channels):
    """Add the Schema (opcode 3) or Channel (opcode 4) record at pos to its mapping."""
    if opcode == 3:
        ident = _U16.unpack_from(buffer, pos)[0]
        name, pos = _string(buffer, pos + 2)
        encoding, pos = _string(buffer, pos)
        length = _U32.unpack_from(buffer, pos)[0]
        start, end = pos + 4, pos + 4 + length
        text = bytes(buffer[start:end]).decode('utf-8')
        schemas[ident] = Schema(name, encoding, text)
    elif opcode == 4:
        ident, schema_id = struct.unpack_from('<HH', buffer, pos)
        topic, pos = _string(buffer, pos + 4)
        encoding, pos = _string(buffer, pos)
        end = pos + 4 + _U32.unpack_from(buffer, pos)[0]
        pos += 4
        metadata = {}
        while pos < end:
            key, pos = _string(buffer, pos)
            metadata[key], pos = _string(buffer, pos)
        channels[ident] = Channel(schema_id, topic, encoding, metadata)


def decompress(compression, data, uncompressed, where):
    """Return the uncompressed records of an lz4 or zstd chunk; where names it in errors."""
    if compression == 'lz4':
        try:
            import lz4.frame
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "MCAP lz4 chunks require the missing module 'lz4' (pip install lz4)"
            ) from error
        errors = RuntimeError

        def run():
            if lz4.frame.get_frame_info(data)['content_size'] not in (0, uncompressed):
                raise RuntimeError('lz4 frame and chunk sizes differ')
            return lz4.frame.decompress(data)
    elif compression == 'zstd':
        try:
            import zstandard
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "MCAP zstd chunks require the missing module 'zstandard' "
                '(pip install zstandard)'
            ) from error
        errors = zstandard.ZstdError

        def run():
            if zstandard.frame_content_size(data) not in (-1, uncompressed):
                raise zstandard.ZstdError('zstd frame and chunk sizes differ')
            return zstandard.ZstdDecompressor().decompress(data, max_output_size=uncompressed)
    else:
        raise ValueError(f'Unsupported MCAP chunk compression: {compression!r}')
    # Both decompressors allocate the declared size up front; zstd expands 2^15 times at most.
    try:
        decoded = run() if uncompressed <= len(data) << 15 else b''
    except errors as error:
        raise ValueError(f'Corrupt MCAP {where}') from error
    if len(decoded) != uncompressed:
        raise ValueError(f'Corrupt MCAP {where}')
    return decoded


class McapFile:
    """Own a mapping, channel definitions, and a bounded decompression cache."""

    def __init__(self, path, chunk_cache_size=4, *, index=True):
        self.path = path
        self.cache_size = chunk_cache_size
        self.cache = OrderedDict()
        self.schemas = {}
        self.channels = {}
        self.entries = []
        with path.open('rb') as stream:
            self.mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            if self.mapping[:8] != MAGIC:
                raise ValueError(f'Not an MCAP file: {path}')
            if index:
                self._index()
        except (struct.error, OverflowError) as error:
            self.close()
            raise ValueError(f'Corrupt MCAP file: {path}') from error
        except BaseException:
            self.close()
            raise

    def _definition(self, buffer, opcode, pos):
        definition(buffer, opcode, pos, self.schemas, self.channels)

    def _message(self, buffer, record, pos, chunk):
        channel = _U16.unpack_from(buffer, pos)[0]
        time = _U64.unpack_from(buffer, pos + 6)[0]
        self.entries.append((channel, time, chunk, record))

    def _scan_chunk(self, offset):
        buffer = self.chunk(offset)
        for opcode, record, pos, _ in records(buffer):
            if opcode in (3, 4):
                self._definition(buffer, opcode, pos)
            elif opcode == 5:
                self._message(buffer, record, pos, offset)

    def _index(self):
        buffer = self.mapping
        summary_start = 0
        footer = len(buffer) - 37
        if footer >= 8 and buffer[-8:] == MAGIC and buffer[footer] == 2:
            summary_start, summary_offsets = struct.unpack_from('<QQ', buffer, footer + 9)
        indexed_chunks = set()
        # Message index entries occupy disjoint 16-byte ranges of the file.
        pairs_left = len(buffer) // 16
        if summary_start:
            end = summary_offsets or footer
            for opcode, _, pos, _ in records(buffer, summary_start, end):
                if opcode in (3, 4):
                    self._definition(buffer, opcode, pos)
                elif opcode == 8:
                    chunk = _U64.unpack_from(buffer, pos + 16)[0]
                    map_size = _U32.unpack_from(buffer, pos + 32)[0]
                    if not map_size:
                        continue
                    indexed_chunks.add(chunk)
                    for entry in range(pos + 36, pos + 36 + map_size, 10):
                        channel, index_offset = struct.unpack_from('<HQ', buffer, entry)
                        size = _U32.unpack_from(buffer, index_offset + 11)[0]
                        pairs = np.frombuffer(buffer, _INDEX_ENTRY, size // 16, index_offset + 15)
                        pairs_left -= len(pairs)
                        if pairs_left < 0:
                            raise ValueError(f'Corrupt MCAP file: {self.path} reuses index bytes')
                        self.entries.extend(
                            (channel, int(pair['time']), chunk, int(pair['offset']))
                            for pair in pairs
                        )
        self.used_summary = bool(indexed_chunks)
        # Top-level headers also locate unchunked messages and unindexed chunks.
        for opcode, record, pos, _ in records(buffer, 8, summary_start or len(buffer)):
            if opcode in (3, 4):
                self._definition(buffer, opcode, pos)
            elif opcode == 5:
                self._message(buffer, record, pos, -1)
            elif opcode == 6 and record not in indexed_chunks:
                self._scan_chunk(record)
            elif opcode == 15:
                break

    def chunk(self, offset):
        """Return a view of a chunk, decompressing at most once while cached."""
        if offset in self.cache:
            self.cache.move_to_end(offset)
            return memoryview(self.cache[offset])
        uncompressed = _U64.unpack_from(self.mapping, offset + 9 + 16)[0]
        compression, pos = _string(self.mapping, offset + 9 + 28)
        size = _U64.unpack_from(self.mapping, pos)[0]
        start, end = pos + 8, pos + 8 + size
        data = memoryview(self.mapping)[start:end]
        if not compression:
            return data
        decoded = decompress(compression, data, uncompressed, f'chunk at {offset} in {self.path}')
        self.cache[offset] = decoded
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return memoryview(decoded)

    def raw(self, chunk, offset):
        """Return the CDR bytes of an indexed message without copying."""
        try:
            buffer = memoryview(self.mapping) if chunk == -1 else self.chunk(chunk)
            size = _U64.unpack_from(buffer, offset + 1)[0]
        except (struct.error, OverflowError) as error:
            raise ValueError(f'Corrupt MCAP record at {offset} in {self.path}') from error
        start, end = offset + 31, offset + 9 + size
        return buffer[start:end]

    def close(self):
        """Release owned buffers; exported views keep their mapping alive."""
        self.cache.clear()
        mapping, self.mapping = self.mapping, None
        if mapping is not None:
            try:
                mapping.close()
            except BufferError:
                # Exported NumPy arrays own the mapping until their last view dies.
                pass


def summary(path):
    """
    Return (schemas, channels, statistics) from an MCAP file's summary section.

    statistics is (first log time, last log time, {channel: message count}) or None.
    """
    with path.open('rb') as stream:
        size = stream.seek(0, 2)
        footer = stream.seek(max(size - 37, 0))
        tail = stream.read(37)
        if size < 8 + 37 or tail[-8:] != MAGIC or tail[0] != 2:
            return {}, {}, None
        start, offsets = struct.unpack_from('<QQ', tail, 9)
        stream.seek(start)
        buffer = stream.read((offsets or footer) - start) if start else b''
    schemas, channels, statistics = {}, {}, None
    try:
        for opcode, _, pos, _ in records(buffer):
            if opcode in (3, 4):
                definition(buffer, opcode, pos, schemas, channels)
            elif opcode == 11:
                first, last = struct.unpack_from('<QQ', buffer, pos + 26)
                end = pos + 46 + _U32.unpack_from(buffer, pos + 42)[0]
                counts = dict(struct.iter_unpack('<HQ', buffer[pos + 46:end]))
                statistics = first, last, counts
    except (struct.error, OverflowError) as error:
        raise ValueError(f'Corrupt MCAP summary: {path}') from error
    return schemas, channels, statistics


class RecordStream:
    """
    Read an MCAP file once in file order through top-level record headers.

    schemas and channels fill as their records pass; position is the file offset of
    the current top-level record.
    """

    def __init__(self, path):
        self.path = path
        self.schemas = {}
        self.channels = {}
        self.position = 0

    def messages(self, want, end_ns=None):
        """
        Yield (channel, log time, size, payload) of every message in file order.

        payload is the CDR bytes when want(channel, log time) is true and None
        otherwise; then the body of a message outside chunks is skipped by its record
        length without reading it. The walk stops at a chunk that starts at or after
        end_ns, which assumes that log times increase in file order.
        """
        with self.path.open('rb', buffering=_READ_BUFFER) as stream:
            if stream.read(8) != MAGIC:
                raise ValueError(f'Not an MCAP file: {self.path}')
            self.position = 8
            try:
                while True:
                    head = stream.read(9)
                    if len(head) < 9:
                        return
                    opcode, size = _RECORD.unpack(head)
                    if opcode == 5:
                        header = stream.read(22)
                        if len(header) < 22 or size < 22:
                            return
                        channel, _, time = _MESSAGE.unpack(header)
                        if want(channel, time):
                            payload = stream.read(size - 22)
                            if len(payload) < size - 22:
                                return
                            yield channel, time, size - 22, payload
                        else:
                            yield channel, time, size - 22, None
                            stream.seek(size - 22, 1)
                    elif opcode == 6:
                        body = stream.read(size)
                        if len(body) < size:
                            return
                        if end_ns is not None and _U64.unpack_from(body)[0] >= end_ns:
                            return
                        yield from self._chunk(body, want)
                    elif opcode in (3, 4):
                        body = stream.read(size)
                        if len(body) < size:
                            return
                        definition(body, opcode, 0, self.schemas, self.channels)
                    elif opcode in (2, 15):
                        return
                    else:
                        stream.seek(size, 1)
                    self.position += 9 + size
            except (struct.error, OverflowError, UnicodeDecodeError) as error:
                where = f'{self.position} in {self.path}'
                raise ValueError(f'Corrupt MCAP record at {where}') from error

    def _chunk(self, body, want):
        uncompressed = _U64.unpack_from(body, 16)[0]
        compression, pos = _string(body, 28)
        size = _U64.unpack_from(body, pos)[0]
        buffer = memoryview(body)[pos + 8:pos + 8 + size]
        if compression:
            where = f'chunk at {self.position} in {self.path}'
            buffer = memoryview(decompress(compression, buffer, uncompressed, where))
        for opcode, _, pos, stop in records(buffer):
            if opcode in (3, 4):
                definition(buffer, opcode, pos, self.schemas, self.channels)
            elif opcode == 5:
                channel, _, time = _MESSAGE.unpack_from(buffer, pos)
                payload = bytes(buffer[pos + 22:stop]) if want(channel, time) else None
                yield channel, time, stop - pos - 22, payload
