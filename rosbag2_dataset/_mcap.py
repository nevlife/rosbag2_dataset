# Copyright 2026 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
_INDEX_ENTRY = np.dtype([('time', '<u8'), ('offset', '<u8')])


def _string(buffer, pos):
    size = _U32.unpack_from(buffer, pos)[0]
    pos += 4
    return bytes(buffer[pos:pos + size]).decode('utf-8'), pos + size


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


@dataclass(frozen=True)
class Schema:
    """Describe an embedded serialization schema."""

    name: str
    encoding: str
    text: str


@dataclass(frozen=True)
class Channel:
    """Associate a topic with a schema and message encoding."""

    schema_id: int
    topic: str
    encoding: str


class McapFile:
    """Own a mapping, channel definitions, and a bounded decompression cache."""

    def __init__(self, path, chunk_cache_size=4):
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
            self._index()
        except BaseException:
            self.close()
            raise

    def _definition(self, buffer, opcode, pos):
        if opcode == 3:
            ident = _U16.unpack_from(buffer, pos)[0]
            name, pos = _string(buffer, pos + 2)
            encoding, pos = _string(buffer, pos)
            length = _U32.unpack_from(buffer, pos)[0]
            text = bytes(buffer[pos + 4:pos + 4 + length]).decode('utf-8')
            self.schemas[ident] = Schema(name, encoding, text)
        elif opcode == 4:
            ident, schema_id = struct.unpack_from('<HH', buffer, pos)
            topic, pos = _string(buffer, pos + 4)
            encoding, pos = _string(buffer, pos)
            self.channels[ident] = Channel(schema_id, topic, encoding)

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
                        pairs = np.frombuffer(buffer, _INDEX_ENTRY, size // 16,
                                              index_offset + 15)
                        self.entries.extend((channel, int(pair['time']), chunk,
                                             int(pair['offset'])) for pair in pairs)
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
        compression, pos = _string(self.mapping, offset + 9 + 28)
        size = _U64.unpack_from(self.mapping, pos)[0]
        data = memoryview(self.mapping)[pos + 8:pos + 8 + size]
        if not compression:
            return data
        if compression == 'lz4':
            try:
                import lz4.frame
            except ModuleNotFoundError as error:
                raise ModuleNotFoundError(
                    "MCAP lz4 chunks require the missing module 'lz4' (pip install lz4)"
                ) from error
            decoded = lz4.frame.decompress(data)
        elif compression == 'zstd':
            try:
                import zstandard
            except ModuleNotFoundError as error:
                raise ModuleNotFoundError(
                    "MCAP zstd chunks require the missing module 'zstandard' "
                    '(pip install zstandard)'
                ) from error
            decoded = zstandard.ZstdDecompressor().decompress(data)
        else:
            raise ValueError(f'Unsupported MCAP chunk compression: {compression!r}')
        self.cache[offset] = decoded
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return memoryview(decoded)

    def raw(self, chunk, offset):
        """Return the CDR bytes of an indexed message without copying."""
        buffer = memoryview(self.mapping) if chunk == -1 else self.chunk(chunk)
        size = _U64.unpack_from(buffer, offset + 1)[0]
        return buffer[offset + 31:offset + 9 + size]

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
