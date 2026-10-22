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

"""Reject malformed inputs found by fuzzing, without ROS."""

import struct
import types

import numpy as np
import pytest
import rosbag2_dataset as dataset
from rosbag2_dataset._mcap import MAGIC, McapFile
from rosbag2_dataset.video import is_keyframe


def record(opcode, body):
    return struct.pack('<BQ', opcode, len(body)) + body


def string(text):
    data = text.encode() if isinstance(text, str) else text
    return struct.pack('<I', len(data)) + data


def schema(ident=1, name='demo/msg/One', text='uint8 value'):
    return record(3, struct.pack('<H', ident) + string(name) + string('ros2msg') + string(text))


def channel(ident=1, schema_id=1, topic='/one'):
    return record(4, struct.pack('<HH', ident, schema_id) + string(topic) + string('cdr')
                  + struct.pack('<I', 0))


def message(channel_id=1, time=10, payload=b'\x00\x01\x00\x00\x07'):
    return record(5, struct.pack('<HIQQ', channel_id, 0, time, time) + payload)


def footer(summary_start):
    return record(2, struct.pack('<QQI', summary_start, 0, 0)) + MAGIC


def write(tmp_path, data):
    path = tmp_path / 'malformed.mcap'
    path.write_bytes(MAGIC + data)
    return path


def chunk_index(index_offset):
    entries = struct.pack('<HQ', 1, index_offset)
    return record(8, struct.pack('<QQQQI', 0, 0, 8, 0, len(entries)) + entries
                  + struct.pack('<Q', 0) + string('') + struct.pack('<QQ', 0, 0))


@pytest.mark.parametrize('data', [
    footer(1000),
    record(3, struct.pack('<HI', 1, 1000)),
    chunk_index(1 << 40) + footer(8),
    chunk_index(1 << 63) + footer(8),
], ids=['summary', 'schema', 'message_index', 'message_index_u64'])
def test_offsets_beyond_file(tmp_path, data):
    with pytest.raises(ValueError, match='Corrupt MCAP'):
        dataset.open(write(tmp_path, data))


def test_raw_offset_beyond_file(tmp_path):
    file = McapFile(write(tmp_path, b''), index=False)
    with pytest.raises(ValueError, match='Corrupt MCAP'):
        file.raw(-1, 1 << 20)
    file.close()


@pytest.mark.parametrize('compression', ['lz4', 'zstd'])
def test_corrupt_compressed_chunk(tmp_path, compression):
    pytest.importorskip({'lz4': 'lz4', 'zstd': 'zstandard'}[compression])
    data = b'\xff' * 32
    body = struct.pack('<QQQI', 0, 0, 64, 0) + string(compression) + struct.pack('<Q', len(data))
    with pytest.raises(ValueError, match='Corrupt MCAP'):
        dataset.open(write(tmp_path, record(6, body + data)))


@pytest.mark.parametrize('data', [b'\x12', b'\x12\x80', b'\x16'],
                         ids=['size', 'leb128', 'extension'])
def test_truncated_av1_packet(data):
    packet = types.SimpleNamespace(format='av1', data=np.frombuffer(data, np.uint8))
    assert is_keyframe(packet, 'av1') is False
