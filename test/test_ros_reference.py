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

"""Compare every field against ROS deserialization and test storage layouts."""

import builtins
import mmap
from pathlib import Path
import struct
import subprocess
import sys

from conftest import make_cloud, make_image
import numpy as np
import pytest
import rosbag2_dataset as dataset
from rosbag2_dataset._mcap import McapFile, records


def compare(actual, expected):
    if hasattr(expected, 'get_fields_and_field_types'):
        assert tuple(actual._fields) == tuple(expected.get_fields_and_field_types())
        for field in expected.get_fields_and_field_types():
            compare(getattr(actual, field), getattr(expected, field))
    elif isinstance(actual, np.ndarray):
        expected = [item[0] if isinstance(item, bytes) else item for item in expected]
        np.testing.assert_array_equal(actual, expected)
        assert not actual.flags.writeable
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            compare(left, right)
    else:
        assert actual == expected


@pytest.mark.parametrize('options', ['', 'noChunking: true\n', 'noMessageIndex: true\n',
                                     'noSummary: true\n', 'compression: Lz4\n'])
def test_every_field(write_bag, reference_events, options):
    from rclpy.serialization import deserialize_message, serialize_message
    path = write_bag(reference_events, options=options)
    with dataset.open(path) as bag:
        actual = list(bag.read())
        assert len(actual) == len(reference_events)
        for (topic, time, msg), (name, stamp, reference) in zip(actual, reference_events):
            assert (topic, time) == (name, stamp)
            compare(msg, deserialize_message(serialize_message(reference), type(reference)))
        assert sum(info.count for info in bag.topics.values()) == len(reference_events)
        assert not list(path.glob('*.idx'))


def test_summary_only_index(write_bag, reference_events, monkeypatch):
    path = write_bag(reference_events)

    def fail(*args):
        pytest.fail('Indexed opening must not access chunk data')

    monkeypatch.setattr(McapFile, 'chunk', fail)
    with dataset.open(path) as bag:
        assert all(file.used_summary for file in bag._files)
        assert len(bag.topic('/image')) == 16


def test_truncated_summary(write_bag, reference_events):
    path = write_bag(reference_events)
    source = next(path.glob('*.mcap')).read_bytes()
    summary = struct.unpack_from('<Q', source, len(source) - 28)[0]
    broken = path.parent / 'killed.mcap'
    broken.write_bytes(source[:summary])
    with dataset.open(broken) as bag:
        assert not bag._files[0].used_summary
        for (topic, time, msg), (name, stamp, reference) in zip(bag.read(), reference_events):
            assert (topic, time) == (name, stamp)
            compare(msg, reference)
        assert sum(info.count for info in bag.topics.values()) == len(reference_events)
    partial = path.parent / 'partial.mcap'
    data_end = next(record for op, record, _, _ in records(source, 8) if op == 15)
    partial.write_bytes(source[:data_end] + struct.pack('<BQ', 6, 1024) + b'partial')
    with dataset.open(partial) as bag:
        assert len(list(bag.read())) == len(reference_events)


def test_split_bag(write_bag):
    image = make_image(width=65, height=64)[0]
    events = [('/image', i * 100, image) for i in range(16)]
    path = write_bag(events, split_size=25000)
    assert len(list(path.glob('*.mcap'))) > 1
    with dataset.open(path) as bag:
        topic = bag.topic('/image')
        assert len(topic) == 16
        for i in [15, 0, 7, 1, 14]:
            assert topic[i][0] == i * 100
            compare(topic[i][1], image)
        assert [time for _, time, _ in bag.read()] == list(range(0, 1600, 100))


@pytest.mark.parametrize('encoding', ['rgb8', 'bgr8', 'rgba8', 'bgra8', 'mono8',
                                      'mono16', '16UC1', '32FC1'])
@pytest.mark.parametrize('bigendian', [False, True])
def test_image_view(write_bag, encoding, bigendian):
    message, expected = make_image(encoding, bigendian)
    path = write_bag([('/image', 10, message)])
    with dataset.open(path) as bag:
        _, decoded = bag.topic('/image')[0]
        array = dataset.image_to_numpy(decoded)
        np.testing.assert_array_equal(array, expected)
        assert array.strides[0] == message.step
        assert np.shares_memory(array, decoded.data)
        assert not array.flags.writeable
        base = decoded.data
        while not isinstance(base, mmap.mmap):
            base = base.obj if isinstance(base, memoryview) else base.base
        assert base is bag._files[0].mapping
        raw = bag.topic('/image').raw(0)[1]
        assert raw.readonly
        assert raw.obj is base
    np.testing.assert_array_equal(array, expected)
    assert bytes(raw[:2]) == b'\x00\x01'
    with pytest.raises(ValueError, match='closed'):
        bag.topic('/image')


@pytest.mark.parametrize('bigendian', [False, True])
def test_cloud_view(write_bag, bigendian):
    path = write_bag([('/cloud', 0, make_cloud(bigendian))])
    with dataset.open(path) as bag:
        message = bag.topic('/cloud')[0][1]
        cloud = dataset.pointcloud2_to_numpy(message)
        assert cloud.shape == (2, 3)
        assert cloud.strides == (103, 32)
        assert cloud.dtype.itemsize == 32
        assert np.shares_memory(cloud, message.data)
        assert not cloud.flags.writeable
        for row in range(2):
            for col in range(3):
                i = row * 3 + col
                assert cloud['x'][row, col] == i + 0.25
                assert cloud['y'][row, col] == -i - 0.5
                assert cloud['z'][row, col] == i + 0.75
                assert cloud['intensity'][row, col] == i + 100
                np.testing.assert_array_equal(cloud['normal'][row, col], [-i, i + 1, i + 2])


def test_lz4_cache(write_bag, monkeypatch):
    frame = pytest.importorskip('lz4.frame')
    images = [make_image(width=65, height=64)[0] for _ in range(3)]
    path = write_bag([('/image', i, msg) for i, msg in enumerate(images)],
                     options='compression: Lz4\nforceCompression: true\n')
    calls = []
    original = frame.decompress

    def decompress(data):
        calls.append(len(data))
        return original(data)

    monkeypatch.setattr(frame, 'decompress', decompress)
    with dataset.open(path, chunk_cache_size=1) as bag:
        assert calls == []
        topic = bag.topic('/image')
        retained = topic[0][1].data
        topic[0]
        assert len(calls) == 1
        topic[1]
        assert len(calls) == 2
        assert len(bag._files[0].cache) == 1
        np.testing.assert_array_equal(retained, images[0].data)
        topic[0]
        assert len(calls) == 3


def test_missing_compression_modules(write_bag, monkeypatch):
    from std_msgs.msg import String
    original = builtins.__import__

    def without_compression(name, *args, **kwargs):
        if name in ('zstandard', 'lz4.frame'):
            raise ModuleNotFoundError(name)
        return original(name, *args, **kwargs)

    for compression, module in [('Zstd', 'zstandard'), ('Lz4', 'lz4')]:
        path = write_bag([('/text', 0, String(data='a' * 10000))],
                         options=f'compression: {compression}\nforceCompression: true\n')
        with dataset.open(path) as bag:
            with monkeypatch.context() as context:
                context.setattr(builtins, '__import__', without_compression)
                with pytest.raises(ModuleNotFoundError, match=module):
                    bag.topic('/text')[0]


def test_runtime_without_ros(write_bag, reference_events):
    path = write_bag(reference_events)
    source = Path(__file__).parents[1]
    script = """
import sys
sys.path.insert(0, sys.argv[1])
class NoRos:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('rclpy', 'rosbag2_py', 'sensor_msgs',
                                     'rosidl_runtime_py', 'yaml', 'torch', 'lz4'):
            raise AssertionError('Unexpected runtime dependency: ' + fullname)
sys.meta_path.insert(0, NoRos())
import rosbag2_dataset
from rosbag2_dataset.torch import BagDataset
with rosbag2_dataset.open(sys.argv[2]) as bag:
    assert len(list(bag.read())) > 20
with BagDataset(sys.argv[2], ['/image']) as samples:
    assert samples[0]['/image'][1].width == 5
"""
    subprocess.run([sys.executable, '-I', '-c', script, str(source), str(path)], check=True)


def test_ancillary_records_and_mixed_messages(write_bag, reference_events):
    path = next(write_bag(reference_events).glob('*.mcap'))
    original = path.read_bytes()
    data_end = next(record for op, record, _, _ in records(original, 8) if op == 15)
    raw = b'\x00\x01\x00\x00' + struct.pack('<I', 4) + b'end\x00'
    with dataset.open(path) as bag:
        channel = next(key for key, value in bag._files[0].channels.items()
                       if value.topic == '/text')
    message = struct.pack('<HIQQ', channel, 0, 999999999, 999999999) + raw

    def record(op, payload):
        return struct.pack('<BQ', op, len(payload)) + payload

    ancillary = b''.join(record(op, b'') for op in (9, 10, 12, 13, 0x80))
    mixed = path.parent / 'mixed.mcap'
    mixed.write_bytes(original[:data_end] + ancillary + record(5, message))
    with dataset.open(mixed) as bag:
        assert bag.topic('/text')[-1][1].data == 'end'
        assert len(list(bag.read())) == len(reference_events) + 1
