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

"""Check indexing, synchronization, and worker-independent dataset ownership."""

import multiprocessing
import os
import pickle

import numpy as np
import pytest
import rosbag2_dataset as dataset
from rosbag2_dataset.torch import BagDataset


@pytest.fixture
def time_bag(write_bag):
    from sensor_msgs.msg import Image, Imu
    events = []
    for i, time in enumerate([300, 100, 200, 200]):
        image = Image(height=1, width=1, step=1, encoding='mono8', data=bytes([i]))
        image.header.stamp.nanosec = [3000, 1000, 2000, 2000][i]
        events.append(('/cam', time, image))
    for log_time, stamp in [(90, 3100), (210, 1100), (400, 2100)]:
        message = Imu()
        message.header.stamp.nanosec = stamp
        events.append(('/imu', log_time, message))
    return write_bag(events)


def test_index_and_reads(time_bag):
    with dataset.open(time_bag) as bag:
        topic = bag.topic('/cam')
        assert len(topic) == 4
        np.testing.assert_array_equal(topic.times, [100, 200, 200, 300])
        assert topic.times.dtype == np.int64
        assert not topic.times.flags.writeable
        for index, expected in [(0, 1), (3, 0), (-1, 0), (1, 2), (2, 3)]:
            assert topic[index][1].data[0] == expected
            assert topic.raw(index)[0] == int(topic.times[index])
        for time, expected in [(0, 0), (100, 0), (150, 0), (199, 1), (200, 1),
                               (299, 3), (999, 3)]:
            assert topic.nearest(time) == expected
        assert [time for _, time, _ in bag.read()] == [90, 100, 200, 200, 210, 300, 400]
        assert [time for _, time, _ in bag.read(['/cam'], 100, 300)] == [100, 200, 200]
        assert list(bag.read(start_ns=500)) == []
        assert list(bag.read(topics=[])) == []
        assert len(list(bag.read('/cam'))) == 4
        assert len(list(bag.read(['/cam', '/cam']))) == 4
        with pytest.raises(IndexError):
            topic[4]
        with pytest.raises(TypeError):
            pickle.dumps(bag)


def test_synced(time_bag):
    with dataset.open(time_bag) as bag:
        pairs = bag.synced(['/cam', '/imu'], reference='/cam', tolerance_ns=10)
        assert list(pairs) == [(0, 0), (1, 1), (2, 1)]
        assert pairs[-1] == (2, 1)
        assert pairs[:2] == [(0, 0), (1, 1)]
        assert not pairs.indices.flags.writeable
        assert list(bag.synced(['/cam', '/imu'], reference='/imu', tolerance_ns=10)) == [
            (0, 0), (1, 1)]
        header = bag.synced(['/cam', '/imu'], reference='/cam', tolerance_ns=100,
                            time_source='header')
        assert list(header) == [(0, 1), (1, 2), (2, 2), (3, 0)]
        assert list(bag.synced(['/cam', '/imu'], tolerance_ns=0)) == []
        with pytest.raises(ValueError, match='reference'):
            bag.synced(['/cam'], reference='/missing')


def _worker(samples, queue):
    sample = samples[0]
    queue.put((os.getpid(), samples._pid, list(sample), sample['/cam'][0]))
    samples.close()


@pytest.mark.parametrize('method', ['fork', 'spawn'])
def test_dataset_workers(time_bag, method):
    samples = BagDataset([time_bag, time_bag], ['/cam', '/imu'],
                         sync={'reference': '/cam', 'tolerance_ns': 10})
    assert len(samples) == 6
    assert samples[0]['/cam'][0] == 100
    assert samples[3]['/cam'][0] == 100
    assert samples[-1]['/cam'][0] == 200
    restored = pickle.loads(pickle.dumps(samples))
    assert restored._bags is None
    assert restored[0]['/cam'][0] == 100
    context = multiprocessing.get_context(method)
    queue = context.Queue()
    worker = context.Process(target=_worker, args=(samples, queue))
    worker.start()
    pid, owner, names, stamp = queue.get(timeout=30)
    worker.join(30)
    assert worker.exitcode == 0
    assert pid == owner and owner != os.getpid()
    assert names == ['/cam', '/imu'] and stamp == 100
    assert samples._pid == os.getpid()
    assert samples[0]['/cam'][0] == 100
    samples.close()
    restored.close()
    queue.close()


def test_dataset_unsynced_and_transform(time_bag):
    with BagDataset(time_bag, ['/cam', '/imu'],
                    transform=lambda sample: next(iter(sample.values()))[0]) as samples:
        assert len(samples) == 7
        assert [samples[i] for i in range(len(samples))] == [90, 100, 200, 200, 210, 300, 400]
        with pytest.raises(IndexError):
            samples[7]
    with BagDataset(time_bag, '/cam') as samples:
        assert len(samples) == 4


def test_empty_bag_and_empty_topic(write_bag):
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
    path = write_bag([('/cam', 1, Image()), ('/text', 2, String())])
    from rosbag2_dataset._mcap import records
    source = next(path.glob('*.mcap')).read_bytes()
    empty = path.parent / 'empty.mcap'
    headers = [source[:8]]
    for opcode, record, _, end in records(source, 8):
        if opcode in (1, 3, 4):
            headers.append(source[record:end])
    empty.write_bytes(b''.join(headers))
    with dataset.open(empty) as bag:
        assert bag.topics['/cam'].count == 0
        assert bag.topics['/cam'].start_ns is None
        assert len(list(bag.read())) == 0
        assert len(bag.synced(['/cam', '/text'])) == 0
        with pytest.raises(ValueError, match='empty'):
            bag.topic('/cam').nearest(0)
