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

"""Check that live samples choose what the whole bag chooses."""

import numpy as np
import pytest
from rosbag2_dataset import LiveSampler
from rosbag2_dataset import open as open_bag
from rosbag2_dataset import SampleRule
from rosbag2_dataset import Window
from rosbag2_dataset.torch import BagDataset

MS = 1_000_000


@pytest.fixture
def jitter_bag(write_bag):
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Image
    from sensor_msgs.msg import Imu

    rng = np.random.default_rng(7)
    events = []
    for i in range(60):
        log = 1000 * MS + i * 100 * MS + int(rng.integers(0, 8)) * MS
        image = Image(height=1, width=1, step=1, encoding='mono8', data=bytes([i]))
        image.header.stamp.sec, image.header.stamp.nanosec = divmod(log - 30 * MS, 10**9)
        events.append(('/cam', log, image))
    seq = 0
    for i in range(600):
        if 120 <= i < 160:
            continue
        log = 1000 * MS + i * 10 * MS + int(rng.integers(0, 3)) * MS
        imu = Imu()
        imu.header.stamp.sec, imu.header.stamp.nanosec = divmod(log - 2 * MS, 10**9)
        imu.linear_acceleration.x = float(seq)
        seq += 1
        events.append(('/imu', log, imu))
    for i in range(120):
        log = 1000 * MS + i * 50 * MS + int(rng.integers(0, 5)) * MS
        odom = Odometry()
        odom.header.stamp.sec, odom.header.stamp.nanosec = divmod(log - 5 * MS, 10**9)
        odom.pose.pose.position.x = float(i)
        events.append(('/odom', log, odom))
        if i % 10 == 3:
            events.append(('/odom', log, odom))
    events.sort(key=lambda event: event[1])
    return write_bag(events)


def _key(entry):
    """Identify a message by its log time and the field each test topic numbers."""
    if entry is None:
        return None
    time, message = entry
    if hasattr(message, 'data'):
        return time, bytes(message.data)
    if hasattr(message, 'linear_acceleration'):
        return time, message.linear_acceleration.x
    return time, message.pose.pose.position.x


def _keys(sample):
    return {
        name: [_key(e) for e in value]
        if isinstance(value, list)
        else value
        if name == '@time'
        else _key(value)
        for name, value in sample.items()
    }


SYNC = {
    'reference': '/cam',
    'tolerance_ns': {'/imu': 6 * MS, '/odom': 30 * MS},
    'optional': ['/imu'],
    'windows': {
        '/cam': Window(every_ns=500 * MS, past=2, tolerance_ns=60 * MS),
        '/odom': Window(every_ns=500 * MS, past=4, future=4, tolerance_ns=30 * MS),
    },
}


def _live(bag, time_source):
    rule = SampleRule(
        ['/cam', '/imu', '/odom'],
        SYNC['reference'],
        SYNC['tolerance_ns'],
        optional=SYNC['optional'],
        windows=SYNC['windows'],
    )
    live = LiveSampler(rule)
    complete = []
    held = []
    for name, log, message in bag.read(['/cam', '/imu', '/odom']):
        clock = (
            log
            if time_source == 'log'
            else (message.header.stamp.sec * 10**9 + message.header.stamp.nanosec)
        )
        live.add(name, clock, message, log_ns=log)
        complete += live.ready()
        held.append(sum(len(times) for times in live._times.values()))
    return complete, live.ready(now_ns=2**62, max_wait_ns=0), max(held)


@pytest.mark.parametrize('time_source', ['log', 'header'])
def test_live_matches_bag(jitter_bag, time_source):
    with open_bag(jitter_bag) as bag:
        complete, early, held = _live(bag, time_source)
    with BagDataset(
        jitter_bag, ['/cam', '/imu', '/odom'], sync=dict(SYNC, time_source=time_source)
    ) as samples:
        full = [samples[i] for i in range(len(samples))]
    assert len(complete) >= 30 and early
    assert [_keys(s) for s in complete] == [_keys(s) for s in full[: len(complete)]]
    assert [s['@time'] for s in complete + early] == [s['@time'] for s in full]
    assert any(s['/imu'] is None for s in complete)
    assert any(None in s['/odom@future'] for s in early)
    assert any(None in s['/cam@past'] for s in complete)
    assert all(len(s['/odom@past']) == 4 and len(s['/odom@future']) == 4 for s in full)
    assert held < 450


def test_live_ties():
    rule = SampleRule(['/a', '/b'], tolerance_ns=5)
    live = LiveSampler(rule)
    for name, time in [('/a', 10), ('/b', 5), ('/b', 15), ('/b', 30)]:
        live.add(name, time, (name, time))
    (sample,) = live.ready()
    assert sample['/b'] == (5, ('/b', 5)) and sample['@time'] == 10


def test_rule_errors():
    with pytest.raises(ValueError, match='windows'):
        SampleRule(['/a'], windows={'/b': Window(every_ns=1)})
    with pytest.raises(ValueError, match='every_ns'):
        Window(every_ns=0)
    live = LiveSampler(SampleRule(['/a', '/b'], tolerance_ns=10))
    live.add('/a', 10, None)
    live.add('/b', 20, None)
    live.add('/b', 25, None)
    assert len(live.ready()) == 1
    with pytest.raises(ValueError, match='clock order'):
        live.add('/a', 5, None)


def test_live_waits_for_reference_future(jitter_bag):
    windows = {'/cam': Window(every_ns=100 * MS, past=1, future=1, tolerance_ns=30 * MS)}
    rule = SampleRule(['/cam', '/odom'], tolerance_ns=30 * MS, windows=windows)
    live = LiveSampler(rule)
    complete = []
    with open_bag(jitter_bag) as bag:
        for name, log, message in bag.read(['/cam', '/odom']):
            live.add(name, log, message)
            complete += live.ready()
    with BagDataset(
        jitter_bag, ['/cam', '/odom'], sync={'tolerance_ns': 30 * MS, 'windows': windows}
    ) as samples:
        full = [samples[i] for i in range(len(samples))]
    assert len(complete) >= 50
    assert [_keys(s) for s in complete] == [_keys(s) for s in full[: len(complete)]]
    assert all(s['/cam@future'][0] is not None for s in complete)


def test_hold_takes_the_last_message_before_the_reference():
    rule = SampleRule(
        ['/a', '/goal'], tolerance_ns={'/goal': 100}, hold=['/goal'], optional=['/goal']
    )
    clocks = {'/a': np.array([10, 60, 120, 400]), '/goal': np.array([5, 50, 110])}
    assert [rule.select(clocks, i)[0][1] for i in range(4)] == [0, 1, 2, -1]


def test_hold_does_not_wait_and_matches_a_recording():
    rule = SampleRule(['/a', '/goal'], tolerance_ns={'/goal': 1000}, hold=['/goal'])
    live = LiveSampler(rule)
    live.add('/goal', 5, 'g')
    live.add('/a', 10, 'a')
    samples = live.ready()
    assert samples == [{'@time': 10, '/a': (10, 'a'), '/goal': (5, 'g')}] and live.early == 0


def test_early_counts_samples_released_by_the_wait():
    live = LiveSampler(SampleRule(['/a', '/b'], tolerance_ns=10, optional=['/b']))
    live.add('/a', 100, 'a')
    assert live.ready(now_ns=100, max_wait_ns=50) == []
    assert len(live.ready(now_ns=200, max_wait_ns=50)) == 1
    assert live.early == 1


@pytest.mark.parametrize(
    'options,match',
    [
        ({'hold': ['/a']}, 'non-reference'),
        ({'hold': ['/b'], 'windows': {'/b': {'every_ns': 1}}}, 'no window'),
    ],
)
def test_hold_errors(options, match):
    with pytest.raises(ValueError, match=match):
        SampleRule(['/a', '/b'], **options)
