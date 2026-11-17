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

"""Rewrite synthetic bags and read the result with this reader and with ROS."""

import io

import numpy as np
import pytest

pytest.importorskip('rosbags')
PIL = pytest.importorskip('PIL.Image')

import rosbag2_dataset as rd  # noqa: E402
from rosbag2_dataset.bag import align  # noqa: E402
from rosbag2_dataset.rewrite import compressed_image  # noqa: E402
from rosbag2_dataset.rewrite import load_spec  # noqa: E402
from rosbag2_dataset.rewrite import main  # noqa: E402
from rosbag2_dataset.rewrite import rewrite  # noqa: E402

START = 1_000_000_000
LIDAR_NS = 100_000_000
MISSING = (6, 7)


def _stamp(header, time):
    header.stamp.sec, header.stamp.nanosec = divmod(time, 1_000_000_000)


def pixels(j, height=48, width=64):
    """Return a smooth rgb8 frame whose channels differ, so swapped order shows."""
    rows, cols = np.mgrid[:height, :width]
    return np.stack([150 + rows + j, cols * 2, np.full_like(rows, 20)], -1).astype(np.uint8)


def _image(j, stamp, encoding):
    from sensor_msgs.msg import Image

    data = pixels(j)
    if encoding == 'bgr8':
        data = data[..., ::-1]
    message = Image(
        height=48,
        width=64,
        encoding=encoding,
        step=64 * 3,
        data=np.ascontiguousarray(data).tobytes(),
    )
    message.header.frame_id = f'camera_{encoding}'
    _stamp(message.header, stamp)
    return message


@pytest.fixture
def source_bag(write_bag, ros):
    """Write a 10 Hz lidar reference with two 20 Hz cameras, TF, and strings."""
    from sensor_msgs.msg import PointCloud2
    from std_msgs.msg import String
    from tf2_msgs.msg import TFMessage

    events = [('/tf_static', START + LIDAR_NS, TFMessage()), ('/late', START * 2, String())]
    for k in range(10):
        cloud = PointCloud2(height=1, width=1, point_step=4, row_step=4, data=bytes([k] * 4))
        _stamp(cloud.header, START + k * LIDAR_NS)
        events.append(('/lidar', START + k * LIDAR_NS + 3_000_000, cloud))
        events.append(('/text', START + k * LIDAR_NS, String(data=str(k))))
    for j in range(20):
        stamp = START + 7_000_000 + j * LIDAR_NS // 2
        if j not in MISSING:
            events.append(('/cam/image_raw', stamp + 1_000_000, _image(j, stamp, 'rgb8')))
        events.append(('/bgr/image_raw', stamp + 2_000_000, _image(j, stamp, 'bgr8')))
    events.sort(key=lambda event: event[1])
    latched = ros._storage.QoS(1).transient_local().reliable()
    return write_bag(events, qos={'/tf_static': [latched], '/cam/image_raw': [latched]})


SPEC = {
    'keep': ['/lidar', '/tf_static', '/cam/image_raw', '/bgr/image_raw', '/late'],
    'jpeg': {'quality': 95, 'topics': ['/cam/image_raw', '/bgr/image_raw']},
    'nearest': {'reference': '/lidar', 'tolerance_ms': 20, 'topics': ['/cam/image_raw']},
}


def test_spec(tmp_path):
    path = tmp_path / 'spec.yaml'
    path.write_text('drop: [/a]\njpeg: {topics: [/b]}\n')
    parsed = load_spec(path)
    assert parsed.keep is None and parsed.drop == ('/a',) and parsed.jpeg == {'/b': 92}
    assert load_spec(SPEC).tolerance_ns == 20_000_000
    assert load_spec({'keep': ['/b'], 'jpeg': {'topics': {'/b': 70}}}).jpeg == {'/b': 70}
    for change, message in [
        ({'extra': 1}, 'Unknown spec'),
        ({'drop': ['/a']}, 'exactly one'),
        ({'jpeg': {'topics': [], 'size': 1}}, 'Unknown jpeg'),
        ({'nearest': {'reference': '/a', 'tolerance_ms': 1, 'max': 1}}, 'Unknown nearest'),
    ]:
        with pytest.raises(ValueError, match=message):
            load_spec({**SPEC, **change})
    with pytest.raises(ValueError, match='exactly one'):
        load_spec({})


@pytest.mark.parametrize('frame_id', ['', 'a', 'cam', 'camera_link'])
def test_compressed_image_cdr(ros, frame_id):
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import CompressedImage

    message = CompressedImage(format='jpeg', data=b'\xff\xd8\x01\x02\x03')
    message.header.frame_id = frame_id
    message.header.stamp.sec, message.header.stamp.nanosec = -3, 123
    assert compressed_image(-3, 123, frame_id, b'\xff\xd8\x01\x02\x03') == serialize_message(
        message
    )


def test_rewrite(source_bag, tmp_path):
    out = tmp_path / 'out'
    window = (START + LIDAR_NS, START + 8 * LIDAR_NS)
    counts = rewrite(SPEC, source_bag, out, start_ns=window[0], end_ns=window[1])
    rows = {row.topic: (row.input_count, row.output_count) for row in counts}
    assert rows == {
        '/lidar': (7, 7),
        '/tf_static': (1, 1),
        '/cam/image_raw/compressed': (12, 6),
        '/bgr/image_raw/compressed': (14, 14),
    }
    with rd.open(source_bag) as source, rd.open(out) as result:
        assert {name: info.type for name, info in result.topics.items()} == {
            '/lidar': 'sensor_msgs/msg/PointCloud2',
            '/tf_static': 'tf2_msgs/msg/TFMessage',
            '/cam/image_raw/compressed': 'sensor_msgs/msg/CompressedImage',
            '/bgr/image_raw/compressed': 'sensor_msgs/msg/CompressedImage',
        }
        lidar = source.topic('/lidar')
        for i in range(7):
            assert result.topic('/lidar').raw(i)[0] == lidar.raw(i + 1)[0]
            assert bytes(result.topic('/lidar').raw(i)[1]) == bytes(lidar.raw(i + 1)[1])
        assert result.topic('/lidar').schema() == lidar.schema()
        for name in ('/tf_static', '/cam/image_raw/compressed'):
            qos = result.topic(name).channel().metadata['offered_qos_profiles']
            assert 'durability: transient_local' in qos
        chosen = [2, 4, 8, 10, 12, 14]
        camera = result.topic('/cam/image_raw/compressed')
        for (time, message), j in zip(camera, chosen, strict=True):
            stamp = START + 7_000_000 + j * LIDAR_NS // 2
            assert time == stamp + 1_000_000 and message.format == 'jpeg'
            assert message.header.frame_id == 'camera_rgb8'
            assert message.header.stamp.sec * 10**9 + message.header.stamp.nanosec == stamp
        jpegs = {'/cam/image_raw/compressed': chosen, '/bgr/image_raw/compressed': range(2, 16)}
        for name, frames in jpegs.items():
            for (_, message), j in zip(result.topic(name), frames, strict=True):
                decoded = np.asarray(PIL.open(io.BytesIO(message.data)))
                assert decoded.shape == (48, 64, 3)
                assert np.abs(decoded.astype(int) - pixels(j)).mean() < 2


def test_ros_reads_rewrite(ros, source_bag, tmp_path):
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import CompressedImage

    out = tmp_path / 'out'
    rewrite({**SPEC, 'nearest': {}}, source_bag, out)
    info = ros.Info().read_metadata(str(out), 'mcap')
    assert {
        topic.topic_metadata.name: (topic.topic_metadata.type, topic.message_count)
        for topic in info.topics_with_message_count
    } == {
        '/lidar': ('sensor_msgs/msg/PointCloud2', 10),
        '/tf_static': ('tf2_msgs/msg/TFMessage', 1),
        '/cam/image_raw/compressed': ('sensor_msgs/msg/CompressedImage', 18),
        '/bgr/image_raw/compressed': ('sensor_msgs/msg/CompressedImage', 20),
        '/late': ('std_msgs/msg/String', 1),
    }
    reader = ros.SequentialReader()
    reader.open(ros.StorageOptions(uri=str(out), storage_id='mcap'), ros.ConverterOptions('', ''))
    times = []
    while reader.has_next():
        topic, data, time = reader.read_next()
        times.append(time)
        if topic == '/bgr/image_raw/compressed':
            message = deserialize_message(data, CompressedImage)
            assert message.format == 'jpeg' and message.header.frame_id == 'camera_bgr8'
    assert len(times) == 50 and times == sorted(times)


def test_cli_and_checks(source_bag, tmp_path, capsys):
    spec = tmp_path / 'spec.yaml'
    spec.write_text('drop: [/text, /cam/image_raw, /bgr/image_raw]\n')
    main([str(spec), str(source_bag), str(tmp_path / 'out')])
    lines = capsys.readouterr().out.splitlines()
    assert lines[:2] == [str(tmp_path / 'out' / name) for name in ('metadata.yaml', 'out.mcap')]
    assert lines[2].split() == ['topic', 'input', 'output', 'output', 'bytes']
    assert sorted(line.split()[0] for line in lines[3:]) == ['/late', '/lidar', '/tf_static']
    for change, message in [
        ({'keep': ['/nothing']}, 'not in the bag'),
        ({'keep': ['/lidar']}, 'must be kept'),
        ({'jpeg': {'topics': ['/lidar']}, 'nearest': {}}, 'is not sensor_msgs/msg/Image'),
    ]:
        with pytest.raises(ValueError, match=message):
            rewrite({**SPEC, **change}, source_bag, tmp_path / 'never')
    assert not (tmp_path / 'never').exists()


TOLERANCE_NS = 30_000_000
THINNED = ('/left', '/right', '/tie')
LAYOUTS = {
    'chunked': {},
    'zstd': {'options': 'compression: Zstd\n'},
    'unchunked': {'options': 'noChunking: true\nnoSummaryCRC: true\n'},
    'split': {'split_size': 200_000},
}


def _frame(stamp, marker):
    from sensor_msgs.msg import Image

    image = Image(height=64, width=64, encoding='mono8', step=64)
    image.data = marker.to_bytes(4, 'little') * 1024
    _stamp(image.header, stamp)
    return image


@pytest.fixture
def jittered_events():
    """Jittered 10 Hz reference stamps; /left misses a second, /tie has equidistant pairs."""
    from sensor_msgs.msg import PointCloud2

    rng = np.random.default_rng(7)
    events = []
    for k in range(60):
        stamp = START + k * LIDAR_NS + int(rng.integers(-4, 5)) * 1_000_000
        cloud = PointCloud2(height=1, width=1, point_step=4, row_step=4, data=bytes([k] * 4))
        _stamp(cloud.header, stamp)
        events.append(('/lidar', stamp + int(rng.integers(5, 40)) * 1_000_000, cloud))
        # Equal distances choose the earlier stamp, equal stamps the first message.
        for n, offset in enumerate((-15_000_000, -15_000_000, 15_000_000)):
            frame = _frame(stamp + offset, k * 3 + n)
            events.append(('/tie', stamp + offset + 2_000_000 + n, frame))
    for name, period, gap in (('/left', 66_000_000, range(20, 35)), ('/right', 50_000_000, ())):
        for j in range(60 * LIDAR_NS // period):
            if j in gap:
                continue
            stamp = START + j * period + int(rng.integers(-5, 6)) * 1_000_000
            events.append((name, stamp + int(rng.integers(1, 20)) * 1_000_000, _frame(stamp, j)))
    events.sort(key=lambda event: event[1])
    return events


def _expected(bag, reference, window):
    """Return the log times that align() chooses per thinned topic over the window."""
    rule = rd.SampleRule([reference, *THINNED], tolerance_ns=TOLERANCE_NS, optional=THINNED)
    with rd.open(bag) as source:
        rows = {}
        for name in rule.topics:
            times = source.topic(name).times
            rows[name] = np.flatnonzero((times >= window[0]) & (times < window[1]))
        clocks = {name: source.topic(name).stamps('header')[rows[name]] for name in rule.topics}
        chosen = align(rule, clocks).indices
        expected = {}
        for column, name in enumerate(THINNED, 1):
            picked = np.unique(chosen[:, column])
            expected[name] = source.topic(name).times[rows[name][picked[picked >= 0]]]
        return expected


@pytest.mark.parametrize('layout', [*LAYOUTS, 'unfinished file'])
@pytest.mark.parametrize('window', [(0, 2**63), (START + 1_550_000_000, START + 4_450_000_000)])
def test_streaming_thinning_matches_align(write_bag, jittered_events, tmp_path, layout, window):
    bag = write_bag(jittered_events, **LAYOUTS.get(layout, LAYOUTS['unchunked']))
    files = len(list(bag.glob('*.mcap')))
    # Under ten splits the file names sort in recording order.
    assert 1 < files < 10 if layout == 'split' else files == 1
    if layout == 'unfinished file':
        # Without its footer the file has no summary and nothing lists its topics.
        unfinished = tmp_path / 'unfinished.mcap'
        unfinished.write_bytes(next(bag.glob('*.mcap')).read_bytes()[:-8])
        bag = unfinished
    keep = [*THINNED] + (['/lidar'] if layout != 'zstd' else [])
    spec = {
        'keep': keep,
        'nearest': {'reference': '/lidar', 'tolerance_ms': 30, 'topics': list(THINNED)},
    }
    out = tmp_path / 'out'
    reports = []
    rewrite(
        spec, bag, out, start_ns=window[0], end_ns=window[1],
        progress=lambda stage, done, total: reports.append((stage, done, total)),
    )
    expected = _expected(bag, '/lidar', window)
    assert expected['/left'].size and expected['/tie'].size
    with rd.open(bag) as source, rd.open(out) as result:
        assert set(result.topics) == set(keep)
        for name in THINNED:
            topic = result.topic(name)
            assert topic.times.tolist() == expected[name].tolist(), name
            original = source.topic(name)
            for i, time in enumerate(expected[name].tolist()):
                index = int(np.searchsorted(original.times, time))
                assert bytes(topic.raw(i)[1]) == bytes(original.raw(index)[1])
        tie = result.topic('/tie')
        assert all(int.from_bytes(message.data[:4], 'little') % 3 == 0 for _, message in tie)
    total = sum(path.stat().st_size for path in ([bag] if bag.is_file() else bag.glob('*.mcap')))
    assert {stage for stage, _, _ in reports} == {'write'}
    assert all(t == total for _, _, t in reports)
    done = [d for _, d, _ in reports]
    assert done == sorted(done)
    if layout == 'unchunked' and window[1] < 2**63:
        # The walk stops at the end of the range instead of reading the rest.
        assert done[-1] < total * 0.8


def test_output_in_log_time_order(write_bag, jittered_events, tmp_path):
    from rosbag2_dataset._mcap import RecordStream

    bag = write_bag(jittered_events, **LAYOUTS['split'])
    assert 2 < len(list(bag.glob('*.mcap'))) < 10
    nearest = {'reference': '/lidar', 'tolerance_ms': 30, 'topics': ['/left']}
    spec = {'drop': [], 'nearest': nearest}
    counts = rewrite(spec, bag, tmp_path / 'out')
    rows = {row.topic: (row.input_count, row.output_count) for row in counts}
    assert rows['/left'][1] < rows['/left'][0] and rows['/right'][0] == rows['/right'][1]
    # Held /left messages are written in log-time order among the other topics.
    stream = RecordStream(next((tmp_path / 'out').glob('*.mcap')))
    times = [time for _, time, _, _ in stream.messages(lambda channel, time: False)]
    assert len(times) == sum(output for _, output in rows.values())
    assert times == sorted(times)
