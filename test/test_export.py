# Copyright 2026 pgw
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

"""Export synthetic bags into scenes and read them back."""

import json
import math
import pickle

import numpy as np
import pytest

pa = pytest.importorskip('pyarrow')
pq = pytest.importorskip('pyarrow.parquet')
pytest.importorskip('PIL')

from rosbag2_dataset.export import export, load_spec, main  # noqa: E402
from rosbag2_dataset.torch import SceneDataset  # noqa: E402

START = 1_000_000_000
LIDAR_NS = 100_000_000
FRAMES = 10


def _stamp(header, time):
    header.stamp.sec, header.stamp.nanosec = divmod(time, 1_000_000_000)


def _transform(parent, child, stamp, x, yaw=0.0):
    from geometry_msgs.msg import TransformStamped
    item = TransformStamped(child_frame_id=child)
    item.header.frame_id = parent
    _stamp(item.header, stamp)
    item.transform.translation.x = x
    item.transform.rotation.z = math.sin(yaw / 2)
    item.transform.rotation.w = math.cos(yaw / 2)
    return item


def _cloud(k):
    from sensor_msgs.msg import PointCloud2, PointField
    points = np.zeros(5, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('pad', '<f4'),
                                ('intensity', '<f4')])
    points['x'] = np.arange(5) + k
    points['y'] = -k
    points['z'] = 0.5
    points['intensity'] = k * 10
    points['x'][4] = np.nan
    fields = [PointField(name=name, offset=offset, datatype=7, count=1)
              for name, offset in [('x', 0), ('y', 4), ('z', 8), ('intensity', 16)]]
    message = PointCloud2(height=1, width=5, fields=fields, point_step=20, row_step=100,
                          data=points.tobytes(), is_dense=False)
    message.header.frame_id = 'lidar'
    return message


def pixels(j, height=48, width=64):
    """Return the smooth rgb8 frame j of the synthetic camera."""
    rows, cols = np.mgrid[:height, :width]
    return np.stack([rows * 4 + j, cols * 3, np.full_like(rows, 8 * j)], -1).astype(np.uint8)


def _image(j, stamp, encoding='rgb8'):
    from sensor_msgs.msg import Image
    data = pixels(j)
    if encoding == 'bgr8':
        data = data[..., ::-1]
    message = Image(height=48, width=64, encoding=encoding, step=64 * 3,
                    data=np.ascontiguousarray(data).tobytes())
    message.header.frame_id = 'camera'
    _stamp(message.header, stamp)
    return message


@pytest.fixture
def scene_bag(write_bag):
    """Write a lidar reference at 10 Hz with a 20 Hz camera, IMU, and TF."""
    from sensor_msgs.msg import CameraInfo, Imu
    from tf2_msgs.msg import TFMessage
    events = [('/tf_static', 0, TFMessage(transforms=[
        _transform('base_link', 'lidar', 0, 1.0),
        _transform('base_link', 'camera', 0, 0.5, yaw=math.pi / 2)]))]
    for k in range(FRAMES):
        stamp = START + k * LIDAR_NS
        cloud = _cloud(k)
        _stamp(cloud.header, stamp)
        events.append(('/lidar', stamp + 3_000_000, cloud))
    for j in range(2 * FRAMES):
        if j not in (6, 7):
            stamp = START + 7_000_000 + j * LIDAR_NS // 2
            events.append(('/cam/image_raw', stamp + 1_000_000, _image(j, stamp, 'bgr8')))
    for j in range(FRAMES // 2):
        stamp = START + j * 2 * LIDAR_NS
        events.append(('/slow/image_raw', stamp + 1_000_000, _image(100 + j, stamp)))
    info = CameraInfo(width=64, height=48, distortion_model='plumb_bob')
    info.header.frame_id = 'camera'
    info.k = [50.0, 0.0, 32.0, 0.0, 50.0, 24.0, 0.0, 0.0, 1.0]
    info.d = [0.1, 0.0, 0.0, 0.0, 0.0]
    events.append(('/cam/camera_info', START, info))
    for i in range(FRAMES * 10):
        imu = Imu()
        _stamp(imu.header, START + i * 10_000_000)
        imu.angular_velocity.z = float(i)
        imu.orientation_covariance = [float(i)] * 9
        events.append(('/imu', START + i * 10_000_000 + 500_000, imu))
    for i in range(-1, 2 * FRAMES + 1):
        stamp = START + i * LIDAR_NS // 2
        events.append(('/tf', stamp, TFMessage(transforms=[
            _transform('odom', 'base_link', stamp, i * 0.05)])))
    events.sort(key=lambda event: event[1])
    return write_bag(events)


def spec(**sensors):
    base = {'reference': 'lidar', 'scene_frames': 4, 'time': 'header',
            'sensors': {'lidar': {'topic': '/lidar', 'store': 'npy',
                                  'fields': ['x', 'y', 'z', 'intensity'], 'finite': True}},
            'tf': {'base': 'base_link', 'dynamic': 'odom'}}
    base['sensors'].update(sensors)
    return base


CAM = {'topic': '/cam/image_raw', 'max_ms': 20}


def frames(scene):
    return pq.read_table(scene / 'frames.parquet').to_pydict()


def names(bag, *frames):
    return [f'{bag.name}_{START + k * LIDAR_NS:019d}' for k in frames]


def test_spec(tmp_path):
    path = tmp_path / 'spec.yaml'
    path.write_text('reference: /lidar\nscene_frames: 3\ntime: log\n'
                    'sensors:\n  lidar: {topic: /lidar, store: ref}\n'
                    '  cam: {topic: /cam, store: jpeg, max_ms: 12.5, required: true}\n'
                    'hooks: test_export:frame_hook\n')
    parsed = load_spec(path)
    assert parsed.reference == '/lidar' and parsed.time == 'log'
    assert [s.name for s in parsed.sensors] == ['lidar', 'cam']
    assert parsed.sensors[1].max_ns == 12_500_000 and parsed.sensors[1].required
    assert parsed.hooks == (frame_hook,) and parsed.base is None
    assert load_spec(spec()).reference == '/lidar'
    for change, message in [({'extra': 1}, 'Unknown spec'),
                            ({'sensors': {'a': {'topic': '/a', 'store': 'gif'}}}, 'store'),
                            ({'sensors': {'a': {'topic': '/a', 'store': 'npy',
                                                'sync': 'linear'}}}, 'nearest'),
                            ({'sensors': {'a': {'topic': '/a', 'store': 'npy', 'x': 1}}},
                             'Unknown sensor'),
                            ({'sensors': {'a': {'topic': '/a', 'store': 'npy'},
                                          'b': {'topic': '/a', 'store': 'npy'}}}, 'distinct'),
                            ({'reference': 'a',
                              'sensors': {'a': {'topic': '/a', 'store': 'table'}}}, 'reference'),
                            ({'time': 'stamp'}, 'time'),
                            ({'tf': {'static': True}}, 'base')]:
        with pytest.raises(ValueError, match=message):
            load_spec({**spec(), **change})


def frame_hook(frame):
    """Add a column from the reference message; fail on request."""
    if frame['timestamp_ns'] == FAIL_AT[0]:
        raise RuntimeError('hook failure')
    return {'lidar_y': float(frame['messages']['lidar'].header.stamp.nanosec)}


FAIL_AT = [None]


def test_sync_offsets_and_files(scene_bag, tmp_path):
    scenes = export(spec(cam={**CAM, 'store': 'png'},
                         slow={'topic': '/slow/image_raw', 'max_ms': 150, 'store': 'npy'}),
                    scene_bag, tmp_path / 'out')
    assert [p.name for p in scenes] == names(scene_bag, 0, 4, 8)
    assert scenes[0].parent == tmp_path / 'out' / scene_bag.name
    rows = {key: sum((frames(p)[key] for p in scenes), []) for key in frames(scenes[0])}
    assert rows['timestamp_ns'] == [START + k * LIDAR_NS for k in range(FRAMES)]
    expected = [7_000_000] * FRAMES
    expected[3] = -43_000_000
    assert rows['cam_offset_ns'] == expected
    assert rows['cam'][3] is None
    assert rows['cam'][4] == 'cam/000000.png'
    assert rows['lidar_offset_ns'] == [0] * FRAMES
    assert rows['slow_offset_ns'] == [0, -100_000_000] * (FRAMES // 2)
    assert rows['slow'][:4] == ['slow/000000.npy'] * 2 + ['slow/000002.npy'] * 2
    np.testing.assert_allclose([pose[3] for pose in rows['T_world_base']],
                               np.arange(FRAMES) * 0.1, atol=1e-12)
    meta = json.loads((scenes[0] / 'scene.json').read_text())
    assert meta['num_frames'] == 4 and meta['time'] == 'header'
    assert meta['reference'] == 'lidar' and meta['tf']['world'] == 'odom'
    assert meta['spec']['sensors']['cam']['store'] == 'png'
    assert meta['source']['start_ns'] == START
    cam = meta['sensors']['cam']
    assert cam['stats']['missing'] == 1 and cam['camera']['K'][0] == [50.0, 0.0, 32.0]
    assert cam['camera']['topic'] == '/cam/camera_info'
    np.testing.assert_allclose(cam['T_base_sensor'],
                               [[0, -1, 0, 0.5], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                               atol=1e-12)
    assert meta['sensors']['slow']['stats']['reused'] == 2
    assert meta['sensors']['lidar']['T_base_sensor'][0][3] == 1.0
    from PIL import Image
    with Image.open(scenes[1] / 'cam/000000.png') as image:
        np.testing.assert_array_equal(np.asarray(image), pixels(8))
    np.testing.assert_array_equal(np.load(scenes[0] / 'slow/000002.npy'), pixels(101))
    cloud = np.load(scenes[0] / 'lidar/000002.npy')
    assert cloud.dtype == np.float32 and cloud.shape == (4, 4)
    np.testing.assert_array_equal(cloud[:, 0], [2, 3, 4, 5])
    np.testing.assert_array_equal(cloud[:, 3], 20)
    required = export(spec(cam={**CAM, 'store': 'jpeg', 'required': True}),
                      scene_bag, tmp_path / 'required')
    times = sum((frames(p)['timestamp_ns'] for p in required), [])
    assert times == [START + k * LIDAR_NS for k in range(FRAMES) if k != 3]
    image = np.asarray(Image.open(required[0] / 'cam/000000.jpg'), dtype=np.int16)
    assert np.abs(image - pixels(0)).mean() < 2


def test_table_and_range(scene_bag, tmp_path):
    scenes = export(spec(imu={'topic': '/imu', 'store': 'table'}), scene_bag, tmp_path / 'out',
                    start_ns=START + LIDAR_NS, end_ns=START + 7 * LIDAR_NS)
    assert [len(frames(p)['timestamp_ns']) for p in scenes] == [4, 2]
    tables = [pq.read_table(p / 'imu.parquet').to_pydict() for p in scenes]
    assert tables[0]['timestamp_ns'] == [START + i * 10_000_000 for i in range(10, 41)]
    assert tables[1]['timestamp_ns'] == [START + i * 10_000_000 for i in range(41, 61)]
    assert tables[0]['log_time_ns'][0] == START + 100_500_000
    assert tables[0]['angular_velocity.z'][0] == 10.0
    assert tables[0]['orientation_covariance'][0] == [10.0] * 9
    assert tables[0]['header.frame_id'][0] == ''
    meta = json.loads((scenes[0] / 'scene.json').read_text())['sensors']['imu']
    assert meta['stats']['count'] == 31 and meta['stats']['max_gap_ms'] == 10.0
    assert 'imu' not in frames(scenes[0])


def test_atomic_rewrite(scene_bag, tmp_path):
    options = spec(cam={**CAM, 'store': 'png'})
    options['hooks'] = 'test_export:frame_hook'
    FAIL_AT[0] = START + 5 * LIDAR_NS
    try:
        with pytest.raises(RuntimeError, match='hook failure'):
            export(options, scene_bag, tmp_path / 'out')
    finally:
        FAIL_AT[0] = None
    root = tmp_path / 'out' / scene_bag.name
    first, second, third = names(scene_bag, 0, 4, 8)
    assert sorted(p.name for p in root.iterdir()) == [f'.{second}.partial', first]
    first_frames = frames(root / first)
    assert first_frames['lidar_y'] == [float(k * LIDAR_NS) for k in range(4)]
    scenes = export(options, scene_bag, tmp_path / 'out')
    assert sorted(p.name for p in root.iterdir()) == [first, second, third]
    assert frames(scenes[0]) == first_frames
    again = export(options, scene_bag, tmp_path / 'out')
    assert sorted(p.name for p in root.iterdir()) == [first, second, third]
    assert [frames(p) for p in again] == [frames(p) for p in scenes]
    with pytest.raises(ValueError, match='no messages'):
        export(spec(cam={'topic': '/missing', 'store': 'npy'}),
               scene_bag, tmp_path / 'out')
    with pytest.raises(ValueError, match='does not accept'):
        export({**spec(), 'sensors': {'lidar': {'topic': '/lidar', 'store': 'png'}}},
               scene_bag, tmp_path / 'out')


@pytest.mark.parametrize('options', ['', 'compression: Lz4\nforceCompression: true\n'])
def test_ref_round_trip(write_bag, scene_bag, tmp_path, options):
    if options:
        scene_bag = write_bag([(t, time, m) for t, time, m in _events(scene_bag)],
                              options=options)
    export(spec(cam={**CAM, 'store': 'npy'}), scene_bag, tmp_path / 'files')
    single = next(scene_bag.glob('*.mcap'))
    refs = export({**spec(cam={**CAM, 'store': 'ref'}),
                   'sensors': {'lidar': {'topic': '/lidar', 'store': 'ref', 'finite': True,
                                         'fields': ['x', 'y', 'z', 'intensity']},
                               'cam': {**CAM, 'store': 'ref'}}},
                  single, tmp_path / 'refs')
    assert refs[0].parent.name == single.stem
    columns = frames(refs[0])
    assert columns['cam_file'][0] == single.name
    assert columns['cam_chunk'][0] >= -1 and columns['cam_record'][0] > 0
    assert not (refs[0] / 'cam').exists()
    meta = json.loads((refs[0] / 'scene.json').read_text())
    assert meta['sensors']['cam']['schema']['name'] == 'sensor_msgs/msg/Image'
    with SceneDataset(tmp_path / 'files') as expected, SceneDataset(tmp_path / 'refs') as actual:
        assert len(expected) == len(actual) == FRAMES
        for index in range(FRAMES):
            a, b = expected[index], actual[index]
            for name in ('lidar', 'cam'):
                if a[name] is None:
                    assert b[name] is None and index == 3
                else:
                    np.testing.assert_array_equal(a[name], b[name])
        np.testing.assert_array_equal(actual[4]['cam'], pixels(8))
    moved = tmp_path / 'moved'
    moved.mkdir()
    scene_bag.rename(moved / scene_bag.name)
    with pytest.raises(FileNotFoundError):
        SceneDataset(tmp_path / 'refs')[0]
    for bag_root in (moved, {str(scene_bag): moved / scene_bag.name}):
        with SceneDataset(tmp_path / 'refs', bag_root=bag_root) as actual:
            np.testing.assert_array_equal(actual[4]['cam'], pixels(8))


def _events(path):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id='mcap'),
                rosbag2_py.ConverterOptions('', ''))
    types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}
    while reader.has_next():
        topic, data, time = reader.read_next()
        yield topic, time, deserialize_message(data, types[topic])


def test_mp4(scene_bag, tmp_path):
    pytest.importorskip('av')
    scenes = export(spec(cam={**CAM, 'store': 'mp4', 'gop': 2, 'crf': 10}),
                    scene_bag, tmp_path / 'out')
    meta = json.loads((scenes[0] / 'scene.json').read_text())['sensors']['cam']
    assert meta['video']['file'] == 'cam.mp4' and meta['video']['fps'] == 10
    assert meta['video']['codec'] == 'libx264' and meta['video']['pix_fmt'] == 'yuv420p'
    assert meta['video']['frames'] == 3
    assert frames(scenes[0])['cam'] == [0, 1, 2, None]
    with SceneDataset(tmp_path / 'out') as samples:
        order = [4, 0, 9, 8, 5, 6, 1, 2, 7]
        for index in order:
            j = 2 * index
            assert np.abs(samples[index]['cam'].astype(np.int16) - pixels(j)).mean() < 3
        assert samples[3]['cam'] is None


def test_scene_dataset(scene_bag, tmp_path):
    export(spec(cam={**CAM, 'store': 'png'}, imu={'topic': '/imu', 'store': 'table'}),
           scene_bag, tmp_path / 'out')
    samples = SceneDataset(tmp_path / 'out')
    assert len(samples) == FRAMES and len(samples.scenes) == 3
    sample = samples[5]
    assert sample['frame'] == 1 and sample['scene'].endswith(names(scene_bag, 4)[0])
    assert sample['timestamp_ns'] == START + 5 * LIDAR_NS
    assert sample['T_world_base'].shape == (4, 4)
    assert sample['T_world_base'][0, 3] == pytest.approx(0.5)
    np.testing.assert_array_equal(sample['imu']['timestamp_ns'],
                                  START + np.arange(41, 51) * 10_000_000)
    np.testing.assert_array_equal(sample['imu']['angular_velocity.z'], np.arange(41, 51))
    assert len(samples[0]['imu']['timestamp_ns']) == 1
    assert samples[-1]['timestamp_ns'] == START + 9 * LIDAR_NS
    with pytest.raises(IndexError):
        samples[FRAMES]
    restored = pickle.loads(pickle.dumps(samples))
    assert restored._files is None
    np.testing.assert_array_equal(restored[4]['cam'], pixels(8))
    only = SceneDataset([tmp_path / 'out' / scene_bag.name / names(scene_bag, 0)[0]],
                        sensors=['cam'],
                        transform=lambda sample: sample['cam'].shape)
    assert len(only) == 4 and only[0] == (48, 64, 3)
    for dataset in (samples, restored, only):
        dataset.close()


def test_cli(scene_bag, tmp_path, capsys):
    path = tmp_path / 'spec.yaml'
    path.write_text(json.dumps(spec(cam={**CAM, 'store': 'jpeg'})))
    main([str(path), str(scene_bag), str(tmp_path / 'out'),
          '--start-ns', str(START + 6 * LIDAR_NS)])
    root = tmp_path / 'out' / scene_bag.name
    printed = capsys.readouterr().out.split()
    assert printed == [str(root / name) for name in names(scene_bag, 6)]
    assert len(frames(root / names(scene_bag, 6)[0])['timestamp_ns']) == 4
    path.write_text(json.dumps({**spec(), 'scene_frames': 5}))
    main([str(path), str(scene_bag), str(tmp_path / 'out'), '--clean'])
    assert sorted(p.name for p in root.iterdir()) == names(scene_bag, 0, 5)


def test_scene_names(scene_bag, tmp_path):
    full = export(spec(), scene_bag, tmp_path / 'out')
    part = export(spec(), scene_bag, tmp_path / 'out', start_ns=START + 2 * LIDAR_NS,
                  end_ns=START + 5 * LIDAR_NS)
    assert [p.name for p in part] == names(scene_bag, 2)
    root = tmp_path / 'out' / scene_bag.name
    assert sorted(p.name for p in root.iterdir()) == names(scene_bag, 0, 2, 4, 8)
    assert [frames(p) for p in export(spec(), scene_bag, tmp_path / 'out')] == [
        frames(p) for p in full]
    export(spec(), scene_bag, tmp_path / 'out', end_ns=START + 4 * LIDAR_NS, clean=True)
    assert sorted(p.name for p in root.iterdir()) == names(scene_bag, 0)
