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

"""Decode and export video packets generated with PyAV."""

from fractions import Fraction
import json
import struct

import numpy as np
import pytest

START = 1_000_000_000
STEP_NS = 100_000_000
FRAMES = 12
FOXGLOVE = 'foxglove_msgs/msg/CompressedVideo'
FFMPEG = 'ffmpeg_image_transport_msgs/msg/FFMPEGPacket'
DEFINITIONS = {
    FOXGLOVE: 'builtin_interfaces/Time timestamp\nstring frame_id\nuint8[] data\nstring format\n',
    FFMPEG: ('std_msgs/Header header\nint32 width       # original image width\n'
             'int32 height\nstring encoding\t  # encoding used, see above\nuint64 pts\n'
             'uint8  flags      # packet flags\nbool is_bigendian\nuint8[] data\n'),
}
ENCODERS = {'h264': ('libx264', {'tune': 'zerolatency'}),
            'h265': ('libx265', {'x265-params': 'bframes=0:log-level=error'}),
            'av1': ('libsvtav1', {'svtav1-params': 'pred-struct=1'})}


@pytest.fixture(autouse=True)
def require_av():
    """Skip per test: ROS launch_testing plugins drop the session on a module skip."""
    pytest.importorskip('av')


def pixels(j):
    rows, cols = np.mgrid[:48, :64]
    return np.stack([rows * 4 + 5 * j, cols * 3, np.full_like(rows, 8 * j)], -1).astype(np.uint8)


def encode(codec):
    """Return (packet bytes, keyframe) of FRAMES frames without B-frames."""
    import av
    name, options = ENCODERS[codec]
    try:
        context = av.CodecContext.create(name, 'w')
    except av.codec.codec.UnknownCodecError:
        pytest.skip(f'PyAV has no {name}')
    context.width, context.height, context.pix_fmt = 64, 48, 'yuv420p'
    context.time_base, context.framerate = Fraction(1, 10), 10
    context.gop_size, context.max_b_frames = 4, 0
    context.options = options
    packets = []
    for j in range(FRAMES):
        frame = av.VideoFrame.from_ndarray(pixels(j), format='rgb24')
        frame.pts = j
        packets += context.encode(frame)
    packets += context.encode(None)
    return [(bytes(packet), packet.is_keyframe) for packet in packets]


def decode_all(codec, packets):
    import av
    context = av.CodecContext.create({'h265': 'hevc', 'av1': 'libdav1d'}.get(codec, codec), 'r')
    frames = [f for data, _ in packets for f in context.decode(av.Packet(data))]
    return [f.to_ndarray(format='rgb24') for f in frames + context.decode(None)]


def cdr(*fields):
    """Serialize (struct code or 'str' or 'bytes', value) fields as little-endian XCDR1."""
    out = bytearray()
    for kind, value in fields:
        if kind == 'str':
            kind, value = 'bytes', value.encode() + b'\0'
        if kind == 'bytes':
            out += b'\0' * (-len(out) % 4) + struct.pack('<I', len(value)) + value
        else:
            size = struct.calcsize(kind)
            out += b'\0' * (-len(out) % size) + struct.pack('<' + kind, value)
    return b'\0\1\0\0' + bytes(out)


@pytest.fixture
def video_bag(write_bag, tmp_path, monkeypatch):
    """Return a writer of a Foxglove and an FFMPEGPacket topic of one stream."""
    import os
    prefix = tmp_path / 'prefix' / 'share'
    for typename, text in DEFINITIONS.items():
        package, _, name = typename.split('/')
        index = prefix / 'ament_index/resource_index/rosidl_interfaces'
        index.mkdir(parents=True, exist_ok=True)
        (index / package).write_text(f'msg/{name}.msg\n')
        (prefix / package / 'msg').mkdir(parents=True)
        (prefix / package / 'msg' / f'{name}.msg').write_text(text)
    monkeypatch.setenv('AMENT_PREFIX_PATH',
                       f"{prefix.parent}:{os.environ['AMENT_PREFIX_PATH']}")

    def write(codec, packets):
        from sensor_msgs.msg import CameraInfo
        info = CameraInfo(width=64, height=48)
        info.header.frame_id = 'camera'
        events = [('/cam/camera_info', START, info)]
        for j, (data, key) in enumerate(packets):
            sec, nanosec = divmod(START + j * STEP_NS, 1_000_000_000)
            events.append(('/cam/image_raw/foxglove', START + j * STEP_NS + 1_000_000, (
                FOXGLOVE, cdr(('i', sec), ('I', nanosec), ('str', 'camera'),
                              ('bytes', data), ('str', codec)))))
            events.append(('/cam/image_raw/ffmpeg', START + j * STEP_NS + 2_000_000, (
                FFMPEG, cdr(('i', sec), ('I', nanosec + 5_000_000), ('str', 'camera'),
                            ('i', 64), ('i', 48), ('str', f'{codec};yuv420p;rgb8;rgb8'),
                            ('Q', j), ('B', int(key)), ('?', False), ('bytes', data)))))
        events.sort(key=lambda event: event[1])
        return write_bag(events)

    return write


@pytest.mark.parametrize('codec', ['h264', 'h265', 'av1'])
def test_read_video(video_bag, codec):
    import rosbag2_dataset as rd
    packets = encode(codec)
    expected = decode_all(codec, packets)
    assert len(expected) == FRAMES
    keys = [key for _, key in packets]
    assert keys[0] and not all(keys)
    with rd.open(video_bag(codec, packets)) as bag:
        for topic in ('/cam/image_raw/foxglove', '/cam/image_raw/ffmpeg'):
            video = bag.video(topic)
            assert video.codec == {'h265': 'hevc'}.get(codec, codec) and len(video) == FRAMES
            assert [video.is_keyframe(i) for i in range(FRAMES)] == keys
            for index in [5, 6, 7, 0, 11, 2, 10, 3, -1]:
                time, frame = video[index]
                assert time == int(video.times[index])
                np.testing.assert_array_equal(frame, expected[index])
            assert np.abs(video[video.nearest(START + 4 * STEP_NS)][1].astype(np.int16)
                          - pixels(4)).mean() < 8
        stamps = bag.topic('/cam/image_raw/foxglove').stamps('header')
        np.testing.assert_array_equal(stamps, START + np.arange(FRAMES) * STEP_NS)
    late = [(data, key) for data, key in packets[1:]]
    with rd.open(video_bag(codec, late)) as bag:
        with pytest.raises(ValueError, match='no keyframe'):
            bag.video('/cam/image_raw/foxglove')[0]


@pytest.mark.parametrize('codec', ['h264', 'av1'])
def test_export_video(video_bag, tmp_path, codec):
    pytest.importorskip('pyarrow')
    pytest.importorskip('PIL')
    import pyarrow.parquet as pq
    from rosbag2_dataset.export import export
    from rosbag2_dataset.torch import SceneDataset
    packets = encode(codec)
    expected = decode_all(codec, packets)
    bag = video_bag(codec, packets)
    first_key = max(i for i, (_, key) in enumerate(packets) if key and i <= 5)
    spec = {'reference': 'front', 'scene_frames': 5, 'time': 'header',
            'sensors': {'front': {'topic': '/cam/image_raw/foxglove', 'store': 'mp4'},
                        'back': {'topic': '/cam/image_raw/ffmpeg', 'store': 'png',
                                 'max_ms': 20}}}
    scenes = export(spec, bag, tmp_path / 'out')
    assert len(scenes) == 3
    columns = pq.read_table(scenes[1] / 'frames.parquet').to_pydict()
    assert columns['front'] == [5 - first_key + i for i in range(5)]
    assert columns['back_offset_ns'] == [5_000_000] * 5
    meta = json.loads((scenes[1] / 'scene.json').read_text())['sensors']
    assert meta['front']['video']['remux'] and meta['front']['video']['fps'] == 10
    assert meta['front']['video']['frames'] == 10 - first_key
    assert meta['front']['codec'] == codec and meta['front']['frame_id'] == 'camera'
    assert meta['back']['camera']['topic'] == '/cam/camera_info'
    with SceneDataset(tmp_path / 'out') as samples:
        for index in [7, 5, 6, 0, 11, 9, 1]:
            np.testing.assert_array_equal(samples[index]['front'], expected[index])
            np.testing.assert_array_equal(samples[index]['back'], expected[index])
    spec['sensors']['front'].update(codec='libx264', crf=10)
    spec['sensors']['back']['store'] = 'npy'
    export(spec, bag, tmp_path / 'encoded')
    with SceneDataset(tmp_path / 'encoded') as samples:
        sample = samples[6]
        assert np.abs(sample['front'].astype(np.int16) - expected[6]).mean() < 3
        np.testing.assert_array_equal(sample['back'], expected[6])
