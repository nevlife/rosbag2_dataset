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

"""Create reference bags using the ROS writer, without starting DDS."""

import struct

import numpy as np
import pytest


@pytest.fixture(scope='session')
def ros():
    pytest.importorskip('rclpy')
    return pytest.importorskip('rosbag2_py')


def make_image(encoding='rgb8', bigendian=False, width=5, height=3):
    from sensor_msgs.msg import Image
    kinds = {'rgb8': ('u1', 3), 'bgr8': ('u1', 3), 'rgba8': ('u1', 4),
             'bgra8': ('u1', 4), 'mono8': ('u1', 1), 'mono16': ('u2', 1),
             '16UC1': ('u2', 1), '32FC1': ('f4', 1)}
    kind, channels = kinds[encoding]
    dtype = np.dtype(('>' if bigendian else '<') + kind)
    shape = (height, width, channels) if channels > 1 else (height, width)
    expected = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    step = width * channels * dtype.itemsize + 7
    buffer = bytearray([0xee] * (height * step))
    for row in range(height):
        data = expected[row].tobytes()
        buffer[row * step:row * step + len(data)] = data
    message = Image(height=height, width=width, encoding=encoding,
                    is_bigendian=int(bigendian), step=step, data=bytes(buffer))
    message.header.frame_id = 'camera'
    message.header.stamp.sec = 12
    return message, expected


def make_cloud(bigendian=False):
    from sensor_msgs.msg import PointCloud2, PointField
    fields = [PointField(name='x', offset=0, datatype=7, count=1),
              PointField(name='y', offset=4, datatype=7, count=1),
              PointField(name='z', offset=8, datatype=8, count=1),
              PointField(name='intensity', offset=16, datatype=4, count=1),
              PointField(name='normal', offset=20, datatype=3, count=3)]
    width, height, point_step, row_step = 3, 2, 32, 103
    buffer = bytearray([0xee] * (row_step * height))
    for row in range(height):
        for col in range(width):
            i = row * width + col
            pos = row * row_step + col * point_step
            struct.pack_into(('>' if bigendian else '<') + 'ffdH', buffer,
                             pos, i + 0.25, -i - 0.5, i + 0.75, i + 100)
            struct.pack_into(('>' if bigendian else '<') + 'hhh', buffer,
                             pos + 20, -i, i + 1, i + 2)
    return PointCloud2(height=height, width=width, fields=fields,
                       is_bigendian=bigendian, point_step=point_step,
                       row_step=row_step, data=bytes(buffer), is_dense=True)


@pytest.fixture
def write_bag(ros, tmp_path):
    from rclpy.serialization import serialize_message

    def write(events, *, options='', split_size=0):
        name = f'bag_{len(list(tmp_path.glob("bag_*")))}'
        path = tmp_path / name
        config = tmp_path / (name + '.yaml')
        defaults = 'chunkSize: 1024\n'
        if 'compression:' not in options:
            defaults += 'compression: None\n'
        config.write_text(defaults + options)
        writer = ros.SequentialWriter()
        writer.open(ros.StorageOptions(uri=str(path), storage_id='mcap',
                                       storage_config_uri=str(config),
                                       max_bagfile_size=split_size),
                    ros.ConverterOptions('', ''))
        known = set()
        for topic, time, message in events:
            if topic not in known:
                cls = type(message)
                typename = cls.__module__.split('.')[0] + '/msg/' + cls.__name__
                writer.create_topic(ros.TopicMetadata(id=0, name=topic, type=typename,
                                                      serialization_format='cdr'))
                known.add(topic)
            writer.write(topic, serialize_message(message), time)
        del writer
        return path

    return write


@pytest.fixture
def reference_events(ros):
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
    from geometry_msgs.msg import TransformStamped
    from sensor_msgs.msg import Imu
    from std_msgs.msg import String
    from test_msgs.msg import Arrays, BasicTypes, BoundedSequences, Constants, Defaults
    from test_msgs.msg import MultiNested, Strings, UnboundedSequences
    from tf2_msgs.msg import TFMessage

    messages = []
    for encoding in ('rgb8', 'bgr8', 'rgba8', 'bgra8', 'mono8', 'mono16', '16UC1', '32FC1'):
        for endian in (False, True):
            messages.append(('/image', make_image(encoding, endian)[0]))
    messages.extend([('/cloud', make_cloud()), ('/cloud', make_cloud(True))])
    imu = Imu()
    imu.header.stamp.sec = -5
    imu.header.stamp.nanosec = 123456789
    imu.header.frame_id = 'imu'
    imu.orientation.x, imu.orientation.w = 0.5, 0.75
    imu.angular_velocity.y = -3.125
    imu.linear_acceleration.z = 9.81
    imu.orientation_covariance = np.arange(9, dtype=np.float64)
    imu.angular_velocity_covariance = -np.arange(9, dtype=np.float64)
    imu.linear_acceleration_covariance = np.arange(9, dtype=np.float64) * 1.25
    messages.extend([('/imu', imu), ('/text', String(data='héllo\nworld'))])
    diag = DiagnosticArray(status=[
        DiagnosticStatus(level=b'\x01', name='motor', message='warm', hardware_id='42',
                         values=[KeyValue(key='temperature', value='41.5'),
                                 KeyValue(key='voltage', value='24')]),
        DiagnosticStatus(name='empty', values=[])])
    messages.append(('/diagnostics', diag))
    transform = TransformStamped(child_frame_id='child')
    transform.header.frame_id = 'world'
    transform.header.stamp.sec = 7
    transform.transform.translation.x = 1.25
    transform.transform.rotation.w = 1.0
    messages.extend([('/tf', TFMessage(transforms=[transform, transform])),
                     ('/tf', TFMessage(transforms=[]))])
    basic = BasicTypes(bool_value=True, byte_value=b'\xfe', char_value=128,
                       float32_value=1.25, float64_value=-3.5, int8_value=-128,
                       uint8_value=255, int16_value=-32768, uint16_value=65535,
                       int32_value=-2147483648, uint32_value=4294967295,
                       int64_value=-9223372036854775808, uint64_value=18446744073709551615)
    seq = UnboundedSequences(basic_types_values=[basic, basic],
                             constants_values=[Constants(), Constants()],
                             defaults_values=[Defaults()])
    bounded = BoundedSequences()
    for message in (seq, bounded):
        for field, kind in message.get_fields_and_field_types().items():
            if not field.endswith('_values'):
                continue
            base = field[:-7]
            if base in ('bool', 'byte', 'char', 'float32', 'float64', 'int8', 'uint8',
                        'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64'):
                value = getattr(basic, base + '_value')
                setattr(message, field, [value, value])
        message.string_values = ['one', '', 'three']
        message.alignment_check = 987654321
    nested = MultiNested(bounded_sequence_of_unbounded_sequences=[seq],
                         unbounded_sequence_of_bounded_sequences=[bounded, bounded])
    for message in (basic, Arrays(), seq, bounded, nested, Constants(), Defaults(),
                    Strings(string_value='unicode café')):
        messages.append(('/test/' + type(message).__name__, message))
    return [(topic, 1_000_000 + i * 100, message)
            for i, (topic, message) in enumerate(messages)]
