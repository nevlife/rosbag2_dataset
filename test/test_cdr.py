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

"""Exercise both CDR byte orders and generated schema grammar without ROS."""

import struct

import numpy as np
import pytest
from rosbag2_dataset._cdr import compile_decoder


class CdrBuilder:

    def __init__(self, endian):
        self.endian = endian
        self.data = bytearray()

    def put(self, fmt, *values):
        size = struct.calcsize(fmt)
        self.data.extend(b'\x00' * (-len(self.data) % size))
        self.data.extend(struct.pack(self.endian + fmt, *values))

    def string(self, text):
        data = text.encode() + b'\x00'
        self.put('I', len(data))
        self.data.extend(data)

    def wstring(self, text):
        units = np.frombuffer(text.encode('utf-16-le'), dtype='<u2')
        self.put('I', len(units))
        for unit in units:
            self.put('I', int(unit))

    def payload(self):
        return (b'\x00\x01\x00\x00' if self.endian == '<' else b'\x00\x00\x00\x00') + self.data


@pytest.mark.parametrize('endian', ['<', '>'])
def test_generated_cdr(endian):
    schema = """bool flag true
byte octet
char character
int8 signed_byte
uint8 unsigned_byte
int16 signed_short
uint16 unsigned_short
int32 signed_int
uint32 unsigned_int
int64 signed_long
uint64 unsigned_long
float32 single
float64 double
string<=30 text "default"
wstring wide
float64[3] fixed
uint16[<=8] bounded
uint32[] empty
Child[] children
uint8 tail
uint8 IGNORED=42
================================================================================
MSG: demo/Child
string name
int32[] values
"""
    builder = CdrBuilder(endian)
    fields = [('?', True), ('B', 254), ('B', 200), ('b', -128), ('B', 255),
              ('h', -32000), ('H', 65000), ('i', -1234567), ('I', 3456789),
              ('q', -1234567890123), ('Q', 1234567890123456), ('f', 1.25), ('d', -9.5)]
    for fmt, value in fields:
        builder.put(fmt, value)
    builder.string('héllo')
    builder.wstring('\u65e5\u672c\u8a9e\U0001d11e')
    for value in (1.0, 2.5, -3.75):
        builder.put('d', value)
    builder.put('I', 3)
    for value in (0, 300, 65535):
        builder.put('H', value)
    builder.put('I', 0)
    builder.put('I', 2)
    for name, values in [('first', [1, -2]), ('second', [])]:
        builder.string(name)
        builder.put('I', len(values))
        for value in values:
            builder.put('i', value)
    builder.put('B', 127)
    decoder = compile_decoder('demo/msg/Root', schema)
    assert decoder is compile_decoder('demo/msg/Root', schema)
    payload = builder.payload()
    message = decoder(payload)
    expected = dict(zip(message._fields[:13], [value for _, value in fields]))
    expected['octet'] = b'\xfe'
    for name, value in expected.items():
        assert getattr(message, name) == value
    assert message.text == 'héllo'
    assert message.wide == '\u65e5\u672c\u8a9e\U0001d11e'
    np.testing.assert_array_equal(message.fixed, [1, 2.5, -3.75])
    np.testing.assert_array_equal(message.bounded, [0, 300, 65535])
    assert len(message.empty) == 0
    assert message.children[0].name == 'first'
    np.testing.assert_array_equal(message.children[0].values, [1, -2])
    assert message.tail == 127
    assert message.fixed.dtype.byteorder in (endian, '=' if endian == '<' else endian)
    assert not message.fixed.flags.writeable
    assert message.fixed.base.obj is payload
    assert message.to_dict()['children'] == [
        {'name': 'first', 'values': [1, -2]}, {'name': 'second', 'values': []}]


@pytest.mark.parametrize('endian', ['<', '>'])
def test_empty_sequence_alignment(endian):
    builder = CdrBuilder(endian)
    builder.put('I', 0)
    builder.put('B', 99)
    message = compile_decoder('demo/Empty', 'float64[] values\nuint8 tail')(builder.payload())
    assert message.tail == 99
    assert len(message.values) == 0


def test_reject_unsupported_serialization():
    decoder = compile_decoder('demo/One', 'uint8 value')
    with pytest.raises(ValueError, match='XCDR1'):
        decoder(b'\x00\x07\x00\x00\x00')
    with pytest.raises(ValueError, match='Missing dependent'):
        compile_decoder('demo/One', 'Missing value')
    with pytest.raises(ValueError, match='Invalid ros2msg'):
        compile_decoder('demo/One', 'invalid')
    with pytest.raises(ValueError, match='Unsupported ros2msg'):
        compile_decoder('demo/One', 'uint8[broken] value')


@pytest.mark.parametrize('typename', ['WStrings', 'Empty'])
def test_ros_wstrings_and_empty(ros, typename):
    from pathlib import Path
    from rclpy.serialization import deserialize_message, serialize_message
    from rosidl_runtime_py import get_interface_path
    from rosidl_runtime_py.utilities import get_message
    from test_ros_reference import compare
    name = 'test_msgs/msg/' + typename
    cls = get_message(name)
    message = cls()
    if typename == 'WStrings':
        message.wstring_value = 'wide \u65e5\u672c\u8a9e'
        message.array_of_wstrings = ['a', '', 'é']
        message.bounded_sequence_of_wstrings = ['abc', '\u65e5\u672c\u8a9e']
        message.unbounded_sequence_of_wstrings = ['', 'end']
    schema = Path(get_interface_path(name)).read_text()
    payload = serialize_message(message)
    compare(compile_decoder(name, schema)(payload), deserialize_message(payload, cls))
