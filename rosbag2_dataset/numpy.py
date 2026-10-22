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

"""Construct strided read-only NumPy views of sensor message buffers."""

import numpy as np


_ENCODINGS = {
    'rgb8': ('u1', 3), 'bgr8': ('u1', 3),
    'rgba8': ('u1', 4), 'bgra8': ('u1', 4),
    'mono8': ('u1', 1), 'mono16': ('u2', 1),
    '16UC1': ('u2', 1), '32FC1': ('f4', 1),
}
_POINT_TYPES = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2',
                5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}


def image_to_numpy(message):
    """Return an HxW or HxWxC view, preserving encoding, endianness, and padding."""
    if message.encoding not in _ENCODINGS:
        raise ValueError(f'Unsupported image encoding: {message.encoding!r}')
    dtype, channels = _ENCODINGS[message.encoding]
    dtype = np.dtype(('>' if message.is_bigendian else '<') + dtype)
    pixel_size = dtype.itemsize * channels
    if message.step < message.width * pixel_size:
        raise ValueError('Image step is smaller than its packed row size')
    shape = (message.height, message.width)
    strides = (message.step, pixel_size)
    if channels != 1:
        shape += (channels,)
        strides += (dtype.itemsize,)
    result = np.ndarray(shape, dtype=dtype, buffer=message.data, strides=strides)
    result.flags.writeable = False
    return result


def pointcloud2_to_numpy(message):
    """Return an organized HxW structured view, preserving row and point padding."""
    endian = '>' if message.is_bigendian else '<'
    names, formats, offsets = [], [], []
    for field in message.fields:
        if field.datatype not in _POINT_TYPES:
            raise ValueError(f'Unsupported PointField datatype: {field.datatype}')
        dtype = np.dtype(endian + _POINT_TYPES[field.datatype])
        names.append(field.name)
        formats.append(dtype if field.count == 1 else (dtype, (field.count,)))
        offsets.append(field.offset)
    dtype = np.dtype({'names': names, 'formats': formats, 'offsets': offsets,
                      'itemsize': message.point_step})
    if message.row_step < message.width * message.point_step:
        raise ValueError('PointCloud2 row_step is smaller than its packed row size')
    result = np.ndarray((message.height, message.width), dtype=dtype,
                        buffer=message.data, strides=(message.row_step, message.point_step))
    result.flags.writeable = False
    return result
