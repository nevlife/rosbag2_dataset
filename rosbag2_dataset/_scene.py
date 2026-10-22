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

"""Convert sensor messages to arrays."""

import io

import numpy as np

from .numpy import image_to_numpy, pointcloud2_to_numpy


IMAGE = 'sensor_msgs/msg/Image'
COMPRESSED_IMAGE = 'sensor_msgs/msg/CompressedImage'
POINT_CLOUD = 'sensor_msgs/msg/PointCloud2'


def image_pixels(message):
    """Return RGB HxWx3 or HxW pixels; alpha is dropped and BGR reordered."""
    pixels = image_to_numpy(message)
    if message.encoding in ('bgr8', 'bgra8'):
        return pixels[..., 2::-1]
    if message.encoding == 'rgba8':
        return pixels[..., :3]
    return pixels.astype(pixels.dtype.newbyteorder('='), copy=False)


def compressed_pixels(message):
    """Decode a CompressedImage to RGB HxWx3 or HxW pixels."""
    from PIL import Image
    with Image.open(io.BytesIO(message.data)) as image:
        return np.asarray(image.convert('RGB') if image.mode not in ('L', 'RGB') else image)


def cloud_points(message, fields, finite):
    """Return an N x len(fields) array of one native dtype from a PointCloud2."""
    view = pointcloud2_to_numpy(message).reshape(-1)
    columns = [view[name] for name in fields]
    dtype = np.result_type(*columns).newbyteorder('=')
    points = np.stack(columns, axis=-1).astype(dtype, copy=False)
    if finite:
        points = points[np.isfinite(points).all(axis=1)]
    return points
