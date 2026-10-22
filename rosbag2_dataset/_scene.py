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

"""Convert sensor messages to arrays and read stored scene samples."""

from fractions import Fraction
import io

import numpy as np

from ._cdr import compile_decoder
from ._mcap import McapFile
from .numpy import image_to_numpy, pointcloud2_to_numpy
from .video import DECODER_THREADS


IMAGE = 'sensor_msgs/msg/Image'
COMPRESSED_IMAGE = 'sensor_msgs/msg/CompressedImage'
POINT_CLOUD = 'sensor_msgs/msg/PointCloud2'
SIXTEEN_BIT = ('mono16', '16UC1')


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


def sample_array(message, sensor):
    """Return the array an exported sensor entry stores for a decoded message."""
    if sensor['type'] == IMAGE:
        return image_pixels(message)
    if sensor['type'] == COMPRESSED_IMAGE:
        return compressed_pixels(message)
    if sensor['type'] == POINT_CLOUD:
        return cloud_points(message, sensor['fields'], sensor['finite'])
    return message


class SceneFiles:
    """Read stored sensor values; MCAP files and videos stay open per process."""

    def __init__(self):
        self._mcaps = {}
        self._videos = {}

    def read(self, scene, sensor, row):
        """Return the array of one sensor in a frames.parquet row."""
        store = sensor['store']
        name = sensor['name']
        if store == 'ref':
            path = scene['bag_dir'] / row[name + '_file']
            if path not in self._mcaps:
                self._mcaps[path] = McapFile(path, index=False)
            payload = self._mcaps[path].raw(row[name + '_chunk'], row[name + '_record'])
            schema = sensor['schema']
            message = compile_decoder(schema['name'], schema['text'])(payload)
            return sample_array(message, sensor)
        if store == 'mp4':
            path = scene['path'] / sensor['video']['file']
            if path not in self._videos:
                self._videos[path] = _Video(path, sensor['video'])
            return self._videos[path].frame(row[name])
        path = scene['path'] / row[name]
        if store == 'npy':
            return np.load(path)
        from PIL import Image
        with Image.open(path) as image:
            pixels = np.asarray(image)
        return pixels.astype(np.uint16) if sensor.get('encoding') in SIXTEEN_BIT else pixels

    def close(self):
        for file in self._mcaps.values():
            file.close()
        for video in self._videos.values():
            video.close()
        self._mcaps.clear()
        self._videos.clear()


class _Video:
    """Decode frames by index, continuing forward when access is sequential."""

    def __init__(self, path, video):
        import av
        self._container = av.open(str(path))
        self._stream = self._container.streams.video[0]
        self._stream.codec_context.thread_count = DECODER_THREADS
        self._fps = Fraction(video['fps'])
        self._gop = video['gop']
        self._format = video['format']
        self._frames = None
        self._next = None

    def frame(self, index):
        if self._next is None or not self._next <= index < self._next + self._gop:
            seconds = Fraction(index) / self._fps
            self._container.seek(int(seconds / self._stream.time_base), stream=self._stream,
                                 backward=True)
            self._frames = self._container.decode(self._stream)
        for frame in self._frames:
            position = round(frame.time * self._fps)
            if position >= index:
                self._next = position + 1
                return frame.to_ndarray(format=self._format)
        raise IndexError(index)

    def close(self):
        self._container.close()
