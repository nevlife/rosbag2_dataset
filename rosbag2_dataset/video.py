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

"""Decode H.264, H.265, and AV1 image topics recorded as video packets."""

from collections import deque

import numpy as np

COMPRESSED_VIDEO = 'foxglove_msgs/msg/CompressedVideo'
FFMPEG_PACKET = 'ffmpeg_image_transport_msgs/msg/FFMPEGPacket'
VIDEO_TYPES = (COMPRESSED_VIDEO, FFMPEG_PACKET)
# codec: (PyAV decoder, raw bitstream demuxer)
CODECS = {'h264': ('h264', 'h264'), 'hevc': ('hevc', 'hevc'), 'av1': ('libdav1d', 'obu')}
_START = b'\x00\x00\x01'
# Freeing a threaded libdav1d decoder that still holds queued frames hangs (PyAV 19).
DECODER_THREADS = 1


def codec_of(message):
    """Return h264, hevc, or av1 from a CompressedVideo format or FFMPEGPacket encoding."""
    name = (
        message.format if hasattr(message, 'format') else message.encoding.split(';')[0]
    ).lower()
    if 'hevc' in name or '265' in name:
        return 'hevc'
    if '264' in name:
        return 'h264'
    if 'av1' in name:
        return 'av1'
    raise ValueError(f'Unsupported video codec {name!r}')


def _annexb_key(data, nal_type, vcl, keys):
    start = data.find(_START)
    while 0 <= start < len(data) - 3:
        kind = nal_type(data[start + 3])
        if kind in vcl:
            return kind in keys
        start = data.find(_START, start + 3)
    return False


def _av1_key(data):
    position = 0
    while position < len(data):
        header = data[position]
        if (header >> 3) & 15 == 1:
            return True
        position += 1 + ((header >> 2) & 1)
        if not header & 2:
            return False
        size = shift = 0
        end = position + 8
        for byte in data[position:end]:
            position += 1
            size |= (byte & 0x7F) << shift
            shift += 7
            if not byte & 0x80:
                break
        else:
            return False
        position += size
    return False


def is_keyframe(message, codec):
    """
    Return whether decoding can start at this message.

    FFMPEGPacket carries the packet's key flag. CompressedVideo is checked for an
    H.264 IDR or H.265 IRAP picture, or an AV1 sequence header, which Foxglove
    requires in every keyframe message.
    """
    if not hasattr(message, 'format'):
        return bool(message.flags & 1)
    data = message.data.tobytes()
    if codec == 'h264':
        return _annexb_key(data, lambda byte: byte & 0x1F, range(1, 6), (5,))
    if codec == 'hevc':
        return _annexb_key(data, lambda byte: (byte >> 1) & 0x3F, range(32), range(16, 24))
    return _av1_key(data)


class VideoTopic:
    """
    Decode a video topic into RGB frames by message index.

    Each message holds one frame of a stream without B-frames. Random access
    decodes from the preceding keyframe; sequential access continues decoding.
    """

    def __init__(self, topic):
        self.topic = topic
        self.times = topic.times
        self.codec = codec_of(topic[0][1])
        self._keys = np.full(len(topic), -1, dtype=np.int8)
        self._context = None
        self._frames = deque()
        self._fed = self._out = 0

    def __len__(self):
        return len(self.topic)

    def nearest(self, time_ns):
        """Return the nearest message index; ties select the earlier timestamp."""
        return self.topic.nearest(time_ns)

    def is_keyframe(self, index):
        if self._keys[index] < 0:
            self._keys[index] = is_keyframe(self.topic[index][1], self.codec)
        return bool(self._keys[index])

    def keyframe(self, index):
        """Return the index of the last keyframe at or before a message."""
        for key in range(index, -1, -1):
            if self.is_keyframe(key):
                return key
        raise ValueError(f'{self.topic.name}: no keyframe at or before message {index}')

    def frame(self, index):
        """Return the av.VideoFrame decoded from a message."""
        import av

        index = range(len(self))[index]
        if (
            self._context is None
            or index < self._out
            or index > self._out
            and self.keyframe(index) > self._out
        ):
            self._context = av.CodecContext.create(CODECS[self.codec][0], 'r')
            self._context.thread_count = DECODER_THREADS
            self._frames.clear()
            self._fed = self._out = self.keyframe(index)
        while True:
            while self._frames:
                frame = self._frames.popleft()
                self._out += 1
                if self._out > index:
                    return frame
            packet = av.Packet(self.topic[self._fed][1].data) if self._fed < len(self) else None
            self._frames.extend(self._context.decode(packet))
            self._fed += 1

    def __getitem__(self, index):
        """Return (log_time_ns, HxWx3 uint8 RGB pixels) of a message."""
        return int(self.times[index]), self.frame(index).to_ndarray(format='rgb24')
