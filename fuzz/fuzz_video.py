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

"""Fuzz codec names and keyframe detection; input is a flags byte and a packet."""

import sys
import types

import atheris
import numpy as np

with atheris.instrument_imports(include=['rosbag2_dataset']):
    from rosbag2_dataset.video import codec_of
    from rosbag2_dataset.video import is_keyframe

# Contract: is_keyframe returns a bool for any packet; codec_of raises only ValueError.


def test_one_input(data):
    if not data:
        return
    packet = np.frombuffer(data, np.uint8, offset=1)
    name = data[1:17].decode('latin-1')
    for message in (
        types.SimpleNamespace(format=name, data=packet),
        types.SimpleNamespace(encoding=name, flags=data[0], data=packet),
    ):
        try:
            codec_of(message)
        except ValueError:
            pass
        for codec in ('h264', 'hevc', 'av1'):
            assert type(is_keyframe(message, codec)) is bool


if __name__ == '__main__':
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()
