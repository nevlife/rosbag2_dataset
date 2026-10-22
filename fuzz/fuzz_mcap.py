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

"""Fuzz MCAP indexing, chunk access, and message decoding through Bag."""

import sys
import types

import atheris
import numpy  # noqa: F401

with atheris.instrument_imports(include=['rosbag2_dataset']):
    from rosbag2_dataset import _mcap
    from rosbag2_dataset import bag

# Contract: a corrupt file raises ValueError; a truncated one yields its complete records.
MESSAGES = 16


class _Buffer(bytes):

    def close(self):
        pass


class _Source:

    def open(self, mode):  # noqa: A003
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def fileno(self):
        return -1


def test_one_input(data):
    _mcap.mmap = types.SimpleNamespace(ACCESS_READ=None, mmap=lambda *_, **__: _Buffer(data))
    bag.McapFile = lambda path, size: _mcap.McapFile(_Source(), size)
    try:
        with bag.Bag('fuzz.mcap', chunk_cache_size=2) as reader:
            for name in reader.topics:
                topic = reader.topic(name)
                for index in range(min(len(topic), MESSAGES)):
                    topic.raw(index)
                    topic[index]
    except ValueError:
        pass


if __name__ == '__main__':
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()
