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

"""Expose indexed topics, time ordered reads, and nearest-time synchronization."""

from dataclasses import dataclass
import heapq
import operator
from pathlib import Path

import numpy as np

from ._cdr import compile_decoder
from ._mcap import McapFile


@dataclass(frozen=True)
class TopicInfo:
    """Describe a topic's type and log-time range across all split files."""

    type: str  # noqa: A003
    count: int
    start_ns: int | None
    end_ns: int | None


class Topic:
    """Provide random access to messages in stable ascending log-time order."""

    def __init__(self, bag, name, typename, entries):
        self._bag = bag
        self.name = name
        self.type = typename
        entries.sort(key=lambda row: row[0])
        self._entries = np.asarray(entries, dtype=np.int64).reshape(-1, 5)
        self._entries.flags.writeable = False
        self.times = self._entries[:, 0]
        self._decoders = {}

    def __len__(self):
        return len(self._entries)

    def _entry(self, index):
        self._bag._check_open()
        return self._entries[operator.index(index)]

    def raw(self, index):
        """Return (log_time_ns, read-only CDR memoryview)."""
        time, file_id, channel, chunk, offset = self._entry(index)
        return int(time), self._bag._files[file_id].raw(int(chunk), int(offset))

    def __getitem__(self, index):
        time, file_id, channel, chunk, offset = self._entry(index)
        file = self._bag._files[file_id]
        key = (int(file_id), int(channel))
        if key not in self._decoders:
            info = file.channels[channel]
            schema = file.schemas[info.schema_id]
            if info.encoding != 'cdr' or schema.encoding != 'ros2msg':
                raise ValueError('Decoding requires cdr messages with a ros2msg schema')
            self._decoders[key] = compile_decoder(schema.name, schema.text)
        payload = file.raw(int(chunk), int(offset))
        return int(time), self._decoders[key](payload)

    def nearest(self, time_ns):
        """Return the nearest index; ties select the earlier timestamp."""
        return _nearest(self.times, time_ns)


def _nearest(times, time):
    if not len(times):
        raise ValueError('Cannot find the nearest message in an empty topic')
    index = int(np.searchsorted(times, time, side='left'))
    if index == len(times):
        index -= 1
    elif index and int(times[index]) - time >= time - int(times[index - 1]):
        index -= 1
    return int(np.searchsorted(times, times[index], side='left'))


class Synced:
    """Hold random-access tuples of topic indices in reference-time order."""

    def __init__(self, topics, indices):
        self.topics = tuple(topics)
        self.indices = np.asarray(indices, dtype=np.int64).reshape(-1, len(topics))
        self.indices.flags.writeable = False

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [tuple(map(int, row)) for row in self.indices[index]]
        return tuple(map(int, self.indices[operator.index(index)]))


class Bag:
    """Read a single MCAP file or all MCAP splits in a bag directory."""

    def __init__(self, path, *, chunk_cache_size=4):
        if chunk_cache_size < 1:
            raise ValueError('chunk_cache_size must be at least 1')
        path = Path(path)
        paths = sorted(path.glob('*.mcap')) if path.is_dir() else [path]
        if not paths:
            raise FileNotFoundError(f'No .mcap files found in {path}')
        self._files = []
        self._closed = False
        try:
            for split in paths:
                self._files.append(McapFile(split, chunk_cache_size))
            entries = {}
            types = {}
            for file_id, file in enumerate(self._files):
                for channel in file.channels.values():
                    schema = file.schemas.get(channel.schema_id)
                    typename = schema.name if schema else ''
                    if channel.topic in types and types[channel.topic] != typename:
                        raise ValueError(f'Conflicting types for topic {channel.topic}')
                    types[channel.topic] = typename
                    entries.setdefault(channel.topic, [])
                for channel_id, time, chunk, offset in file.entries:
                    channel = file.channels[channel_id]
                    entries[channel.topic].append((time, file_id, channel_id, chunk, offset))
                file.entries.clear()
            self._topics = {name: Topic(self, name, types[name], rows)
                            for name, rows in entries.items()}
            self.topics = {
                name: TopicInfo(topic.type, len(topic),
                                int(topic.times[0]) if len(topic) else None,
                                int(topic.times[-1]) if len(topic) else None)
                for name, topic in self._topics.items()}
        except BaseException:
            self.close()
            raise

    def _check_open(self):
        if self._closed:
            raise ValueError('Bag is closed')

    def topic(self, name):
        """Return an indexed topic."""
        self._check_open()
        return self._topics[name]

    def read(self, topics=None, start_ns=None, end_ns=None):
        """Yield messages by log time in the half-open interval [start, end)."""
        self._check_open()
        if isinstance(topics, str):
            topics = [topics]
        selected = [self.topic(name) for name in
                    (self._topics if topics is None else dict.fromkeys(topics))]

        def times(topic):
            start = 0 if start_ns is None else int(np.searchsorted(topic.times, start_ns))
            end = len(topic) if end_ns is None else int(np.searchsorted(topic.times, end_ns))
            return ((int(topic.times[i]), topic.name, i) for i in range(start, end))

        for _, name, index in heapq.merge(*(times(topic) for topic in selected)):
            time, message = self._topics[name][index]
            yield name, time, message

    def synced(self, topics, reference=None, tolerance_ns=0, *, time_source='log'):
        """Align nearest times, omitting reference rows outside the tolerance."""
        topics = tuple(topics)
        if not topics or len(set(topics)) != len(topics):
            raise ValueError('topics must contain distinct topic names')
        reference = topics[0] if reference is None else reference
        if reference not in topics:
            raise ValueError('reference must be one of topics')
        if tolerance_ns < 0:
            raise ValueError('tolerance_ns must be nonnegative')
        if time_source not in ('log', 'header'):
            raise ValueError("time_source must be 'log' or 'header'")
        clocks = {}
        original_clocks = {}
        orders = {}
        for name in topics:
            topic = self.topic(name)
            if time_source == 'log':
                clock = topic.times
            else:
                clock = np.array([message.header.stamp.sec * 1_000_000_000 +
                                  message.header.stamp.nanosec for _, message in topic],
                                 dtype=np.int64)
            original_clocks[name] = clock
            orders[name] = np.argsort(clock, kind='stable')
            clocks[name] = clock[orders[name]]
        rows = []
        for ref_index in range(len(self.topic(reference))):
            ref_time = int(original_clocks[reference][ref_index])
            row = []
            for name in topics:
                if name == reference:
                    row.append(ref_index)
                elif not len(clocks[name]):
                    break
                else:
                    index = _nearest(clocks[name], int(ref_time))
                    if abs(int(clocks[name][index]) - int(ref_time)) > tolerance_ns:
                        break
                    row.append(int(orders[name][index]))
            if len(row) == len(topics):
                rows.append(row)
        return Synced(topics, rows)

    def close(self):
        """Close the bag; previously exported arrays remain valid."""
        if not self._closed:
            self._closed = True
            for file in self._files:
                file.close()

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *args):
        self.close()

    def __getstate__(self):
        raise TypeError('Bag mappings cannot be pickled; use rosbag2_dataset.torch.BagDataset')
