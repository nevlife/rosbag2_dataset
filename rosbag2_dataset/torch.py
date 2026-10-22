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

"""Provide the PyTorch map-style dataset protocol without importing torch."""

from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Mapping
import json
import operator
import os
from pathlib import Path

import numpy as np

from ._scene import SceneFiles
from .bag import Bag


class BagDataset:
    """
    Return topic-to-(time, message) dictionaries, optionally transformed.

    With sync=None, each sample contains one message in merged log-time order.
    With a sync options dictionary, each sample contains an aligned topic group.
    Use a transform/collate_fn to convert Message objects into training tensors.
    """

    def __init__(self, paths, topics, sync=None, transform=None):
        if isinstance(paths, (str, Path)):
            paths = [paths]
        self.paths = tuple(str(Path(path).resolve()) for path in paths)
        self.topics = (topics,) if isinstance(topics, str) else tuple(topics)
        if not self.topics or len(set(self.topics)) != len(self.topics):
            raise ValueError('topics must contain distinct topic names')
        self.sync = None if sync is None else dict(sync)
        self.transform = transform
        self._bags = None
        self._pid = None
        self._tables = None
        self._ends = []
        total = 0
        for path in self.paths:
            with Bag(path) as bag:
                total += len(self._table(bag))
                self._ends.append(total)

    def _table(self, bag):
        if self.sync is not None:
            return bag.synced(self.topics, **self.sync).indices
        rows = [(int(time), topic_id, index)
                for topic_id, name in enumerate(self.topics)
                for index, time in enumerate(bag.topic(name).times)]
        rows.sort()
        return np.array([(topic, index) for _, topic, index in rows],
                        dtype=np.int64).reshape(-1, 2)

    def _ensure_open(self):
        if self._pid != os.getpid() or self._bags is None:
            self.close()
            self._bags = []
            try:
                for path in self.paths:
                    self._bags.append(Bag(path))
                self._tables = [self._table(bag) for bag in self._bags]
            except BaseException:
                self.close()
                raise
            self._pid = os.getpid()

    def __len__(self):
        return self._ends[-1] if self._ends else 0

    def __getitem__(self, index):
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        self._ensure_open()
        bag_id = bisect_right(self._ends, index)
        local = index - (self._ends[bag_id - 1] if bag_id else 0)
        row = self._tables[bag_id][local]
        bag = self._bags[bag_id]
        if self.sync is None:
            topic_id, message_id = map(int, row)
            name = self.topics[topic_id]
            sample = {name: bag.topic(name)[message_id]}
        else:
            sample = {name: bag.topic(name)[int(message_id)]
                      for name, message_id in zip(self.topics, row)}
        return self.transform(sample) if self.transform else sample

    def close(self):
        """Close mappings owned by the current process."""
        if self._bags is not None:
            for bag in self._bags:
                bag.close()
        self._bags = None
        self._tables = None
        self._pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_bags=None, _tables=None, _pid=None)
        return state

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class SceneDataset:
    """
    Return one dictionary per frame of scenes written by rosbag2_dataset.export.

    A sample holds every frames.parquet column plus scene (path) and frame (row).
    Each loaded sensor replaces its column with a NumPy array, or None where the
    frame has no message. A table sensor gives its rows with timestamps in
    (previous frame, this frame] as column arrays. T_world_base becomes 4x4.

    ref sensors read the bag directory recorded in scene.json. bag_root moves it:
    a mapping from recorded to new directories, or a directory that holds the
    bags under their recorded directory names.
    """

    table_cache_size = 8

    def __init__(self, roots, sensors=None, transform=None, bag_root=None):
        import pyarrow.parquet as pq
        roots = [roots] if isinstance(roots, (str, Path)) else roots
        paths = []
        for root in map(Path, roots):
            if (root / 'scene.json').exists():
                paths.append(root.resolve())
            else:
                paths.extend(sorted(path.parent.resolve() for path in root.glob('**/scene.json')
                                    if not path.parent.name.startswith('.')))
        self.scenes = []
        self._frames = []
        self._ends = []
        total = 0
        for path in paths:
            scene = json.loads((path / 'scene.json').read_text())
            scene['path'] = path
            scene['bag_dir'] = _bag_dir(scene['source']['dir'], bag_root)
            for name, sensor in scene['sensors'].items():
                sensor['name'] = name
            self.scenes.append(scene)
            self._frames.append(pq.read_table(path / 'frames.parquet').to_pydict())
            total += scene['num_frames']
            self._ends.append(total)
        self.sensors = None if sensors is None else tuple(sensors)
        self.transform = transform
        self._files = None
        self._tables = None
        self._pid = None

    def __len__(self):
        return self._ends[-1] if self._ends else 0

    def _ensure_open(self):
        if self._pid != os.getpid() or self._files is None:
            self._files = SceneFiles()
            self._tables = OrderedDict()
            self._pid = os.getpid()

    def _table(self, scene, sensor):
        import pyarrow.parquet as pq
        key = (scene['path'], sensor['name'])
        if key not in self._tables:
            self._tables[key] = pq.read_table(scene['path'] / sensor['file'])
            if len(self._tables) > self.table_cache_size:
                self._tables.popitem(last=False)
        self._tables.move_to_end(key)
        return self._tables[key]

    def __getitem__(self, index):
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        self._ensure_open()
        scene_id = bisect_right(self._ends, index)
        frame = index - (self._ends[scene_id - 1] if scene_id else 0)
        scene = self.scenes[scene_id]
        frames = self._frames[scene_id]
        row = {name: column[frame] for name, column in frames.items()}
        sample = dict(row, scene=str(scene['path']), frame=frame)
        if row.get('T_world_base') is not None:
            sample['T_world_base'] = np.array(row['T_world_base']).reshape(4, 4)
        for name, sensor in scene['sensors'].items():
            if self.sensors is not None and name not in self.sensors:
                continue
            if sensor['store'] == 'table':
                table = self._table(scene, sensor)
                stamps = table.column('timestamp_ns').to_numpy()
                times = frames['timestamp_ns']
                low = np.searchsorted(stamps, times[frame - 1], side='right') if frame else 0
                high = np.searchsorted(stamps, times[frame], side='right')
                window = table.slice(low, max(0, high - low))
                sample[name] = {column: window.column(column).to_numpy(zero_copy_only=False)
                                for column in window.column_names}
            elif row[name + '_file' if sensor['store'] == 'ref' else name] is None:
                sample[name] = None
            else:
                sample[name] = self._files.read(scene, sensor, row)
        return self.transform(sample) if self.transform else sample

    def close(self):
        """Close bag mappings and videos owned by the current process."""
        if self._files is not None:
            self._files.close()
        self._files = None
        self._tables = None
        self._pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_files=None, _tables=None, _pid=None)
        return state

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _bag_dir(recorded, bag_root):
    if bag_root is None:
        return Path(recorded)
    if isinstance(bag_root, Mapping):
        moved = {str(Path(old)): new for old, new in bag_root.items()}
        return Path(moved.get(recorded, recorded))
    return Path(bag_root) / Path(recorded).name
