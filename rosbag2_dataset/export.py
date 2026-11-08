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

"""
Export synchronized rosbag2 MCAP frames into self-describing scene folders.

python -m rosbag2_dataset.export spec.yaml bag out [--start-ns T] [--end-ns T] [--clean]
"""

import argparse
from dataclasses import dataclass
from dataclasses import replace
from fractions import Fraction
import importlib
import io
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ._scene import cloud_points
from ._scene import COMPRESSED_IMAGE
from ._scene import compressed_pixels
from ._scene import IMAGE
from ._scene import image_pixels
from ._scene import POINT_CLOUD
from .bag import Bag
from .video import codec_of
from .video import CODECS
from .video import VIDEO_TYPES

FORMAT_VERSION = 1
STORES = ('jpeg', 'png', 'npy', 'mp4', 'ref', 'table')
_STORE_TYPES = {
    'jpeg': {IMAGE, COMPRESSED_IMAGE, *VIDEO_TYPES},
    'png': {IMAGE, *VIDEO_TYPES},
    'mp4': {IMAGE, COMPRESSED_IMAGE, *VIDEO_TYPES},
    'npy': {IMAGE, COMPRESSED_IMAGE, POINT_CLOUD, *VIDEO_TYPES},
}
_SPEC_KEYS = {'reference', 'scene_frames', 'time', 'sensors', 'tf', 'hooks'}
_SENSOR_KEYS = {
    'topic',
    'store',
    'sync',
    'max_ms',
    'required',
    'time',
    'fields',
    'finite',
    'info',
    'quality',
    'codec',
    'crf',
    'gop',
    'fps',
}
_TF_KEYS = {'base', 'static', 'dynamic'}
_POSE = pa.list_(pa.float64(), 16)


@dataclass(frozen=True)
class Sensor:
    """One sensor entry of a spec."""

    name: str
    topic: str
    store: str
    max_ns: int | None = None
    required: bool = False
    time: str | None = None
    fields: tuple | None = None
    finite: bool = False
    info: str | None = None
    quality: int = 92
    codec: str | None = None
    crf: int = 23
    gop: int = 10
    fps: int | None = None


@dataclass(frozen=True)
class Spec:
    """A parsed export spec; source keeps the mapping it was read from."""

    reference: str
    scene_frames: int
    time: str
    sensors: tuple
    base: str | None
    static: bool
    dynamic: str | None
    hooks: tuple
    source: dict


def _check_keys(where, mapping, allowed):
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f'Unknown {where} keys: {sorted(unknown)}')


def _time_source(value):
    if value not in ('header', 'log'):
        raise ValueError(f"time must be 'header' or 'log', not {value!r}")
    return value


def load_spec(source):
    """Parse a spec from a YAML path or a mapping."""
    if isinstance(source, (str, Path)):
        import yaml

        raw = yaml.safe_load(Path(source).read_text())
    else:
        raw = dict(source)
    _check_keys('spec', raw, _SPEC_KEYS)
    sensors = []
    for name, entry in raw['sensors'].items():
        _check_keys(f'sensor {name!r}', entry, _SENSOR_KEYS)
        if entry['store'] not in STORES:
            raise ValueError(f'Sensor {name!r}: store must be one of {STORES}')
        if entry.get('sync', 'nearest') != 'nearest':
            raise ValueError(f"Sensor {name!r}: only sync 'nearest' is supported")
        max_ms = entry.get('max_ms')
        options = {
            key: entry[key]
            for key in ('required', 'finite', 'info', 'quality', 'codec', 'crf', 'gop', 'fps')
            if key in entry
        }
        sensors.append(
            Sensor(
                name,
                entry['topic'],
                entry['store'],
                max_ns=None if max_ms is None else round(max_ms * 1e6),
                time=_time_source(entry['time']) if 'time' in entry else None,
                fields=tuple(entry['fields']) if 'fields' in entry else None,
                **options,
            )
        )
    topics = [sensor.topic for sensor in sensors]
    if len(set(topics)) != len(topics):
        raise ValueError('Sensors must use distinct topics')
    by_name = {sensor.name: sensor.topic for sensor in sensors}
    reference = by_name.get(raw['reference'], raw['reference'])
    if any(s.topic == reference and s.store == 'table' for s in sensors):
        raise ValueError('The reference sensor cannot use store: table')
    tf = raw.get('tf') or {}
    _check_keys('tf', tf, _TF_KEYS)
    if tf and 'base' not in tf:
        raise ValueError('tf needs a base frame')
    hooks = raw.get('hooks') or []
    hooks = [hooks] if isinstance(hooks, str) else hooks
    resolved = []
    for hook in hooks:
        module, function = hook.split(':')
        resolved.append(getattr(importlib.import_module(module), function))
    return Spec(
        reference,
        int(raw['scene_frames']),
        _time_source(raw['time']),
        tuple(sensors),
        tf.get('base'),
        bool(tf.get('static', True)) if tf else False,
        tf.get('dynamic'),
        tuple(resolved),
        raw,
    )


def _matrix(translation, quaternion):
    x, y, z, w = quaternion
    result = np.eye(4)
    result[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    result[:3, 3] = translation
    return result


def _slerp(q0, q1, weight):
    dot = float(q0 @ q1)
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + weight * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = np.arccos(dot)
    return (np.sin((1 - weight) * theta) * q0 + np.sin(weight * theta) * q1) / np.sin(theta)


def _transforms(reader, topic):
    """Yield (child, parent, stamp_ns, translation, quaternion) of a TF topic."""
    if topic not in reader.topics:
        return
    for _, message in reader.topic(topic):
        for item in message.transforms:
            stamp = item.header.stamp.sec * 1_000_000_000 + item.header.stamp.nanosec
            t, q = item.transform.translation, item.transform.rotation
            yield (
                item.child_frame_id,
                item.header.frame_id,
                stamp,
                (t.x, t.y, t.z),
                np.array([q.x, q.y, q.z, q.w]),
            )


class _Tree:
    """Resolve transforms over static edges and interpolated dynamic edges."""

    def __init__(self, static, dynamic):
        self.static = static
        self.dynamic = dynamic

    def _step(self, frame, time):
        if frame in self.dynamic:
            parent, times, translations, quaternions = self.dynamic[frame]
            index = int(np.searchsorted(times, time))
            if index < len(times) and times[index] == time:
                return parent, _matrix(translations[index], quaternions[index])
            if index == 0 or index == len(times):
                return parent, None
            weight = (time - times[index - 1]) / (times[index] - times[index - 1])
            translation = translations[index - 1] + weight * (
                translations[index] - translations[index - 1]
            )
            return parent, _matrix(
                translation, _slerp(quaternions[index - 1], quaternions[index], weight)
            )
        return self.static[frame]

    def _root(self, frame, time):
        pose = np.eye(4)
        while frame in self.dynamic or frame in self.static:
            frame, step = self._step(frame, time)
            if step is None:
                return None, None
            pose = step @ pose
        return frame, pose

    def lookup(self, target, source, time=None):
        """Return T_target_source, or None when the frames are not connected."""
        root_target, target_pose = self._root(target, time)
        root_source, source_pose = self._root(source, time)
        if root_target is None or root_source is None or root_target != root_source:
            return None
        return np.linalg.inv(target_pose) @ source_pose


def _flatten(value, prefix, row):
    for key, item in value.items():
        if isinstance(item, dict):
            _flatten(item, f'{prefix}{key}.', row)
        else:
            row[prefix + key] = item


def _offset_stats(offsets):
    if not len(offsets):
        return None
    values = np.abs(offsets) / 1e6
    return {
        'p50': float(np.percentile(values, 50)),
        'p95': float(np.percentile(values, 95)),
        'max': float(values.max()),
    }


def _pixels(kind, video, index, message):
    if kind == IMAGE:
        return image_pixels(message)
    if kind == COMPRESSED_IMAGE:
        return compressed_pixels(message)
    return video[index][1]


class _FileStore:
    """Write one jpg, png, or npy file per distinct message."""

    def __init__(self, sensor, kind, video, scene_dir):
        self.sensor = sensor
        self.kind = kind
        self.video = video
        self.scene_dir = scene_dir
        self.extension = {'jpeg': 'jpg', 'png': 'png', 'npy': 'npy'}[sensor.store]
        (scene_dir / sensor.name).mkdir()

    def put(self, number, index, message):
        relative = f'{self.sensor.name}/{number:06d}.{self.extension}'
        path = self.scene_dir / relative
        if (
            self.sensor.store == 'jpeg'
            and self.kind == COMPRESSED_IMAGE
            and 'jpeg' in message.format
        ):
            path.write_bytes(message.data)
            return relative
        if self.kind == POINT_CLOUD:
            array = cloud_points(message, self.sensor.fields, self.sensor.finite)
        else:
            array = _pixels(self.kind, self.video, index, message)
        if self.sensor.store == 'npy':
            np.save(path, array)
        else:
            from PIL import Image

            options = {'quality': self.sensor.quality} if self.sensor.store == 'jpeg' else {}
            Image.fromarray(np.ascontiguousarray(array)).save(path, **options)
        return relative

    def close(self):
        return {}


class _VideoStore:
    """Encode distinct messages as consecutive video frames of one mp4 per scene."""

    def __init__(self, sensor, kind, video, scene_dir, fps):
        import av

        self.sensor = sensor
        self.kind = kind
        self.video = video
        self.codec = sensor.codec or 'libx264'
        self.file = f'{sensor.name}.mp4'
        self.container = av.open(str(scene_dir / self.file), 'w')
        self.fps = fps
        self.stream = None
        self.format = None
        self.count = 0

    def put(self, number, index, message):
        import av

        pixels = _pixels(self.kind, self.video, index, message)
        if pixels.dtype != np.uint8:
            raise ValueError(f'Sensor {self.sensor.name!r}: mp4 needs 8-bit images')
        if self.stream is None:
            self.format = 'gray' if pixels.ndim == 2 else 'rgb24'
            self.stream = self.container.add_stream(self.codec, rate=self.fps)
            self.stream.width, self.stream.height = pixels.shape[1], pixels.shape[0]
            self.stream.pix_fmt = 'yuv420p'
            self.stream.codec_context.gop_size = self.sensor.gop
            self.stream.options = {'crf': str(self.sensor.crf)}
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(pixels), format=self.format)
        frame.pts = self.count
        self.container.mux(self.stream.encode(frame))
        self.count += 1
        return self.count - 1

    def close(self):
        if self.stream is not None:
            self.container.mux(self.stream.encode(None))
        self.container.close()
        if self.stream is None:
            return {}
        return {
            'video': {
                'file': self.file,
                'codec': self.codec,
                'pix_fmt': 'yuv420p',
                'format': self.format,
                'fps': self.fps,
                'gop': self.sensor.gop,
                'crf': self.sensor.crf,
                'frames': self.count,
                'width': self.stream.width,
                'height': self.stream.height,
            }
        }


class _RemuxStore:
    """Copy the packets from the keyframe before the scene's first message into an mp4."""

    def __init__(self, sensor, video, scene_dir, fps, used):
        self.video = video
        self.path = scene_dir / f'{sensor.name}.mp4'
        self.fps = fps
        self.first = video.keyframe(int(used.min())) if len(used) else None
        self.last = int(used.max()) if len(used) else None

    def put(self, number, index, message):
        return index - self.first

    def close(self):
        import av

        if self.first is None:
            return {}
        video = self.video
        span = range(self.first, self.last + 1)
        packets = [video.topic[i][1].data for i in span]
        keys = [video.is_keyframe(i) for i in span]
        rate = Fraction(1, self.fps)
        with av.open(io.BytesIO(packets[0]), format=CODECS[video.codec][1]) as source:
            template = source.streams.video[0]
            width, height = template.codec_context.width, template.codec_context.height
            with av.open(str(self.path), 'w') as container:
                stream = container.add_stream_from_template(template, opaque=True)
                stream.time_base = rate
                for number, data in enumerate(packets):
                    packet = av.Packet(data)
                    packet.stream = stream
                    packet.pts = packet.dts = number
                    packet.duration = 1
                    packet.time_base = rate
                    packet.is_keyframe = keys[number]
                    container.mux(packet)
        return {
            'video': {
                'file': self.path.name,
                'codec': video.codec,
                'remux': True,
                'format': 'rgb24',
                'fps': self.fps,
                'gop': int(np.diff([*np.flatnonzero(keys), len(keys)]).max()),
                'frames': len(packets),
                'width': width,
                'height': height,
            }
        }


class _RefStore:
    """Record where each message lives in the bag instead of copying it."""

    def __init__(self, sensor, topic, source_dir):
        self.topic = topic
        self.source_dir = source_dir

    def put(self, index):
        path, chunk, record = self.topic.location(index)
        return str(path.relative_to(self.source_dir)), chunk, record

    def close(self):
        name, encoding, text = self.topic.schema()
        return {'schema': {'name': name, 'encoding': encoding, 'text': text}}


class Exporter:
    """Synchronize a bag once and write its scenes one at a time."""

    def __init__(self, spec, reader, bag_path):
        self.spec = spec if isinstance(spec, Spec) else load_spec(spec)
        self.reader = reader
        self.bag_path = Path(bag_path).resolve()
        self.source_dir = self.bag_path if self.bag_path.is_dir() else self.bag_path.parent
        self.bag_name = self.bag_path.name if self.bag_path.is_dir() else self.bag_path.stem
        spec = self.spec
        for sensor in spec.sensors:
            if sensor.topic not in reader.topics or not reader.topics[sensor.topic].count:
                raise ValueError(f'Sensor {sensor.name!r}: no messages on {sensor.topic}')
            kind = reader.topics[sensor.topic].type
            if sensor.store in _STORE_TYPES and kind not in _STORE_TYPES[sensor.store]:
                raise ValueError(
                    f'Sensor {sensor.name!r}: store {sensor.store} does not accept {kind}'
                )
        self.frame_sensors = []
        for sensor in spec.sensors:
            if sensor.store == 'table':
                continue
            if reader.topics[sensor.topic].type == POINT_CLOUD and sensor.fields is None:
                fields = reader.topic(sensor.topic)[0][1].fields
                sensor = replace(sensor, fields=tuple(f.name for f in fields if f.count == 1))
            self.frame_sensors.append(sensor)
        self.table_sensors = [s for s in spec.sensors if s.store == 'table']
        self.videos = {
            s.name: reader.video(s.topic)
            for s in self.frame_sensors
            if reader.topics[s.topic].type in VIDEO_TYPES and s.store != 'ref'
        }
        topics = [spec.reference] + [
            s.topic for s in self.frame_sensors if s.topic != spec.reference
        ]
        never = np.iinfo(np.int64).max
        self.sync = reader.synced(
            topics,
            reference=spec.reference,
            time_source=spec.time,
            tolerance_ns={
                s.topic: never if s.max_ns is None else s.max_ns
                for s in self.frame_sensors
                if s.topic != spec.reference
            },
            optional=[
                s.topic for s in self.frame_sensors if not s.required and s.topic != spec.reference
            ],
        )
        self.columns = {topic: i for i, topic in enumerate(topics)}
        reference_times = np.sort(self.sync.times[:, 0])
        steps = np.diff(reference_times)
        self.rate_hz = float(1e9 / np.median(steps)) if len(steps) else None
        self.tables = {}
        for sensor in self.table_sensors:
            clock = reader.topic(sensor.topic).stamps(sensor.time or spec.time)
            order = np.argsort(clock, kind='stable')
            self.tables[sensor.name] = (clock[order], order)
        static = {}
        dynamic = {}
        if spec.base is not None and spec.static:
            for child, parent, _, translation, quaternion in _transforms(reader, '/tf_static'):
                static[child] = (parent, _matrix(translation, quaternion))
        if spec.dynamic is not None:
            edges = {}
            for child, parent, stamp, translation, quaternion in _transforms(reader, '/tf'):
                edges.setdefault(child, (parent, []))[1].append((stamp, translation, quaternion))
            for child, (parent, samples) in edges.items():
                samples.sort(key=lambda sample: sample[0])
                dynamic[child] = (
                    parent,
                    np.array([s[0] for s in samples], dtype=np.int64),
                    np.array([s[1] for s in samples]),
                    np.array([s[2] for s in samples]),
                )
        self.static_tree = _Tree(static, {})
        self.tree = _Tree(static, dynamic)

    def scenes(self, start_ns=None, end_ns=None):
        """
        Yield (rows, table start, table stop) per scene of scene_frames rows.

        Table rows of a scene follow the previous scene's last frame, up to and
        including its own last frame; the first scene starts at its first frame.
        """
        times = self.sync.times[:, 0]
        keep = np.ones(len(times), dtype=bool)
        if start_ns is not None:
            keep &= times >= start_ns
        if end_ns is not None:
            keep &= times < end_ns
        rows = np.flatnonzero(keep)
        size = self.spec.scene_frames
        parts = [rows[i:][:size] for i in range(0, len(rows), size)]
        for k, part in enumerate(parts):
            start = int(times[parts[k - 1][-1]]) + 1 if k else int(times[part[0]])
            yield part, start, int(times[part[-1]]) + 1

    def write(self, root, rows, start, stop):
        """Write one scene, named by bag and first frame time, through a temporary folder."""
        name = f'{self.bag_name}_{int(self.sync.times[rows[0], 0]):019d}'
        final = Path(root) / name
        partial = Path(root) / f'.{name}.partial'
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir(parents=True)
        meta = self._write_frames(partial, rows)
        for sensor in self.table_sensors:
            meta['sensors'][sensor.name] = self._write_table(partial, sensor, start, stop)
        meta['scene'] = name
        (partial / 'scene.json').write_text(json.dumps(meta, indent=1))
        if final.exists():
            previous = Path(root) / f'.{name}.previous'
            if previous.exists():
                shutil.rmtree(previous)
            final.rename(previous)
            partial.rename(final)
            shutil.rmtree(previous)
        else:
            partial.rename(final)
        return final

    def _write_frames(self, scene_dir, rows):
        spec = self.spec
        reader = self.reader
        times = self.sync.times[rows]
        indices = self.sync.indices[rows]
        reference_times = times[:, 0]
        columns = {'timestamp_ns': pa.array(reference_times, pa.int64())}
        sensors = {}
        stores = {}
        for sensor in self.frame_sensors:
            topic = reader.topic(sensor.topic)
            kind = reader.topics[sensor.topic].type
            video = self.videos.get(sensor.name)
            if sensor.store == 'ref':
                stores[sensor.name] = _RefStore(sensor, topic, self.source_dir)
            elif sensor.store == 'mp4' and video is not None and sensor.codec is None:
                used = indices[:, self.columns[sensor.topic]]
                steps = np.diff(topic.times)
                fps = sensor.fps or (max(1, round(1e9 / np.median(steps))) if len(steps) else 1)
                stores[sensor.name] = _RemuxStore(sensor, video, scene_dir, fps, used[used >= 0])
            elif sensor.store == 'mp4':
                fps = sensor.fps or max(1, round(self.rate_hz or 1))
                stores[sensor.name] = _VideoStore(sensor, kind, video, scene_dir, fps)
            else:
                stores[sensor.name] = _FileStore(sensor, kind, video, scene_dir)
        values = {sensor.name: [] for sensor in self.frame_sensors}
        reused = dict.fromkeys(values, 0)
        previous = {}
        extras = []
        for number, row in enumerate(indices):
            messages = {}
            for sensor in self.frame_sensors:
                index = int(row[self.columns[sensor.topic]])
                if index < 0:
                    value = None
                    messages[sensor.name] = None
                elif previous.get(sensor.name, (None,))[0] == index:
                    value = previous[sensor.name][1]
                    reused[sensor.name] += 1
                    if spec.hooks:
                        messages[sensor.name] = reader.topic(sensor.topic)[index][1]
                elif sensor.store == 'ref':
                    value = stores[sensor.name].put(index)
                    if spec.hooks:
                        messages[sensor.name] = reader.topic(sensor.topic)[index][1]
                else:
                    message = reader.topic(sensor.topic)[index][1]
                    messages[sensor.name] = message
                    value = stores[sensor.name].put(number, index, message)
                if index >= 0:
                    previous[sensor.name] = (index, value)
                values[sensor.name].append(value)
            if spec.hooks:
                frame = {
                    'bag': reader,
                    'scene_dir': scene_dir,
                    'frame': number,
                    'timestamp_ns': int(reference_times[number]),
                    'messages': messages,
                }
                extra = {}
                for hook in spec.hooks:
                    extra.update(hook(frame) or {})
                extras.append(extra)
        first_log = int(reader.topic(spec.reference).times[indices[0, 0]])
        for sensor in self.frame_sensors:
            column = self.columns[sensor.topic]
            present = indices[:, column] >= 0
            name = sensor.name
            if sensor.store == 'ref':
                refs = [value or (None, None, None) for value in values[name]]
                columns[name + '_file'] = pa.array([r[0] for r in refs], pa.string())
                columns[name + '_chunk'] = pa.array([r[1] for r in refs], pa.int64())
                columns[name + '_record'] = pa.array([r[2] for r in refs], pa.int64())
            else:
                kind = pa.int64() if sensor.store == 'mp4' else pa.string()
                columns[name] = pa.array(values[name], kind)
            offsets = times[:, column] - reference_times
            columns[name + '_offset_ns'] = pa.array(offsets, pa.int64())
            sensors[name] = self._sensor_meta(sensor, indices[present, column], first_log)
            sensors[name].update(stores[name].close())
            sensors[name]['stats'] = {
                'missing': int((~present).sum()),
                'reused': reused[name],
                'offset_ms': _offset_stats(offsets[present]),
            }
        if spec.dynamic is not None:
            poses = [
                self.tree.lookup(spec.dynamic, spec.base, int(time)) for time in reference_times
            ]
            columns['T_world_base'] = pa.array(
                [None if pose is None else pose.reshape(16).tolist() for pose in poses], _POSE
            )
        for key in dict.fromkeys(key for extra in extras for key in extra):
            columns[key] = pa.array([extra.get(key) for extra in extras])
        pq.write_table(pa.table(columns), scene_dir / 'frames.parquet')
        reference = next(
            (s.name for s in spec.sensors if s.topic == spec.reference), spec.reference
        )
        return {
            'format': 'rosbag2_dataset/scene',
            'version': FORMAT_VERSION,
            'num_frames': len(rows),
            'reference': reference,
            'time': spec.time,
            'rate_hz': self.rate_hz,
            'source': {
                'bag': str(self.bag_path),
                'dir': str(self.source_dir),
                'files': sorted(
                    str(file.path.relative_to(self.source_dir)) for file in reader._files
                ),
                'start_ns': int(reference_times.min()),
                'end_ns': int(reference_times.max()),
            },
            'tf': {'base': spec.base, 'static': spec.static, 'world': spec.dynamic},
            'spec': spec.source,
            'sensors': sensors,
        }

    def _sensor_meta(self, sensor, present, first_log):
        reader = self.reader
        topic = reader.topic(sensor.topic)
        kind = reader.topics[sensor.topic].type
        message = topic[int(present[0]) if len(present) else 0][1]
        meta = {
            'topic': sensor.topic,
            'type': kind,
            'store': sensor.store,
            'max_ms': None if sensor.max_ns is None else sensor.max_ns / 1e6,
            'required': sensor.required,
        }
        header = getattr(message, 'header', message)
        if hasattr(header, 'frame_id'):
            meta['frame_id'] = header.frame_id
        if kind == IMAGE:
            meta.update(encoding=message.encoding, width=message.width, height=message.height)
        if kind in VIDEO_TYPES:
            meta['codec'] = codec_of(message)
        if kind == POINT_CLOUD:
            meta.update(fields=list(sensor.fields), finite=sensor.finite)
        if sensor.store == 'jpeg':
            meta['quality'] = sensor.quality
        info = sensor.info
        if info is None and kind in (IMAGE, COMPRESSED_IMAGE, *VIDEO_TYPES):
            siblings = [sensor.topic.rsplit('/', 1)[0] + '/camera_info']
            if kind != IMAGE:
                siblings.append(sensor.topic.rsplit('/', 2)[0] + '/camera_info')
            info = next((topic for topic in siblings if topic in reader.topics), None)
        if info is not None:
            camera = reader.topic(info)
            calibration = camera[camera.nearest(first_log)][1]
            meta['camera'] = {
                'topic': info,
                'frame_id': calibration.header.frame_id,
                'width': calibration.width,
                'height': calibration.height,
                'distortion_model': calibration.distortion_model,
                'D': calibration.d.tolist() if len(calibration.d) else [],
                'K': np.reshape(calibration.k, (3, 3)).tolist(),
                'R': np.reshape(calibration.r, (3, 3)).tolist(),
                'P': np.reshape(calibration.p, (3, 4)).tolist(),
            }
        if self.spec.base is not None and 'frame_id' in meta:
            pose = self.static_tree.lookup(self.spec.base, meta['frame_id'])
            if pose is not None:
                meta['T_base_sensor'] = pose.tolist()
        return meta

    def _write_table(self, scene_dir, sensor, start, stop):
        clock, order = self.tables[sensor.name]
        topic = self.reader.topic(sensor.topic)
        low, high = np.searchsorted(clock, start), np.searchsorted(clock, stop)
        stamps = clock[low:high]
        rows = []
        for stamp, index in zip(stamps, order[low:high]):
            log_time, message = topic[int(index)]
            row = {'timestamp_ns': int(stamp), 'log_time_ns': log_time}
            _flatten(message.to_dict(), '', row)
            rows.append(row)
        table = (
            pa.Table.from_pylist(rows)
            if rows
            else pa.table(
                {'timestamp_ns': pa.array([], pa.int64()), 'log_time_ns': pa.array([], pa.int64())}
            )
        )
        pq.write_table(table, scene_dir / f'{sensor.name}.parquet')
        gaps = np.diff(stamps)
        meta = {
            'topic': sensor.topic,
            'type': self.reader.topics[sensor.topic].type,
            'store': 'table',
            'file': f'{sensor.name}.parquet',
            'time': sensor.time or self.spec.time,
            'stats': {
                'count': len(rows),
                'max_gap_ms': float(gaps.max() / 1e6) if len(gaps) else None,
                'rate_hz': float(1e9 / np.median(gaps)) if len(gaps) else None,
            },
        }
        return meta


def export(spec, bag, out, *, start_ns=None, end_ns=None, clean=False):
    """
    Write each scene of a bag under out/<bag name>/ and return the scene paths.

    clean removes the bag's other scenes and partial folders after the export.
    """
    return list(_export(spec, bag, out, start_ns, end_ns, clean))


def _export(spec, bag, out, start_ns, end_ns, clean):
    with Bag(bag) as reader:
        exporter = Exporter(spec, reader, bag)
        root = Path(out) / exporter.bag_name
        written = set()
        for rows, start, stop in exporter.scenes(start_ns, end_ns):
            path = exporter.write(root, rows, start, stop)
            written.add(path.name)
            yield path
    if clean and root.exists():
        for path in root.iterdir():
            if path.name not in written and (
                (path / 'scene.json').exists() or path.name.endswith(('.partial', '.previous'))
            ):
                shutil.rmtree(path)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='python -m rosbag2_dataset.export',
        description='Export a rosbag2 MCAP bag into scenes.',
    )
    parser.add_argument('spec', help='export spec YAML')
    parser.add_argument('bag', help='bag directory or .mcap file')
    parser.add_argument('out', help='output directory; scenes go to out/<bag name>/')
    parser.add_argument('--start-ns', type=int, help='first reference time to export')
    parser.add_argument('--end-ns', type=int, help='reference time to stop before')
    parser.add_argument(
        '--clean', action='store_true', help="remove the bag's scenes that this run did not write"
    )
    args = parser.parse_args(argv)
    for path in _export(args.spec, args.bag, args.out, args.start_ns, args.end_ns, args.clean):
        print(path, flush=True)


if __name__ == '__main__':
    main()
