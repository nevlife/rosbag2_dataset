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
Write a smaller rosbag2 MCAP bag: keep topics, thin them, and compress images to JPEG.

python -m rosbag2_dataset.rewrite spec.yaml in_bag out_bag [--start-ns T] [--end-ns T]
"""

import argparse
from collections import deque
from dataclasses import dataclass
import io
from pathlib import Path
import struct

import numpy as np
from PIL import Image
from rosbags.rosbag2 import StoragePlugin
from rosbags.rosbag2 import Writer
from rosbags.rosbag2.metadata import parse_qos
from rosbags.typesys import get_types_from_msg
from rosbags.typesys import get_typestore
from rosbags.typesys import Stores

from ._cdr import compile_decoder
from ._mcap import RecordStream
from ._mcap import summary
from ._scene import COMPRESSED_IMAGE
from ._scene import IMAGE
from ._scene import image_pixels
from .sample import SampleRule

_SPEC_KEYS = {'keep', 'drop', 'jpeg', 'nearest'}
_JPEG_KEYS = {'quality', 'topics'}
_NEAREST_KEYS = {'reference', 'tolerance_ms', 'topics'}
_SAMPLES = 5
_HEAD_BYTES = 1 << 30
_PENDING, _WRITE, _DROP = range(3)


@dataclass(frozen=True)
class Spec:
    """A parsed rewrite spec; keep is None when the spec lists topics to drop."""

    keep: tuple | None
    drop: tuple
    jpeg: dict
    reference: str | None
    tolerance_ns: int
    nearest: tuple


@dataclass(frozen=True)
class TopicCount:
    """Messages of an input topic in the window and of its output topic."""

    topic: str
    input_count: int
    output_count: int
    output_bytes: int


def _check_keys(where, mapping, allowed):
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f'Unknown {where} keys: {sorted(unknown)}')


def load_spec(source):
    """Parse a spec from a YAML path or a mapping."""
    if isinstance(source, (str, Path)):
        import yaml

        raw = yaml.safe_load(Path(source).read_text())
    else:
        raw = dict(source)
    _check_keys('spec', raw, _SPEC_KEYS)
    if ('keep' in raw) == ('drop' in raw):
        raise ValueError('The spec needs exactly one of keep or drop')
    jpeg = raw.get('jpeg') or {}
    _check_keys('jpeg', jpeg, _JPEG_KEYS)
    nearest = raw.get('nearest') or {}
    _check_keys('nearest', nearest, _NEAREST_KEYS)
    topics = jpeg.get('topics', ())
    if not isinstance(topics, dict):
        topics = dict.fromkeys(topics, jpeg.get('quality', 92))
    return Spec(
        tuple(raw['keep']) if 'keep' in raw else None,
        tuple(raw.get('drop', ())),
        {name: int(quality) for name, quality in topics.items()},
        nearest.get('reference'),
        round(nearest['tolerance_ms'] * 1e6) if nearest else 0,
        tuple(nearest.get('topics', ())),
    )


def compressed_image(sec, nanosec, frame_id, data, image_format='jpeg'):
    """Serialize a CompressedImage as little-endian XCDR1, aligned from after the header."""
    out = bytearray(b'\x00\x01\x00\x00')
    out += struct.pack('<iI', sec, nanosec)
    for text in (frame_id.encode(), image_format.encode()):
        out += bytes(-(len(out) - 4) % 4)
        out += struct.pack('<I', len(text) + 1) + text + b'\x00'
    out += bytes(-(len(out) - 4) % 4)
    out += struct.pack('<I', len(data)) + data
    return bytes(out)


def _jpeg(message, quality):
    buffer = io.BytesIO()
    pixels = np.ascontiguousarray(image_pixels(message))
    Image.fromarray(pixels).save(buffer, 'JPEG', quality=quality)
    stamp = message.header.stamp
    return compressed_image(stamp.sec, stamp.nanosec, message.header.frame_id, buffer.getvalue())


def _files(path):
    path = Path(path)
    paths = sorted(path.glob('*.mcap')) if path.is_dir() else [path]
    if not paths:
        raise FileNotFoundError(f'No .mcap files found in {path}')
    return paths


def _type(schemas, channel):
    schema = schemas.get(channel.schema_id)
    return schema.name if schema else ''


def _listed_types(path, paths):
    stats = metadata_stats(path)[0]
    if stats:
        return {name: info.type for name, info in stats.items()}
    types = {}
    for file in paths:
        schemas, channels, _ = summary(file)
        types.update({channel.topic: _type(schemas, channel) for channel in channels.values()})
    return types


def _topics(spec, types):
    if spec.keep is not None:
        names = list(spec.keep)
    else:
        names = [name for name in types if name not in spec.drop]
    named = {*names, *spec.drop, *spec.jpeg, *spec.nearest}
    if spec.reference is not None:
        named.add(spec.reference)
    missing = named - set(types)
    if missing:
        raise ValueError(f'Topics not in the bag: {sorted(missing)}')
    unkept = {*spec.jpeg, *spec.nearest} - set(names)
    if unkept:
        raise ValueError(f'jpeg and nearest topics must be kept: {sorted(unkept)}')
    for name in spec.jpeg:
        if types[name] != IMAGE:
            raise ValueError(f'jpeg topic {name} is not {IMAGE}')
        if name + '/compressed' in names:
            raise ValueError(f'{name}/compressed is kept and would be written twice')
    return names


def _kept(spec, name):
    return name in spec.keep if spec.keep is not None else name not in spec.drop


def _timed(spec):
    return {spec.reference, *spec.nearest} if spec.nearest else set()


def _check_stamp(name, schema):
    """Require a ros2msg schema whose first field is a header or a Foxglove timestamp."""
    field = None
    for line in schema.text.splitlines() if schema and schema.encoding == 'ros2msg' else ():
        code = line.split('#')[0]
        words = code.split()
        if len(words) >= 2 and '=' not in code:
            field = words[0].split('/')[-1], words[1]
            break
    if field not in (('Header', 'header'), ('Time', 'timestamp')):
        raise ValueError(f'nearest needs {name} to start with a header stamp')


def _header_stamp(payload):
    """Read header.stamp (or timestamp), the first field after the CDR encapsulation."""
    sec, nanosec = struct.unpack_from('<iI' if payload[1] else '>iI', payload, 4)
    return sec * 1_000_000_000 + nanosec


class _Nearest:
    """
    Choose, per reference message, the nearest message of each thinned topic.

    A reference stamp is settled for a topic once that topic has a stamp later than it
    plus the tolerance. A held message is decided once the reference has a stamp later
    than its own plus the tolerance and no unsettled reference can still choose it.
    Both steps assume that header stamps do not decrease in file order.
    """

    def __init__(self, spec):
        self.tolerance = spec.tolerance_ns
        self.rules = {
            name: SampleRule([spec.reference, name], tolerance_ns=self.tolerance, optional=[name])
            for name in spec.nearest
        }
        self.waiting = {name: deque() for name in spec.nearest}
        self.held = {name: [] for name in spec.nearest}
        self.latest = dict.fromkeys(spec.nearest)
        self.latest_reference = None

    def reference(self, stamp):
        if self.latest_reference is None or stamp > self.latest_reference:
            self.latest_reference = stamp
        for name, waiting in self.waiting.items():
            waiting.append(stamp)
            self._settle(name)

    def candidate(self, name, stamp, entry):
        self.held[name].append((stamp, entry))
        if self.latest[name] is None or stamp > self.latest[name]:
            self.latest[name] = stamp
        self._settle(name)

    def finish(self):
        for name in self.waiting:
            self._settle(name, final=True)

    def _settle(self, name, final=False):
        waiting, latest = self.waiting[name], self.latest[name]
        while waiting and (final or latest is not None and latest > waiting[0] + self.tolerance):
            self._choose(name, waiting.popleft())
        unsettled = min(waiting, default=None)
        held = []
        for stamp, entry in self.held[name]:
            horizon = stamp + self.tolerance
            passed = self.latest_reference is not None and self.latest_reference > horizon
            if final or passed and (unsettled is None or unsettled > horizon):
                if entry[3] == _PENDING:
                    entry[3] = _DROP
            else:
                held.append((stamp, entry))
        self.held[name] = held

    def _choose(self, name, stamp):
        held = self.held[name]
        if not held:
            return
        rule = self.rules[name]
        stamps = np.array([held_stamp for held_stamp, _ in held], dtype=np.int64)
        order = np.argsort(stamps, kind='stable')
        clocks = {rule.reference: np.array([stamp], dtype=np.int64), name: stamps[order]}
        index = rule.select(clocks, 0)[0][1]
        if index >= 0:
            held[order[index]][1][3] = _WRITE


def _compressed_schema(path):
    for schema in summary(path)[0].values():
        if schema.name == COMPRESSED_IMAGE:
            return schema.text
    return get_typestore(Stores.ROS2_JAZZY).generate_msgdef(COMPRESSED_IMAGE, ros_version=2)[0]


def _connection(writer, name, typename, text, channel):
    store = get_typestore(Stores.EMPTY)
    store.register(get_types_from_msg(text, typename))
    return writer.add_connection(
        name,
        typename,
        msgdef=text,
        rihs01=store.hash_rihs01(typename),
        serialization_format=channel.encoding,
        offered_qos_profiles=parse_qos(channel.metadata['offered_qos_profiles']),
    )


class _Output:
    """Write kept messages in input order; held thinned messages hold back later ones."""

    def __init__(self, spec, writer, compressed):
        self.spec = spec
        self.writer = writer
        self.compressed = compressed
        self.nearest = _Nearest(spec) if spec.nearest else None
        self.queue = deque()
        self.connections = {}
        self.decoders = {}
        self.counts = {}
        self.timed = _timed(spec)
        self.checked = set()

    def add(self, stream, channel, time, payload):
        info = stream.channels[channel]
        name = info.topic
        schema = stream.schemas.get(info.schema_id)
        if name in self.timed:
            if name not in self.checked:
                _check_stamp(name, schema)
                self.checked.add(name)
            stamp = _header_stamp(payload)
        entry = [time, name, payload, _WRITE]
        if _kept(self.spec, name):
            if name not in self.counts:
                self._connect(name, info, schema)
            self.counts[name][0] += 1
            if name in self.spec.nearest:
                entry[3] = _PENDING
                self.nearest.candidate(name, stamp, entry)
            self.queue.append(entry)
        if self.nearest is not None and name == self.spec.reference:
            self.nearest.reference(stamp)
        self.flush()

    def _connect(self, name, info, schema):
        if name in self.spec.jpeg:
            if schema.name != IMAGE:
                raise ValueError(f'jpeg topic {name} is not {IMAGE}')
            self.decoders[name] = compile_decoder(schema.name, schema.text)
            typename, text, output = COMPRESSED_IMAGE, self.compressed, name + '/compressed'
        else:
            typename, text, output = schema.name, schema.text, name
        self.connections[name] = _connection(self.writer, output, typename, text, info)
        self.counts[name] = [0, 0, 0]

    def flush(self):
        queue = self.queue
        while queue and queue[0][3] != _PENDING:
            time, name, payload, state = queue.popleft()
            if state == _DROP:
                continue
            if name in self.spec.jpeg:
                payload = _jpeg(self.decoders[name](payload), self.spec.jpeg[name])
            self.writer.write(self.connections[name], time, payload)
            self.counts[name][1] += 1
            self.counts[name][2] += len(payload)

    def finish(self):
        if self.nearest is not None:
            self.nearest.finish()
        self.flush()


def _wanted(spec, stream, start_ns, end_ns):
    """Return want(channel, log time) for RecordStream.messages; it caches each channel."""
    cache = {}
    timed = _timed(spec)

    def want(channel, time):
        if (start_ns is not None and time < start_ns) or (end_ns is not None and time >= end_ns):
            return False
        if channel not in cache:
            if channel not in stream.channels:
                raise ValueError(f'{stream.path}: message on unknown channel {channel}')
            name = stream.channels[channel].topic
            cache[channel] = _kept(spec, name) or name in timed
        return cache[channel]

    return want


def _no_progress(stage, done, total):
    pass


def rewrite(spec, bag, out, *, start_ns=None, end_ns=None, progress=_no_progress):
    """
    Write the kept messages with log time in [start_ns, end_ns) to a new bag at out.

    bag is a bag directory or an MCAP file, read once in file order without an index.
    progress('write', done, total) is called before each message with the input bytes
    passed and the total input bytes. Returns a TopicCount per output topic. Topics
    without messages in the range are not written.
    """
    spec = load_spec(spec)
    paths = _files(bag)
    listed = _listed_types(bag, paths)
    names = _topics(spec, listed) if listed else None
    compressed = _compressed_schema(paths[0]) if spec.jpeg else None
    total = sum(path.stat().st_size for path in paths)
    done = 0
    seen = {}
    ended = False
    with Writer(out, version=9, storage_plugin=StoragePlugin.MCAP) as writer:
        output = _Output(spec, writer, compressed)
        for path in paths:
            stream = RecordStream(path)
            want = _wanted(spec, stream, start_ns, end_ns)
            for channel, time, _, payload in stream.messages(want, end_ns):
                progress('write', done + stream.position, total)
                ended = end_ns is not None and time >= end_ns
                if ended:
                    break
                if payload is not None:
                    output.add(stream, channel, time, payload)
            for info in stream.channels.values():
                seen.setdefault(info.topic, _type(stream.schemas, info))
            if ended:
                break
            done += path.stat().st_size
        output.finish()
    if names is None:
        names = _topics(spec, seen)
    outputs = {name: name + '/compressed' if name in spec.jpeg else name for name in names}
    written = [name for name in names if name in output.counts]
    return [TopicCount(outputs[name], *output.counts[name]) for name in written]


@dataclass(frozen=True)
class TopicStats:
    """Type, message count, rate over the bag's duration, and estimated bytes, or None."""

    type: str  # noqa: A003
    count: int
    hz: float | None
    bytes: int | None  # noqa: A003


@dataclass(frozen=True)
class Choice:
    """What the window does with a topic; quality None copies an image unchanged."""

    name: str
    keep: bool
    quality: int | None = None
    thin: bool = False


def _hz(count, duration_ns):
    return count * 1e9 / duration_ns if duration_ns > 0 else None


def _stats(types, counts, spans):
    first = min((start for start, _ in spans), default=None)
    duration = max(end for _, end in spans) - first if spans else 0
    stats = {}
    for name, typename in types.items():
        count = counts.get(name, 0)
        stats[name] = TopicStats(typename, count, _hz(count, duration), None)
    return stats, first


def metadata_stats(path):
    """Read topics and the first log time from a bag directory's metadata.yaml, or ({}, None)."""
    import yaml

    metadata = Path(path) / 'metadata.yaml'
    if not metadata.is_file():
        return {}, None
    info = yaml.safe_load(metadata.read_text())['rosbag2_bagfile_information']
    duration = info['duration']['nanoseconds']
    stats = {}
    for row in info['topics_with_message_count']:
        topic = row['topic_metadata']
        count = row['message_count']
        stats[topic['name']] = TopicStats(topic['type'], count, _hz(count, duration), None)
    return stats, info['starting_time']['nanoseconds_since_epoch']


def summary_stats(paths):
    """Count topics from the summary statistics of MCAP files; ({}, None) if one has none."""
    types, counts, spans = {}, {}, []
    for path in paths:
        schemas, channels, statistics = summary(path)
        if statistics is None:
            return {}, None
        first, last, by_channel = statistics
        for ident, channel in channels.items():
            types.setdefault(channel.topic, _type(schemas, channel))
            counts[channel.topic] = counts.get(channel.topic, 0) + by_channel.get(ident, 0)
        if any(by_channel.values()):
            spans.append((first, last))
    return _stats(types, counts, spans)


def _skip(channel, time):
    return False


def walk_stats(paths, progress=_no_progress):
    """Count topics from every message record header; progress('count', bytes, total)."""
    types, counts, spans = {}, {}, []
    total = sum(path.stat().st_size for path in paths)
    done = 0
    for path in paths:
        stream = RecordStream(path)
        first = last = None
        for channel, time, _, _ in stream.messages(_skip):
            progress('count', done + stream.position, total)
            name = stream.channels[channel].topic
            counts[name] = counts.get(name, 0) + 1
            first = time if first is None else min(first, time)
            last = time if last is None else max(last, time)
        for channel in stream.channels.values():
            types.setdefault(channel.topic, _type(stream.schemas, channel))
        if first is not None:
            spans.append((first, last))
        done += path.stat().st_size
    return _stats(types, counts, spans)


def head_sizes(path, topics=(), progress=_no_progress, limit=_HEAD_BYTES, samples=_SAMPLES):
    """
    Return the mean CDR size of each topic's first messages in an MCAP file's first bytes.

    The walk reads record headers up to limit bytes and ends early once each of topics
    has samples sizes; progress('head', bytes, limit) follows it.
    """
    sizes = {}
    left = set(topics)
    stream = RecordStream(Path(path))
    for channel, _, size, _ in stream.messages(_skip):
        if stream.position >= limit:
            break
        progress('head', stream.position, limit)
        name = stream.channels[channel].topic
        rows = sizes.setdefault(name, [])
        if len(rows) < samples:
            rows.append(size)
            if len(rows) == samples:
                left.discard(name)
                if topics and not left:
                    break
    return {name: float(np.mean(rows)) for name, rows in sizes.items()}


def recorded_preset(path):
    """Return the MCAP storage preset that the recording session names, or ''."""
    import yaml

    folder = Path(path) if Path(path).is_dir() else Path(path).parent
    # rosbag_recorder writes summary.yaml into the bag folder.
    recorded = folder / 'summary.yaml'
    if recorded.is_file():
        return yaml.safe_load(recorded.read_text()).get('storage_preset', '')
    # Older sessions hold the bag in <session>/bag with the ros2 bag command in manifest.yaml.
    manifest = folder.parent / 'manifest.yaml'
    if manifest.is_file():
        command = yaml.safe_load(manifest.read_text()).get('command', [])
        if '--storage-preset-profile' in command:
            return command[command.index('--storage-preset-profile') + 1]
    return ''


def build_spec(choices, reference=None, tolerance_ms=50):
    """
    Return the rewrite spec mapping of the choices.

    Thinning to the nearest reference message applies to kept topics with thin set,
    other than the reference; there is none without a reference.
    """
    kept = [choice for choice in choices if choice.keep]
    spec = {'keep': [choice.name for choice in kept]}
    jpeg = {choice.name: choice.quality for choice in kept if choice.quality is not None}
    if jpeg:
        spec['jpeg'] = {'topics': jpeg}
    thin = [choice.name for choice in kept if choice.thin and choice.name != reference]
    if reference and thin:
        spec['nearest'] = {'reference': reference, 'tolerance_ms': tolerance_ms, 'topics': thin}
    return spec


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='python -m rosbag2_dataset.rewrite',
        description='Write a smaller rosbag2 MCAP bag from a spec.',
    )
    parser.add_argument('spec', help='rewrite spec YAML')
    parser.add_argument('bag', help='input bag directory or .mcap file')
    parser.add_argument('out', help='output bag directory; must not exist')
    parser.add_argument('--start-ns', type=int, help='first log time to keep')
    parser.add_argument('--end-ns', type=int, help='log time to stop before')
    args = parser.parse_args(argv)
    counts = rewrite(args.spec, args.bag, args.out, start_ns=args.start_ns, end_ns=args.end_ns)
    for path in sorted(Path(args.out).iterdir()):
        print(path)
    width = max([len('topic')] + [len(row.topic) for row in counts])
    print(f'{"topic":<{width}}  {"input":>8}  {"output":>8}  {"output bytes":>14}')
    for row in counts:
        print(
            f'{row.topic:<{width}}  {row.input_count:>8}  {row.output_count:>8}  '
            f'{row.output_bytes:>14}'
        )


if __name__ == '__main__':
    main()
