# rosbag2_dataset

Read rosbag2 MCAP bags for ML training with Python and NumPy, without ROS.
Random access, time alignment, and read-only array views avoid per-byte Python
objects and repeated image/point-cloud copies. The reader implements the
[MCAP specification](https://mcap.dev/spec) directly; no MCAP Python SDK is needed.

```python
import rosbag2_dataset as rd

with rd.open('/data/run') as bag:  # Also accepts a single .mcap file.
    print(bag.topics)             # name -> TopicInfo(type, count, start_ns, end_ns)
    camera = bag.topic('/camera/image')
    time_ns, msg = camera[camera.nearest(123456789)]
    pixels = rd.image_to_numpy(msg)
    time_ns, cdr = camera.raw(0)   # Read-only memoryview, including encapsulation.
    times = camera.times          # Read-only numpy int64, sorted by log time.
    fields = msg.to_dict()        # Explicitly copies arrays into lists.
    for topic, stamp, msg in bag.read(start_ns=0, end_ns=123456790):
        pass
    pairs = bag.synced(['/camera/image', '/lidar/points'],
                       reference='/camera/image', tolerance_ns=20_000_000)
    camera_index, lidar_index = pairs[0]
```

`read()` merges selected topics in log-time order with an inclusive start and
exclusive end. Topic indexing supports negative indices. Nearest-time ties
select the earlier timestamp and then its first message. Synchronization returns
tuples in the requested topic order, retains reference log-time order, permits
reuse of matching messages, and drops groups beyond tolerance. Set
`time_source='header'` to use `header.stamp`; this decodes each message once while
building alignment and supports stamps recorded out of order. `tolerance_ns` may
also map each non-reference topic to its own tolerance. Topics named in
`optional` keep the row and get index -1 when beyond tolerance. `pairs.times`
holds the clock of each chosen message (for -1, the nearest one).

Uncompressed CDR arrays view the read-only mmap. Compressed arrays view immutable
decompressed bytes; a per-file LRU retains four chunks by default
(`open(..., chunk_cache_size=4)`). Install `lz4` for lz4 chunks or `zstandard` for
zstd chunks. Missing modules produce an explicit error when a chunk is accessed.
Exported arrays and raw views remain valid after closing a bag or evicting a chunk;
release those views to release their storage. Strings, scalar values, nested
objects, and the in-memory index require allocations. No files are written beside
bags. A directory includes all its `*.mcap` files without requiring PyYAML.
A compressed chunk is decompressed whole even when only one of its topics is read;
record with the MCAP storage option `chunkByTopic: true` so that each chunk holds
one topic.

`image_to_numpy()` returns HxW or HxWxC views for rgb8, bgr8, rgba8, bgra8, mono8,
mono16, 16UC1, and 32FC1. `pointcloud2_to_numpy()` returns an HxW structured view
with named fields and subarrays for fields with count > 1. Both preserve byte
order and row padding; images preserve channel order.

```python
with rd.open('/data/run') as bag:
    video = bag.video('/camera/image_raw/foxglove')
    time_ns, rgb = video[video.nearest(123456789)]   # HxWx3 uint8
```

`bag.video()` decodes H.264, H.265, and AV1 topics recorded as
`foxglove_msgs/msg/CompressedVideo` (Foxglove's `format`; `foxglove`
image_transport, rosbag2_broll) or `ffmpeg_image_transport_msgs/msg/FFMPEGPacket`
(`ffmpeg_image_transport`; first token of `encoding`) with PyAV (`pip install .[video]`).
Each message must hold one frame and the stream must have no B-frames, as Foxglove
requires. Reading a message decodes from the keyframe at or before it; reading the
next message continues the decoder. Keyframes are the FFMPEGPacket key flag, or in
CompressedVideo an H.264 IDR, an H.265 IRAP picture, or an AV1 sequence header.
Messages before the first keyframe raise an error. `stamps('header')` uses the
Foxglove `timestamp` field. Decoders run single-threaded.

```python
from rosbag2_dataset.torch import BagDataset

samples = BagDataset(['/data/run1', '/data/run2'], ['/camera/image', '/imu'],
                     sync={'reference': '/camera/image',
                           'tolerance_ns': 20_000_000},
                     transform=prepare_training_sample)
# Each input to transform is {topic: (log_time_ns, decoded_message)}.
# Pass samples to torch.utils.data.DataLoader with your chosen collate_fn.
```

The adapter implements PyTorch's map-style protocol without needing to import
PyTorch. With `sync=None`, samples contain one message each in merged log-time
order within each bag; bags are concatenated in path order. Dataset serialization
contains paths/options/counts, never mappings. Workers reopen their own bags after
fork or spawn. Use a top-level transform that returns tensors or ordinary
collatable values; generated Message objects need a transform/custom collation
for multiprocessing. Read-only NumPy views need a copy before tensor mutation.
Call `samples.close()` or use it as a context manager when finished.

## Export to scenes

`rosbag2_dataset.export` writes a bag into self-describing scene folders that
training code reads without ROS or the bag reader; `SceneDataset` reads them back.
It needs `pip install .[export]` (pyarrow, PyYAML, Pillow); `store: mp4` also needs
`.[video]` (PyAV). Reading bags stays NumPy-only.

```bash
python -m rosbag2_dataset.export spec.yaml /data/run out [--start-ns T] [--end-ns T] [--clean]
```

```yaml
reference: lidar                 # sensor name or topic; one frame per reference message
scene_frames: 300                # frames per scene; the last scene may be shorter
time: header                     # header (header.stamp) or log (log time)
sensors:
  lidar: {topic: /velodyne_points, store: npy, fields: [x, y, z, intensity], finite: true}
  front: {topic: /camera/image_raw, sync: nearest, max_ms: 50, store: jpeg}
  depth: {topic: /camera/depth, max_ms: 50, store: png, required: true}
  imu: {topic: /imu, store: table}
tf: {base: base_link, static: true, dynamic: odom}
hooks: my_package.export:extra   # optional, one or a list of module:function
```

| Sensor key | Meaning |
| --- | --- |
| `topic` | Topic to export; each topic appears in one sensor only |
| `store` | `jpeg`, `png`, `npy` (one file per message), `mp4` (one video per scene), `ref` (bag location, no copy), `table` (native-rate Parquet) |
| `sync`, `max_ms` | Nearest message to the reference time; beyond `max_ms` the frame has no value for this sensor. No `max_ms` means no limit |
| `required` | Drop frames where this sensor has no message within `max_ms` |
| `time` | Clock of this sensor when it differs from the spec's `time` |
| `fields`, `finite` | PointCloud2 fields as N x C columns of one dtype (default: all scalar fields); `finite` drops points with any non-finite field |
| `info` | CameraInfo topic; default is the sibling `camera_info` of an image topic |
| `quality` | JPEG quality (92) |
| `codec`, `crf`, `gop`, `fps` | mp4 encoding (`libx264`, 23, 10, reference rate); a video topic is remuxed unless `codec` is set |

Stores accept `sensor_msgs/msg/Image` (`jpeg`, `png`, `npy`, `mp4`),
`CompressedImage` (`jpeg`, `npy`, `mp4`; JPEG data is copied unchanged),
`CompressedVideo` and `FFMPEGPacket` (`jpeg`, `png`, `npy`, `mp4`; decoded RGB), and
`PointCloud2` (`npy`). `ref` and `table` accept any type. Images are stored as RGB
or single-channel pixels; `bgr8` is reordered and alpha is dropped.

`mp4` on a video topic copies the packets without re-encoding: from the keyframe at
or before the scene's first message to its last message, including messages no
frame uses, at the topic's median rate (or `fps`). Its frames.parquet value is
the message index minus that keyframe's. Setting `codec` decodes and re-encodes the
distinct messages instead, as for images.

`tf.base` resolves `T_base_sensor` of each sensor's `header.frame_id` over
`/tf_static`. `tf.dynamic` names the world frame; `T_world_base` is then
interpolated over `/tf` and `/tf_static` at each reference time (header or log
clock, as `time` says), or null outside the recorded range.

A hook is called once per frame with a dictionary of `bag`, `scene_dir`,
`frame`, `timestamp_ns`, and `messages` (sensor name to decoded message or
None) and returns a dictionary of extra `frames.parquet` columns. Files it writes
under `scene_dir` move with the scene.

```text
out/<bag name>/<bag name>_<first frame time, 19 digits>/
  scene.json        format version, spec copy, reference, time source, rate, source bag
                    files and range, tf frames, sensors: topic, type, store, frame_id,
                    video codec, camera intrinsics (K, D, R, P), T_base_sensor, video,
                    schema, stats
  frames.parquet    timestamp_ns, per sensor: <name> path (mp4: video frame index),
                    or <name>_file, _chunk, _record for ref; <name>_offset_ns;
                    T_world_base (16 float64, row-major) with tf.dynamic; hook columns
  <name>.parquet    table sensors: timestamp_ns, log_time_ns, flattened fields
  <name>/000000.jpg one file per distinct message; repeated messages repeat the path
  <name>.mp4        consecutive distinct messages of the scene, or copied packets
```

`<name>_offset_ns` is the chosen message's time minus the frame's time. Where the
sensor has no message within `max_ms`, the path is null and the offset belongs to
the nearest message, so stale and missing frames can be filtered before training.
Sensor stats in `scene.json` report missing frames, reused messages, and offset
percentiles; table stats report count, largest gap, and median rate. A table of
one scene holds rows after the previous scene's last frame up to its own last
frame; the first scene starts at its first frame.

`ref` records the MCAP file relative to the bag directory, the chunk offset
(-1 when unchunked), and the record offset inside it, plus the topic schema in
`scene.json`. The reader maps the file and decodes the message without the bag
index. It reads the bag directory recorded in `source.dir` (the directory of a
single exported `.mcap` file); `SceneDataset(..., bag_root=...)` reads a moved bag,
given a directory holding bags under their recorded directory names or a mapping
from recorded to new directories. Compressed chunks are decompressed whole; record
with `chunkByTopic: true`.

A scene is named after the bag and the time of its first frame (`time` clock,
nanoseconds zero-padded to 19 digits), so names sort by time, a rerun with the same
spec and range rewrites the same scenes, and a run over another time range adds
scenes beside the existing ones. Each scene is written to `.<scene>.partial` and
renamed into place, replacing an existing scene of the same name. Scenes of an
earlier run that this run does not rewrite stay, including scenes that overlap
after `scene_frames` or the range changed; `--clean` (`export(..., clean=True)`)
removes them and leftover partial folders after the export finishes. An
interrupted export leaves complete scenes and a hidden partial folder that a
rerun of the same scene, or `--clean`, removes. Messages are read one frame at a
time.

```python
from rosbag2_dataset.torch import SceneDataset

samples = SceneDataset('out', sensors=['lidar', 'front', 'imu'])
sample = samples[0]   # frames.parquet columns plus scene, frame; front is HxWx3 uint8,
                      # imu is {column: array} of rows in (previous frame, this frame]
```

`SceneDataset` takes scene folders or roots to search, returns NumPy arrays
(None where a frame has no message) and a 4x4 `T_world_base`, and reopens bags and
videos in each worker. mp4 decoding continues forward for sequential reads and
seeks to the previous keyframe otherwise. A `ref` sensor on a video topic returns
the decoded packet message, not pixels.

## Limits

Only `ros2msg` schemas and plain XCDR1 CDR (both byte orders) are decoded. Embedded
dependent definitions, all ROS message primitives, strings/wstrings, nested
messages, fixed arrays, bounded/unbounded sequences, constants, and defaults are
supported. IDL-only schemas, XCDR2, and parameter-list CDR are unsupported;
`raw()` still exposes their bytes. PointField datatypes 1-8 are supported.
Timestamps must fit signed int64. Attachments, metadata, and summary statistics
are skipped; CRCs are not validated. Indexed opening reads summary/message
indexes and top-level record headers, never indexed chunk contents. Missing
indexes trigger scanning of complete records; a killed recorder's incomplete
final record is ignored. Bags must remain unchanged while open. Resident mmap
pages count toward RSS, so zero-copy does not guarantee lower peak RSS.

## Build, tests, and benchmark

```bash
cd <workspace>
colcon build --symlink-install --packages-select rosbag2_dataset
source install/setup.bash
cd <path to rosbag2_dataset>
pytest test --basetemp="$BAG_SCRATCH/pytest"
python3 benchmark/bench_read.py --bag "$BAG_SCRATCH/benchmark"
python3 benchmark/bench_read.py --bag "$BAG_SCRATCH/benchmark" --reuse --touch
```

Set `BAG_SCRATCH` to your writable temporary bag directory and `ROS_LOG_DIR` to a
writable log directory. Export tests are skipped without pyarrow
and Pillow, and the mp4 and video tests without PyAV. Video tests encode H.264, H.265,
and AV1 with PyAV and write CompressedVideo and FFMPEGPacket bags from their message
definitions without those packages. Tests use ROS writers and serializers as
references, without DDS. The benchmark defaults to 1000 1920x1080 rgb8 images plus 2000 Imus,
uncompressed, with three fresh processes per reader. It reports open+full-message
decode time and process peak RSS, including imports in RSS but excluding imports
from timing. Decode-only leaves pixel views lazy; `--touch` additionally sums every
image byte for both readers. See [measured results](benchmark/RESULTS.md).

## License

MIT. See [LICENSE](LICENSE).
