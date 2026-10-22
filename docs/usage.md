# Usage

This page covers reading bags. The generated API reference lists every class and
signature ([how to build it](testing.md)); this page describes
behaviour and caveats.

## Open a bag and read messages

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
```

- A directory includes all its `*.mcap` files without requiring PyYAML.
- `read()` merges selected topics in log-time order with an inclusive start and
  exclusive end.
- Topic indexing supports negative indices.
- Nearest-time ties select the earlier timestamp and then its first message.

## Time alignment

```python
pairs = bag.synced(['/camera/image', '/lidar/points'],
                   reference='/camera/image', tolerance_ns=20_000_000)
camera_index, lidar_index = pairs[0]
```

Synchronization returns tuples in the requested topic order, retains reference
log-time order, permits reuse of matching messages, and drops groups beyond
tolerance.

| Option | Behaviour |
| --- | --- |
| `reference` | Topic that sets the rows; defaults to the first topic |
| `tolerance_ns` | One value, or a mapping from each non-reference topic to its own tolerance |
| `time_source='header'` | Uses `header.stamp`; decodes each message once while building alignment and supports stamps recorded out of order |
| `optional` | Topics that keep the row and get index -1 when beyond tolerance |

`pairs.times` holds the clock of each chosen message (for -1, the nearest one).

## Memory behaviour

- Uncompressed CDR arrays view the read-only mmap.
- Compressed arrays view immutable decompressed bytes. A per-file LRU retains four
  chunks by default (`open(..., chunk_cache_size=4)`).
- Install `lz4` for lz4 chunks or `zstandard` for zstd chunks. Missing modules
  produce an explicit error when a chunk is accessed.
- A compressed chunk is decompressed whole even when only one of its topics is read.
  Record with the MCAP storage option `chunkByTopic: true` so that each chunk holds
  one topic.
- Exported arrays and raw views remain valid after closing a bag or evicting a
  chunk; release those views to release their storage.
- Strings, scalar values, nested objects, and the in-memory index require
  allocations. No files are written beside bags.
- Resident mmap pages count toward RSS, so zero-copy does not guarantee lower peak
  RSS.

## NumPy helpers

`image_to_numpy()` returns HxW or HxWxC views for rgb8, bgr8, rgba8, bgra8, mono8,
mono16, 16UC1, and 32FC1. `pointcloud2_to_numpy()` returns an HxW structured view
with named fields and subarrays for fields with count > 1. Both preserve byte
order and row padding; images preserve channel order.

## PyTorch datasets

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
PyTorch.

- With `sync=None`, samples contain one message each in merged log-time order
  within each bag; bags are concatenated in path order.
- Dataset serialization contains paths, options, and counts, never mappings.
  Workers reopen their own bags after fork or spawn.
- Use a top-level transform that returns tensors or ordinary collatable values.
  Generated Message objects need a transform or custom collation for
  multiprocessing.
- Read-only NumPy views need a copy before tensor mutation.
- Call `samples.close()` or use it as a context manager when finished.

`SceneDataset` reads exported scenes; see [Scene format](scene-format.md).

## Limits

- Only `ros2msg` schemas and plain XCDR1 CDR (both byte orders) are decoded.
  Embedded dependent definitions, all ROS message primitives, strings/wstrings,
  nested messages, fixed arrays, bounded/unbounded sequences, constants, and
  defaults are supported.
- IDL-only schemas, XCDR2, and parameter-list CDR are unsupported; `raw()` still
  exposes their bytes.
- PointField datatypes 1-8 are supported.
- Timestamps must fit signed int64.
- Attachments, metadata, and summary statistics are skipped; CRCs are not
  validated.
- Indexed opening reads summary/message indexes and top-level record headers,
  never indexed chunk contents. Missing indexes trigger scanning of complete
  records; a killed recorder's incomplete final record is ignored.
- Bags must remain unchanged while open.
