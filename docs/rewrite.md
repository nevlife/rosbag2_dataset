# Rewrite

`rosbag2_dataset.rewrite` writes a new, smaller rosbag2 MCAP bag from a bag: it
keeps or drops topics, thins topics to the messages nearest a reference topic, and
compresses `sensor_msgs/msg/Image` topics to JPEG `CompressedImage` topics. ROS
tools (`ros2 bag info`, `ros2 bag play`) and this package read the result.

It needs `pip install .[rewrite]` (rosbags, PyYAML, Pillow), and the window also
PyQt5 (`pip install .[gui]`). The input is read once in file order without an index
(see [One pass](#one-pass)) and written with `rosbags.rosbag2.Writer` (rosbag2
version 9, MCAP storage).

## Window

```bash
rosbag2_dataset [bag]                 # installed by pip install .[gui]
python -m rosbag2_dataset.gui [bag]
```

1. Open a bag folder or a `.mcap` file. A session folder that holds the bag in a
   `bag/` subfolder is accepted too; the subfolder is used. A folder's `metadata.yaml`
   fills the topic table at once with types, counts, and rates. Without it (a single
   `.mcap` file), the summary statistics of the MCAP files fill the table. A file
   without a summary, such as a recording that did not finish, is counted by walking
   its record headers in a worker thread; the progress bar shows the bytes walked and
   Cancel stops it. Next to the bag path the window shows how the first MCAP file was
   written (see the table below).
2. The Est. size column is the mean size of the first 5 messages of each topic in the
   first 1 GB of the first file, times the count. Reading it walks only record headers
   and ends once every topic has 5 sizes. A topic without messages in that 1 GB shows
   `-`, and the kept total below the table leaves it out. Run waits for this read;
   Cancel stops it and leaves the sizes at `-`.
3. Uncheck Keep to leave a topic out; Keep all and Keep none set every row, and the
   kept total below the table follows. Each `sensor_msgs/msg/Image` topic has a JPEG
   quality (95); `raw` copies it unchanged.
4. To thin topics, check Thin on them and choose the reference topic and the
   tolerance. Time range limits the output to seconds after the bag's first message.
5. Choose an output directory that does not exist and click Run. The progress bar
   shows the input bytes passed of the total; Cancel stops the rewrite and removes the
   partial output.

The layout line reads the MCAP footer, the summary up to its first chunk index, and
the records up to the first chunk or message. "full scan needed" refers to
`rd.open()`, which indexes such a file; rewrite and the window do not:

| Layout                                   | File                                                               |
| ---------------------------------------- | ------------------------------------------------------------------ |
| `chunked zstd` (or `chunked lz4`)        | Compressed chunks with a summary index                             |
| `chunked`                                | Uncompressed chunks with a summary index                           |
| `unchunked (no index, full scan needed)` | Messages outside chunks, as the rosbag2 `fastwrite` preset writes  |
| `chunked, no index (full scan needed)`   | Chunks without a summary, as in a file whose writer did not finish |

`preset: <name>` follows when the recording names the MCAP storage preset: the
`storage_preset` key of `summary.yaml` in the bag folder (rosbag_recorder), or the
`--storage-preset-profile` argument in the `command` list of `manifest.yaml` in the
parent of a `bag/` subfolder.

Show spec shows the spec and the equivalent command line; Save spec writes the spec
to a YAML file for `python -m rosbag2_dataset.rewrite`. The window writes nothing
next to the input bag. The time range counts from the `starting_time` of
`metadata.yaml`, or the first log time of the summary statistics or of the walk.
The window does not index the bag: Run reads it once, as the command line does.

## Command line

```bash
python -m rosbag2_dataset.rewrite spec.yaml /data/run out [--start-ns T] [--end-ns T]
```

| Argument     | Meaning                                 |
| ------------ | --------------------------------------- |
| `spec.yaml`  | Rewrite spec                            |
| `/data/run`  | Bag directory or `.mcap` file           |
| `out`        | Output bag directory; it must not exist |
| `--start-ns` | First log time to keep                  |
| `--end-ns`   | Log time to stop before                 |

The command prints the paths of the written files and a table of each output topic
with its input count in the time range, output count, and output bytes. From Python,
`rewrite(spec, bag, out, *, start_ns=None, end_ns=None, progress=...)` does the same
and returns the table rows as `TopicCount` values; `spec` is a YAML path or a
mapping, and `bag` is a bag directory or a `.mcap` file.
`progress('write', done, total)` is called before each message record with the
input bytes passed (`done`) and the sum of the input file sizes (`total`); an
exception it raises stops the rewrite.

## Spec

```yaml
keep: # or drop: [...]
  - /camera/image_raw
  - /camera/camera_info
  - /velodyne_points
  - /tf
  - /tf_static
jpeg:
  quality: 95
  topics: [/camera/image_raw]
nearest:
  reference: /velodyne_points
  tolerance_ms: 50
  topics: [/camera/image_raw]
```

| Key                    | Meaning                                                                                                                   |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `keep` or `drop`       | Topics to write, or topics to leave out of all the others; give exactly one                                               |
| `jpeg.topics`          | Kept `sensor_msgs/msg/Image` topics written as `<topic>/compressed` of type `sensor_msgs/msg/CompressedImage`             |
| `jpeg.quality`         | JPEG quality of a `jpeg.topics` list (92)                                                                                 |
| `nearest.reference`    | Topic whose messages choose the kept messages; it does not have to be kept                                                |
| `nearest.topics`       | Kept topics thinned to, for each reference message, the one message whose `header.stamp` is nearest the reference's stamp |
| `nearest.tolerance_ms` | A reference message keeps nothing of a topic whose nearest message is farther away                                        |

`jpeg.topics` can also map each topic to its own quality; the window writes this
form:

```yaml
jpeg:
  topics:
    /camera/image_raw: 95
    /camera2/image_raw: 80
```

Unknown keys are rejected. Every named topic must exist in the bag: in
`metadata.yaml`, in the MCAP summaries, or, for a file without either, among the
channels that the walk passed; that last check runs after the walk.

## Messages

1. The time range selects the input of every topic by log time, with an inclusive
   start and exclusive end, as `Bag.read()` does. Latched topics such as
   `/tf_static` keep only the messages logged in the range.
2. `nearest` chooses what `Bag.synced(time_source='header')` chooses over the
   messages in the range (see [Thinning in one pass](#thinning-in-one-pass)). A
   message nearest to several reference messages is written once.
3. Kept messages that are not compressed are copied as CDR bytes without decoding,
   with their log time, schema, serialization format, and offered QoS profiles.
4. `jpeg` decodes each remaining image with `image_to_numpy()` and writes JPEG data
   with the image's header and `format: jpeg`. It takes 8-bit encodings (`rgb8`,
   `bgr8`, `rgba8`, `bgra8`, `mono8`); `bgr8` is reordered to RGB and alpha is
   dropped. The output schema is the `CompressedImage` schema in the first file's
   MCAP summary, or the ROS 2 Jazzy definition when it has none. Only written
   messages are compressed; a thinned image that is not chosen is never decoded.

Messages are written in input order. A kept topic without messages in the range is
not written.

## One pass

Rewrite reads each MCAP file once, the files in name order as `rd.open()` sorts
them, and does not build an index first.

```text
record header (opcode, length)
  Message, topic dropped or outside the range -> skip the body by its length
  Message, topic kept or reference            -> read the body once
  Chunk                                       -> read, decompress (lz4, zstd), walk its records
  Schema, Channel                             -> register
  DataEnd                                     -> end of the file
```

- Message bodies outside chunks that are not needed are not read by this package;
  the disk readahead of the operating system may still read them.
- A chunk is read and decompressed whole, because its records are inside it.
- The read assumes that log times increase in file order, within a file and from
  one split file to the next, as rosbag2 records them. The output keeps the input
  order, and the walk stops at the first message (or chunk start) at or after the
  end of the time range instead of reading the rest of the bag.

Measured on an unchunked 273 GB bag (4 split files, 1.26 M messages) on a hard disk
that reads about 130 MB/s, with 14 kept topics and 3 cameras thinned to a 10 Hz
lidar and written as JPEG: 34 min, 272 GB read from the disk, 180 MB peak resident
memory. The output has the same messages, log times, and bytes as the rewrite that
indexed the bag first, which took 2 h 25 min and read about 1 TB.

### Thinning in one pass

For each thinned topic the rewrite holds the messages that a reference message can
still choose, with their `header.stamp`. The stamp is read from the first 12 bytes
of the CDR body (the encapsulation header, then `sec` and `nanosec`), so the
reference and thinned topics must start with a `std_msgs/Header header` or a
Foxglove `builtin_interfaces/Time timestamp` field; other types are rejected.

1. A reference message with stamp `t` waits, per thinned topic, until that topic
   has a message with a stamp later than `t + tolerance`, or the input ends.
2. Then `SampleRule.select()` chooses among the held messages, with the same ties as
   `align()`: equal distances choose the earlier stamp, equal stamps the first
   message. A chosen message is written.
3. A held message with stamp `s` is decided once the reference has a stamp later
   than `s + tolerance` and every reference with a stamp up to `s + tolerance` has
   chosen. A message that no reference chose is then dropped.

This gives the choice of `align()` over the whole range when the header stamps of
the reference and of each thinned topic do not decrease in file order. A message
that arrives with an older stamp than one already passed can make the choice
differ.

### Memory

Kept messages are written in input order. A held thinned message therefore also
holds every later kept message, as CDR bytes before JPEG, until it is decided. The
held stretch runs from the oldest undecided thinned message to the current message:

| Input                                                 | Held stretch of log time                                                                                           |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Every topic publishes                                 | About 2 x tolerance, plus the longest message interval of the reference and thinned topics, plus their stamp delay |
| The reference or a thinned topic pauses for G seconds | G seconds more                                                                                                     |
| The reference or a thinned topic stops before the end | Up to the end of the range: all later kept messages stay in memory                                                 |

Memory is about the input rate of the kept topics times the held stretch. Thinned
messages that are not chosen are counted until they are dropped.

## Limits

- `rosbags` 0.11.6 writes every message with publish time equal to its log time and
  sequence 0, and stores only `offered_qos_profiles` as channel metadata. Other
  channel metadata of the input is not carried.
- The `type_description_hash` of each topic is computed by rosbags from the schema
  text in the bag.
- A thinned topic that stops publishing keeps the rest of the kept messages in
  memory (see [Memory](#memory)).
