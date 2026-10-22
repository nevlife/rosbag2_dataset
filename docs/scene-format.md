# Scene format

An export writes one folder per scene. Scenes are written by
`rosbag2_dataset.export` from a [spec](export-spec.md) and read back by
`SceneDataset` without ROS or the bag reader.

## Layout

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

## Frames and statistics

- `<name>_offset_ns` is the chosen message's time minus the frame's time.
- Where the sensor has no message within `max_ms`, the path is null and the offset
  belongs to the nearest message, so stale and missing frames can be filtered
  before training.
- Sensor stats in `scene.json` report missing frames, reused messages, and offset
  percentiles. Table stats report count, largest gap, and median rate.
- A table of one scene holds rows after the previous scene's last frame up to its
  own last frame; the first scene starts at its first frame.

## Bag references

`ref` records the MCAP file relative to the bag directory, the chunk offset (-1
when unchunked), and the record offset inside it, plus the topic schema in
`scene.json`. The reader maps the file and decodes the message without the bag
index.

It reads the bag directory recorded in `source.dir` (the directory of a single
exported `.mcap` file). `SceneDataset(..., bag_root=...)` reads a moved bag, given
a directory holding bags under their recorded directory names or a mapping from
recorded to new directories. Compressed chunks are decompressed whole; record with
`chunkByTopic: true`.

## Names and reruns

A scene is named after the bag and the time of its first frame (`time` clock,
nanoseconds zero-padded to 19 digits).

- Names sort by time.
- A rerun with the same spec and range rewrites the same scenes, and a run over
  another time range adds scenes beside the existing ones.
- Each scene is written to `.<scene>.partial` and renamed into place, replacing an
  existing scene of the same name.
- Scenes of an earlier run that this run does not rewrite stay, including scenes
  that overlap after `scene_frames` or the range changed. `--clean`
  (`export(..., clean=True)`) removes them and leftover partial folders after the
  export finishes.
- An interrupted export leaves complete scenes and a hidden partial folder that a
  rerun of the same scene, or `--clean`, removes.
- Messages are read one frame at a time.

## Reading scenes

```python
from rosbag2_dataset.torch import SceneDataset

samples = SceneDataset('out', sensors=['lidar', 'front', 'imu'])
sample = samples[0]   # frames.parquet columns plus scene, frame; front is HxWx3 uint8,
                      # imu is {column: array} of rows in (previous frame, this frame]
```

- `SceneDataset` takes scene folders or roots to search and returns NumPy arrays
  (None where a frame has no message) and a 4x4 `T_world_base`.
- It reopens bags and videos in each worker.
- mp4 decoding continues forward for sequential reads and seeks to the previous
  keyframe otherwise.
- A `ref` sensor on a video topic returns the decoded packet message, not pixels.
