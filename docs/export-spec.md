# Export spec

`rosbag2_dataset.export` writes a bag into self-describing scene folders that
training code reads without ROS or the bag reader. The folder layout is described
in [Scene format](scene-format.md).

It needs `pip install .[export]` (pyarrow, PyYAML, Pillow); `store: mp4` also needs
`.[video]` (PyAV). Reading bags stays NumPy-only.

## Command line

```bash
python -m rosbag2_dataset.export spec.yaml /data/run out [--start-ns T] [--end-ns T] [--clean]
```

| Argument     | Meaning                                             |
| ------------ | --------------------------------------------------- |
| `spec.yaml`  | Export spec                                         |
| `/data/run`  | Bag directory or `.mcap` file                       |
| `out`        | Output directory; scenes go to `out/<bag name>/`    |
| `--start-ns` | First reference time to export                      |
| `--end-ns`   | Reference time to stop before                       |
| `--clean`    | Remove the bag's scenes that this run did not write |

The command prints each scene path as it is written. From Python,
`export(spec, bag, out, *, start_ns=None, end_ns=None, clean=False)` does the same
and returns the scene paths; `spec` is a YAML path or a mapping.

## Spec

```yaml
reference: lidar # sensor name or topic; one frame per reference message
scene_frames: 300 # frames per scene; the last scene may be shorter
time: header # header (header.stamp) or log (log time)
sensors:
  lidar:
    {
      topic: /velodyne_points,
      store: npy,
      fields: [x, y, z, intensity],
      finite: true,
    }
  front: { topic: /camera/image_raw, sync: nearest, max_ms: 50, store: jpeg }
  depth: { topic: /camera/depth, max_ms: 50, store: png, required: true }
  imu: { topic: /imu, store: table }
tf: { base: base_link, static: true, dynamic: odom }
hooks: my_package.export:extra # optional, one or a list of module:function
```

Unknown keys are rejected. The reference sensor cannot use `store: table`.

## Sensor keys

| Sensor key                   | Meaning                                                                                                                                                  |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `topic`                      | Topic to export; each topic appears in one sensor only                                                                                                   |
| `store`                      | `jpeg`, `png`, `npy` (one file per message), `mp4` (one video per scene), `ref` (bag location, no copy), `table` (native-rate Parquet)                   |
| `sync`, `max_ms`             | Nearest message to the reference time (`nearest` is the only `sync`); beyond `max_ms` the frame has no value for this sensor. No `max_ms` means no limit |
| `required`                   | Drop frames where this sensor has no message within `max_ms`                                                                                             |
| `time`                       | Clock of this sensor when it differs from the spec's `time`                                                                                              |
| `fields`, `finite`           | PointCloud2 fields as N x C columns of one dtype (default: all scalar fields); `finite` drops points with any non-finite field                           |
| `info`                       | CameraInfo topic; default is the sibling `camera_info` of an image topic                                                                                 |
| `quality`                    | JPEG quality (92)                                                                                                                                        |
| `codec`, `crf`, `gop`, `fps` | mp4 encoding (`libx264`, 23, 10, reference rate); a video topic is remuxed unless `codec` is set                                                         |

## Stores and message types

| Message type                      | Stores                                               |
| --------------------------------- | ---------------------------------------------------- |
| `sensor_msgs/msg/Image`           | `jpeg`, `png`, `npy`, `mp4`                          |
| `sensor_msgs/msg/CompressedImage` | `jpeg` (JPEG data is copied unchanged), `npy`, `mp4` |
| `CompressedVideo`, `FFMPEGPacket` | `jpeg`, `png`, `npy`, `mp4` (decoded RGB)            |
| `sensor_msgs/msg/PointCloud2`     | `npy`                                                |
| Any type                          | `ref`, `table`                                       |

Images are stored as RGB or single-channel pixels; `bgr8` is reordered and alpha
is dropped.

`mp4` on a video topic copies the packets without re-encoding: from the keyframe at
or before the scene's first message to its last message, including messages no
frame uses, at the topic's median rate (or `fps`). Its `frames.parquet` value is
the message index minus that keyframe's. Setting `codec` decodes and re-encodes the
distinct messages instead, as for images.

## Transforms

`tf.base` resolves `T_base_sensor` of each sensor's `header.frame_id` over
`/tf_static`. `tf.dynamic` names the world frame; `T_world_base` is then
interpolated over `/tf` and `/tf_static` at each reference time (header or log
clock, as `time` says), or null outside the recorded range.

## Hooks

A hook is called once per frame with a dictionary of `bag`, `scene_dir`,
`frame`, `timestamp_ns`, and `messages` (sensor name to decoded message or
None) and returns a dictionary of extra `frames.parquet` columns. Files it writes
under `scene_dir` move with the scene.
