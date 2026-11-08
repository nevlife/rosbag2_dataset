# rosbag2_dataset

Read rosbag2 MCAP bags for ML training with Python and NumPy, without ROS.
Random access, time alignment, and read-only array views avoid per-byte Python
objects and repeated image/point-cloud copies. The reader implements the
[MCAP specification](https://mcap.dev/spec) directly; no MCAP Python SDK is needed.
`rosbag2_dataset.export` writes bags into self-describing scene folders that
training code reads back without the bag.

## Supported environment

| Item   | Support                                                                                          |
| ------ | ------------------------------------------------------------------------------------------------ |
| OS     | Ubuntu 24.04                                                                                     |
| Python | 3.10 or newer, NumPy                                                                             |
| ROS 2  | Jazzy for building as a ROS package and for the tests; reading and exporting need no ROS runtime |
| Bags   | rosbag2 MCAP files or split bag directories, `ros2msg` schemas, XCDR1 CDR                        |

Optional modules: `lz4` and `zstandard` for compressed chunks, `pyarrow`, `PyYAML`,
and `Pillow` for export (`.[export]`), PyAV for video (`.[video]`), and PyTorch only
for your own training code.

## Install

```bash
cd <workspace>
colcon build --symlink-install --packages-select rosbag2_dataset
source install/setup.bash
```

Without ROS, `pip install .` (with `.[export]`, `.[video]`, `.[lz4]`, `.[zstd]` as needed)
installs the same Python package.

## Quick start

```python
import rosbag2_dataset as rd

with rd.open('/data/run') as bag:  # Also accepts a single .mcap file.
    print(bag.topics)             # name -> TopicInfo(type, count, start_ns, end_ns)
    camera = bag.topic('/camera/image')
    time_ns, msg = camera[camera.nearest(123456789)]
    pixels = rd.image_to_numpy(msg)
    pairs = bag.synced(['/camera/image', '/lidar/points'],
                       reference='/camera/image', tolerance_ns=20_000_000)
```

```bash
python -m rosbag2_dataset.export spec.yaml /data/run out
```

## Documentation

| Document                                          | Content                                                                           |
| ------------------------------------------------- | --------------------------------------------------------------------------------- |
| [Usage](docs/usage.md)                            | Reading, time alignment, memory behaviour, NumPy helpers, `BagDataset`, limits    |
| [Video](docs/video.md)                            | Decoding `CompressedVideo` and `FFMPEGPacket` topics                              |
| [Export spec](docs/export-spec.md)                | Export CLI, spec keys, stores, transforms, hooks                                  |
| [Scene format](docs/scene-format.md)              | Scene folder layout, columns, references, reruns, `SceneDataset`                  |
| [Testing](docs/testing.md)                        | Tests and the reader benchmark                                                    |
| [Release notes](docs/release-notes/unreleased.md) | Changes since the last release; [CHANGELOG.rst](CHANGELOG.rst) per ROS convention |

The Python API reference is generated from docstrings with
[rosdoc2](https://github.com/ros-infrastructure/rosdoc2) (`rosdoc2 build -p .`);
`rosdoc2.yaml` configures it and `doc` links to `docs/` so these pages are included.

## License

MIT. See [LICENSE](LICENSE).
