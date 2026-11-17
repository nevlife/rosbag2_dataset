# rosbag2_dataset

Read rosbag2 MCAP bags for ML training with Python and NumPy, without ROS.
Random access, time alignment, and read-only array views avoid per-byte Python
objects and repeated image/point-cloud copies. The reader implements the
[MCAP specification](https://mcap.dev/spec) directly; no MCAP Python SDK is needed.
`rosbag2_dataset.export` writes bags into self-describing scene folders that
training code reads back without the bag. `rosbag2_dataset.rewrite` writes a smaller
rosbag2 bag with selected topics, thinned topics, and JPEG images.

## Supported environment

| Item   | Support                                                                                          |
| ------ | ------------------------------------------------------------------------------------------------ |
| OS     | Ubuntu 24.04                                                                                     |
| Python | 3.10 or newer, NumPy                                                                             |
| ROS 2  | Jazzy for building as a ROS package and for the tests; reading and exporting need no ROS runtime |
| Bags   | rosbag2 MCAP files or split bag directories, `ros2msg` schemas, XCDR1 CDR                        |

Optional modules: `lz4` and `zstandard` for compressed chunks, `pyarrow`, `PyYAML`,
and `Pillow` for export (`.[export]`), `rosbags`, `PyYAML`, and `Pillow` for rewrite
(`.[rewrite]`), the same and PyQt5 for the rewrite window (`.[gui]`), PyAV for video
(`.[video]`), and PyTorch only for your own training code.

## Install

```bash
cd <workspace>
colcon build --symlink-install --packages-select rosbag2_dataset
source install/setup.bash
```

Without ROS, `pip install .` (with `.[export]`, `.[rewrite]`, `.[gui]`, `.[video]`, `.[lz4]`, `.[zstd]` as needed)
installs the same Python package, and `.[gui]` also installs the `rosbag2_dataset` command.

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

To make a smaller bag, open the rewrite window, choose the bag, the topics, JPEG
qualities, thinning, and an output directory, and click Run:

```bash
rosbag2_dataset /data/run          # after pip install .[gui]
python -m rosbag2_dataset.gui /data/run
```

The window shows the spec it runs (Show spec, Save spec). Scripts run the same spec,
and export has its own:

```bash
python -m rosbag2_dataset.rewrite rewrite.yaml /data/run /data/run_small
python -m rosbag2_dataset.export spec.yaml /data/run out
```

## Documentation

| Document                                          | Content                                                                           |
| ------------------------------------------------- | --------------------------------------------------------------------------------- |
| [Usage](docs/usage.md)                            | Reading, time alignment, memory behaviour, NumPy helpers, `BagDataset`, limits    |
| [Video](docs/video.md)                            | Decoding `CompressedVideo` and `FFMPEGPacket` topics                              |
| [Export spec](docs/export-spec.md)                | Export CLI, spec keys, stores, transforms, hooks                                  |
| [Rewrite](docs/rewrite.md)                        | Rewrite window, CLI and spec: kept topics, nearest thinning, JPEG images, limits  |
| [Scene format](docs/scene-format.md)              | Scene folder layout, columns, references, reruns, `SceneDataset`                  |
| [Testing](docs/testing.md)                        | Tests and the reader benchmark                                                    |
| [Release notes](docs/release-notes/unreleased.md) | Changes since the last release; [CHANGELOG.rst](CHANGELOG.rst) per ROS convention |

The Python API reference is generated from docstrings with
[rosdoc2](https://github.com/ros-infrastructure/rosdoc2) (`rosdoc2 build -p .`);
`rosdoc2.yaml` configures it and `doc` links to `docs/` so these pages are included.

## License

MIT. See [LICENSE](LICENSE).
