# Testing

## Tests

```bash
cd <workspace>
colcon build --symlink-install --packages-select rosbag2_dataset
source install/setup.bash
cd <path to rosbag2_dataset>
pytest test --basetemp="$BAG_SCRATCH/pytest"
```

Set `BAG_SCRATCH` to your writable temporary bag directory and `ROS_LOG_DIR` to a
writable log directory.

- Tests use ROS writers and serializers as references, without DDS.
- Export tests are skipped without pyarrow and Pillow, and the mp4 and video tests
  without PyAV.
- Video tests encode H.264, H.265, and AV1 with PyAV and write CompressedVideo and
  FFMPEGPacket bags from their message definitions without those packages.
- `test_lint.py` runs the ament flake8, pep257, and copyright checks.

## Benchmark

```bash
python3 benchmark/bench_read.py --bag "$BAG_SCRATCH/benchmark"
python3 benchmark/bench_read.py --bag "$BAG_SCRATCH/benchmark" --reuse --touch
```

The benchmark defaults to 1000 1920x1080 rgb8 images plus 2000 Imus,
uncompressed, with three fresh processes per reader. It reports open+full-message
decode time and process peak RSS, including imports in RSS but excluding imports
from timing. Decode-only leaves pixel views lazy; `--touch` additionally sums every
image byte for both readers. See [measured results](../benchmark/RESULTS.md).

## API reference

The Python API reference is generated from docstrings with rosdoc2, which also
renders these pages (`doc` links to `docs/`). Install rosdoc2 in a virtual
environment and build outside the source tree:

```bash
rosdoc2 build -p <path to rosbag2_dataset> -o <output dir> -d <work dir>
```

The entry page is `<output dir>/rosbag2_dataset/index.html`.
