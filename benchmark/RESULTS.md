# Reader benchmark

Measured on 2026-10-20 with Python 3.12.3, NumPy 1.26.4, ROS 2 Jazzy,
Linux 7.0.0-38 x86_64, and an Intel Core i7-14700K (32 GB RAM).
The uncompressed MCAP bag is 6,221,723,775 bytes: 1000 1920x1080 rgb8 images
and 2000 Imu messages. The storage chunk target is 8 MiB. Page caches were warm;
no caches were flushed between runs.

Each number is the median of three fresh worker processes. Time includes bag
opening, indexing, schema compilation, iteration, and decoding of all 3000
messages; imports and bag generation are excluded. Peak RSS uses Linux
`getrusage(RUSAGE_SELF).ru_maxrss` and includes imports and resident mmap pages.
Messages are consumed and released individually by both readers.

| Workload | Reader | Seconds | Peak RSS (MiB) |
| --- | --- | ---: | ---: |
| Decode all messages | rosbag2_py + rclpy | 9.539 | 121.2 |
| Decode all messages | rosbag2_dataset | 0.0210 | 1066.4 |
| Decode + sum every image byte | rosbag2_py + rclpy | 10.402 | 115.6 |
| Decode + sum every image byte | rosbag2_dataset | 1.448 | 5967.9 |

Decode-only leaves pixel buffers lazy in rosbag2_dataset; the ROS reader copies
and deserializes their bytes. The second workload forces both readers to access
every pixel byte and is 7.2 times faster here. Message counts and checksums agree
for every run. These warm-cache results do not predict cold-storage throughput.

Zero-copy does not imply lower RSS: the mmap reader retains resident file pages
until the mapping is released or the kernel reclaims them. This implementation
does not advise eviction during iteration. The ROS reader releases/reuses heap
buffers as it goes, so its RSS is lower in this workload. Neither the benchmark
nor the package claims an RSS reduction.

Reproduce from a sourced workspace (set a writable ROS_LOG_DIR):

```bash
python3 benchmark/bench_read.py --bag "$BAG_SCRATCH/benchmark" --runs 3
python3 benchmark/bench_read.py --bag "$BAG_SCRATCH/benchmark" --reuse --touch --runs 3
```

The runtime has no zstandard or torch installed. The dataset's fork/spawn and
pickle behavior was exercised with multiprocessing; a real PyTorch DataLoader
run remains untested in this environment.
