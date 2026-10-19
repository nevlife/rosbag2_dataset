# Copyright 2026 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate a training-sized bag and compare readers in fresh processes."""

import argparse
import json
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time


_DEFAULT_BAG = Path('/tmp/rosbag2_dataset_benchmark')


def generate(path, image_count, imu_count):
    import numpy as np
    from rclpy.serialization import serialize_message
    import rosbag2_py
    from sensor_msgs.msg import Image, Imu

    config = path.parent / (path.name + '_storage.yaml')
    config.write_text('compression: None\nchunkSize: 8388608\nnoChunkCRC: true\n')
    writer = rosbag2_py.SequentialWriter()
    options = rosbag2_py.StorageOptions(uri=str(path), storage_id='mcap',
                                        storage_config_uri=str(config))
    writer.open(options, rosbag2_py.ConverterOptions('', ''))
    for name, typename in [('/image', 'sensor_msgs/msg/Image'), ('/imu', 'sensor_msgs/msg/Imu')]:
        metadata = rosbag2_py.TopicMetadata(
            id=0, name=name, type=typename, serialization_format='cdr')
        writer.create_topic(metadata)
    pixels = np.arange(1920 * 1080 * 3, dtype=np.uint8)
    image = Image(height=1080, width=1920, encoding='rgb8', step=1920 * 3,
                  data=pixels.tobytes())
    image.header.frame_id = 'camera'
    imu = Imu()
    imu.header.frame_id = 'imu'
    imu.linear_acceleration.z = 9.81
    image_bytes, imu_bytes = serialize_message(image), serialize_message(imu)
    schedule = sorted([(i * 1_000_000_000 // 30, '/image', image_bytes)
                       for i in range(image_count)] +
                      [(i * 1_000_000_000 // 60, '/imu', imu_bytes)
                       for i in range(imu_count)], key=lambda row: row[0])
    for stamp, name, data in schedule:
        writer.write(name, data, stamp)
    del writer


def measure(backend, path, touch):
    import numpy as np
    if backend == 'ros':
        from rclpy.serialization import deserialize_message
        import rosbag2_py
        from sensor_msgs.msg import Image, Imu
        classes = {'/image': Image, '/imu': Imu}
    else:
        import rosbag2_dataset

    start = time.perf_counter()
    count, image_count, imu_count, checksum = 0, 0, 0, 0.0

    def consume(name, msg):
        nonlocal count, image_count, imu_count, checksum
        count += 1
        if name == '/image':
            image_count += 1
            checksum += msg.height + msg.width + msg.step
            if touch:
                checksum += int(np.frombuffer(msg.data, dtype=np.uint8).sum(dtype=np.uint64))
        else:
            imu_count += 1
            checksum += msg.linear_acceleration.z

    if backend == 'ros':
        reader = rosbag2_py.SequentialReader()
        reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id='mcap'),
                    rosbag2_py.ConverterOptions('', ''))
        while reader.has_next():
            name, payload, _ = reader.read_next()
            message = deserialize_message(payload, classes[name])
            consume(name, message)
        del reader
    else:
        with rosbag2_dataset.open(path) as bag:
            for name, _, message in bag.read():
                consume(name, message)
    elapsed = time.perf_counter() - start
    return {'backend': backend, 'seconds': elapsed,
            'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            'messages': count, 'images': image_count, 'imus': imu_count,
            'checksum': checksum, 'touch_pixels': touch}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', type=Path, default=_DEFAULT_BAG)
    parser.add_argument('--images', type=int, default=1000)
    parser.add_argument('--imus', type=int, default=2000)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--reuse', action='store_true')
    parser.add_argument('--touch', action='store_true', help='Sum all image bytes after decoding')
    parser.add_argument('--backend', choices=['ros', 'dataset'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.backend:
        print(json.dumps(measure(args.backend, args.bag, args.touch)))
        return
    if not args.reuse:
        if args.bag.exists():
            parser.error('Bag already exists; use --reuse or choose a new --bag path')
        args.bag.parent.mkdir(parents=True, exist_ok=True)
        generate(args.bag, args.images, args.imus)
    runs = []
    for _ in range(args.runs):
        for backend in ('ros', 'dataset'):
            command = [sys.executable, str(Path(__file__).resolve()), '--bag', str(args.bag),
                       '--backend', backend]
            if args.touch:
                command.append('--touch')
            output = subprocess.run(command, check=True, capture_output=True, text=True)
            result = json.loads(output.stdout.strip().splitlines()[-1])
            if result['images'] != args.images or result['imus'] != args.imus:
                raise ValueError('Bag message counts do not match --images and --imus')
            runs.append(result)
            print(json.dumps(result), flush=True)
    if len({(run['messages'], round(run['checksum'], 3)) for run in runs}) != 1:
        raise ValueError('Reader counts or checksums differ')
    summary = {}
    for backend in ('ros', 'dataset'):
        selected = [run for run in runs if run['backend'] == backend]
        summary[backend] = {
            'median_seconds': statistics.median(run['seconds'] for run in selected),
            'median_peak_rss_mib': statistics.median(run['peak_rss_mib'] for run in selected)}
    print(json.dumps({'python': platform.python_version(), 'platform': platform.platform(),
                      'bag_bytes': sum(path.stat().st_size for path in args.bag.glob('*.mcap')),
                      'touch_pixels': args.touch, 'summary': summary}, indent=2))


if __name__ == '__main__':
    main()
