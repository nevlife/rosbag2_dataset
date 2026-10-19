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

from setuptools import find_packages, setup


package_name = 'rosbag2_dataset'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md', 'LICENSE']),
        ('share/' + package_name + '/benchmark',
         ['benchmark/bench_read.py', 'benchmark/RESULTS.md']),
    ],
    install_requires=['setuptools', 'numpy'],
    extras_require={'lz4': ['lz4'], 'zstd': ['zstandard'], 'torch': ['torch']},
    python_requires='>=3.10',
    zip_safe=False,
    maintainer='rosbag2 maintainers',
    maintainer_email='ros-tooling@googlegroups.com',
    description='Indexed NumPy views of rosbag2 MCAP messages without ROS',
    license='Apache-2.0',
    tests_require=['pytest'],
)
