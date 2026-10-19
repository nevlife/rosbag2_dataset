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

"""Read rosbag2 MCAP datasets without a ROS runtime."""

from .bag import Bag, Synced, Topic, TopicInfo
from .numpy import image_to_numpy, pointcloud2_to_numpy


__all__ = ['Bag', 'Synced', 'Topic', 'TopicInfo', 'open',
           'image_to_numpy', 'pointcloud2_to_numpy']


def open(path, *, chunk_cache_size=4):  # noqa: A001
    """Open an MCAP file or split bag directory; write no cache files."""
    return Bag(path, chunk_cache_size=chunk_cache_size)
