<!-- markdownlint-disable MD013 Entries are one line each -->

# rosbag2_dataset v?.?.?

Update this file with every change. At a release, rename it to `v<version>.md`,
fill in the version, copy [template.md](template.md) to a new `unreleased.md`, and
add the same entries to `CHANGELOG.rst`.

## [v?.?.?](https://github.com/nevlife/rosbag2_dataset/tree/v?.?.?)

### Features

* Read rosbag2 MCAP files and split bag directories without ROS: indexed topics, nearest-time lookup, raw CDR views, merged reads by log time
* Decode `ros2msg` schemas and XCDR1 CDR into messages whose numeric arrays are read-only NumPy views of the mmap or of decompressed chunks
* `Bag.synced()` aligns topics to a reference by log time or `header.stamp`, with per-topic tolerances and optional topics
* `image_to_numpy()` and `pointcloud2_to_numpy()` views; `BagDataset` map-style dataset for PyTorch workers
* `Bag.video()` decodes H.264, H.265, and AV1 from `CompressedVideo` and `FFMPEGPacket` topics
* `rosbag2_dataset.export` writes scene folders from a spec: jpeg, png, npy, mp4, ref, and table stores, transforms, hooks
* `SampleRule` and `Window`: one alignment rule with past and future windows for `Bag.synced()` and `BagDataset`; `LiveSampler` applies it to messages as they arrive and gives the samples the recorded bag gives
* `hold` topics of `SampleRule` give their last message before the reference; `LiveSampler.early` counts samples released by the wait
* `SceneDataset` reads exported scenes, including `ref` sensors from moved bags (`bag_root`)
* Scenes are named by bag and first frame time; `--clean` removes scenes a run did not write

### Bugfixes

* `BagDataset` with `sync` returned the last message for an optional topic beyond tolerance; it is now `None`. Synced samples also carry `'@time'`

### Refactoring

### Workflow

* Moved out of the rosbag2 fork into its own repository with its history; relicensed to MIT
* Split the README into an entry page and `docs/`; added release notes, `CHANGELOG.rst`, and a rosdoc2 configuration

### New API features

### API Breaking Changes

<!-- markdownlint-enable MD013 -->
