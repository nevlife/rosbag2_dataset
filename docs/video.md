# Video topics

`bag.video()` decodes H.264, H.265, and AV1 topics into RGB frames with PyAV
(`pip install .[video]`).

```python
with rd.open('/data/run') as bag:
    video = bag.video('/camera/image_raw/foxglove')
    time_ns, rgb = video[video.nearest(123456789)]   # HxWx3 uint8
```

## Supported messages

| Message type | Recorded by | Codec taken from |
| --- | --- | --- |
| `foxglove_msgs/msg/CompressedVideo` | `foxglove` image_transport, rosbag2_broll | `format` |
| `ffmpeg_image_transport_msgs/msg/FFMPEGPacket` | `ffmpeg_image_transport` | first token of `encoding` |

The bag does not need these message packages installed; their definitions come
from the bag.

## Decoding rules

- Each message must hold one frame and the stream must have no B-frames, as
  Foxglove requires.
- Reading a message decodes from the keyframe at or before it. Reading the next
  message continues the decoder.
- Keyframes are the FFMPEGPacket key flag, or in CompressedVideo an H.264 IDR, an
  H.265 IRAP picture, or an AV1 sequence header.
- Messages before the first keyframe raise an error.
- `Topic.stamps('header')` and `synced(..., time_source='header')` use the
  Foxglove `timestamp` field for CompressedVideo.
- Decoders run single-threaded.

Exporting video topics (re-encoding or copying packets into mp4) is described in
[Export spec](export-spec.md).
