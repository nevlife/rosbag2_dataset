# Copyright 2026 nevlife
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Build rewrite specs from window choices, count topics, and drive the window offscreen."""

import os
import time

import numpy as np
import pytest

pytest.importorskip('rosbags')

import rosbag2_dataset as rd  # noqa: E402
from rosbag2_dataset.rewrite import build_spec  # noqa: E402
from rosbag2_dataset.rewrite import Choice  # noqa: E402
from rosbag2_dataset.rewrite import head_sizes  # noqa: E402
from rosbag2_dataset.rewrite import load_spec  # noqa: E402
from rosbag2_dataset.rewrite import metadata_stats  # noqa: E402
from rosbag2_dataset.rewrite import recorded_preset  # noqa: E402
from rosbag2_dataset.rewrite import summary_stats  # noqa: E402
from rosbag2_dataset.rewrite import walk_stats  # noqa: E402

START = 1_000_000_000
STEP = 100_000_000


def test_build_spec():
    choices = [
        Choice('/lidar', True, thin=True),
        Choice('/cam', True, 80, True),
        Choice('/raw', True, None, True),
        Choice('/off', False, 95, True),
    ]
    spec = build_spec(choices, '/lidar', 20)
    assert spec == {
        'keep': ['/lidar', '/cam', '/raw'],
        'jpeg': {'topics': {'/cam': 80}},
        'nearest': {'reference': '/lidar', 'tolerance_ms': 20, 'topics': ['/cam', '/raw']},
    }
    parsed = load_spec(spec)
    assert parsed.jpeg == {'/cam': 80} and parsed.tolerance_ns == 20_000_000
    assert build_spec(choices[2:], '') == {'keep': ['/raw']}
    assert build_spec([Choice('/lidar', True, thin=True)], '/lidar') == {'keep': ['/lidar']}


@pytest.fixture
def small_events():
    from sensor_msgs.msg import Image
    from sensor_msgs.msg import PointCloud2

    events = []
    for k in range(10):
        cloud = PointCloud2(height=1, width=1, point_step=4, row_step=4, data=bytes(4))
        cloud.header.stamp.sec, cloud.header.stamp.nanosec = divmod(START + k * STEP, 10**9)
        events.append(('/lidar', START + k * STEP, cloud))
    for j in range(20):
        image = Image(height=8, width=8, encoding='rgb8', step=24, data=bytes(range(192)))
        image.header.stamp.sec, image.header.stamp.nanosec = divmod(START + j * STEP // 2, 10**9)
        events.append(('/cam', START + j * STEP // 2 + 1, image))
    events.sort(key=lambda event: event[1])
    return events


@pytest.fixture
def small_bag(write_bag, small_events):
    return write_bag(small_events)


def test_session_folder_resolves_to_bag(small_bag, tmp_path):
    from rosbag2_dataset.gui import _bag_dir

    session = tmp_path / 'session'
    session.mkdir()
    (session / 'bag').symlink_to(small_bag)
    assert _bag_dir(session) == session / 'bag'
    assert _bag_dir(small_bag) == small_bag
    assert _bag_dir(tmp_path / 'missing') == tmp_path / 'missing'


def test_layout_and_recorded_preset(write_bag, small_events, small_bag, tmp_path):
    from rosbag2_dataset._mcap import file_layout
    from rosbag2_dataset.gui import _written
    import yaml

    def first(bag):
        return next(bag.glob('*.mcap'))

    assert file_layout(first(small_bag)) == 'chunked'
    zstd = write_bag(small_events, options='compression: Zstd\n')
    assert file_layout(first(zstd)) == 'chunked zstd'
    fastwrite = write_bag(small_events, options='noChunking: true\nnoSummaryCRC: true\n')
    assert file_layout(first(fastwrite)) == 'unchunked (no index, full scan needed)'
    unfinished = tmp_path / 'unfinished.mcap'
    unfinished.write_bytes(first(small_bag).read_bytes()[:-8])
    assert file_layout(unfinished) == 'chunked, no index (full scan needed)'
    (tmp_path / 'empty.mcap').write_bytes(b'')
    assert _written(tmp_path / 'empty.mcap') == 'layout: unreadable'

    assert recorded_preset(small_bag) == ''
    (small_bag / 'summary.yaml').write_text('session_id: s\nstorage_preset: none\n')
    assert recorded_preset(first(small_bag)) == 'none'
    assert _written(small_bag) == 'layout: chunked   preset: none'
    session = tmp_path / 'session'
    session.mkdir()
    (session / 'bag').symlink_to(fastwrite)
    command = ['ros2', 'bag', 'record', '--storage-preset-profile', 'fastwrite', '/a']
    (session / 'manifest.yaml').write_text(yaml.safe_dump({'command': command}))
    assert _written(session / 'bag') == (
        'layout: unchunked (no index, full scan needed)   preset: fastwrite'
    )


@pytest.mark.parametrize('options', ['', 'noChunking: true\nnoSummaryCRC: true\n'])
def test_stats_without_an_index(write_bag, small_events, options):
    bag = write_bag(small_events, options=options)
    files = sorted(bag.glob('*.mcap'))
    listed, start_ns = metadata_stats(bag)
    assert {name: (s.type, s.count) for name, s in listed.items()} == {
        '/lidar': ('sensor_msgs/msg/PointCloud2', 10),
        '/cam': ('sensor_msgs/msg/Image', 20),
    }
    assert start_ns == START and metadata_stats(files[0]) == ({}, None)
    reports = []
    walked, walked_start = walk_stats(files, lambda *report: reports.append(report))
    summarized, summary_start = summary_stats(files)
    assert walked == summarized and walked_start == summary_start == START
    total = sum(path.stat().st_size for path in files)
    assert reports and {stage for stage, _, _ in reports} == {'count'}
    assert all(0 < done < total == t for _, done, t in reports)
    duration = (START + 19 * STEP // 2 + 1 - START) / 1e9
    for name, count in (('/lidar', 10), ('/cam', 20)):
        assert walked[name].count == listed[name].count == count
        assert np.isclose(walked[name].hz, count / duration)
    with rd.open(bag) as source:
        sizes = {name: len(source.topic(name).raw(0)[1]) for name in source.topics}
    assert head_sizes(files[0]) == sizes
    assert head_sizes(files[0], limit=0) == {}

    def stop(stage, done, total):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        walk_stats(files, stop)


def _wait(app, window):
    deadline = time.monotonic() + 30
    while window.busy() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()


def test_window_rewrites(small_bag, tmp_path, monkeypatch):
    pytest.importorskip('PyQt5')
    monkeypatch.setenv('QT_QPA_PLATFORM', os.environ.get('QT_QPA_PLATFORM', 'offscreen'))
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtWidgets import QPushButton

    from rosbag2_dataset import gui
    from rosbag2_dataset.gui import QUALITY
    from rosbag2_dataset.gui import RewriteWindow
    from rosbag2_dataset.gui import THIN

    app = QApplication.instance() or QApplication([])
    window = RewriteWindow()
    window.load(str(small_bag))
    _wait(app, window)
    assert window.stats is not None and window.table.rowCount() == 2
    assert window.start_ns == START
    assert window.written.text() == 'layout: chunked'
    buttons = {button.text(): button for button in window.findChildren(QPushButton)}
    buttons['Keep none'].click()
    assert window.total.text() == 'Kept: 0 topics, 0 B before JPEG and thinning'
    assert window.spec() == {'keep': []}
    buttons['Keep all'].click()
    assert window.total.text().startswith('Kept: 2 topics, ')
    rows = {window.table.item(row, 1).text(): row for row in range(2)}
    assert window.table.cellWidget(rows['/lidar'], QUALITY) is None
    window.table.cellWidget(rows['/cam'], QUALITY).setValue(70)
    window.table.item(rows['/cam'], THIN).setCheckState(Qt.Checked)
    window.reference.setCurrentText('/lidar')
    window.tolerance.setValue(10)
    out = tmp_path / 'out'
    window.output.setText(str(out))
    assert window.spec() == {
        'keep': ['/cam', '/lidar'],
        'jpeg': {'topics': {'/cam': 70}},
        'nearest': {'reference': '/lidar', 'tolerance_ms': 10, 'topics': ['/cam']},
    }
    assert window.command() == (
        f'python -m rosbag2_dataset.rewrite spec.yaml {small_bag} {out}'
    )
    window.show_spec().close()
    window.run()
    _wait(app, window)
    assert window.status.text().startswith('Wrote 2 topics'), window.status.text()
    with rd.open(out) as result:
        assert {name: info.count for name, info in result.topics.items()} == {
            '/cam/compressed': 10,
            '/lidar': 10,
        }

    calls = []

    def cancel_on_the_fourth(worker, stage, done, total):
        calls.append(stage)
        if calls.count('write') == 4:
            raise gui.Cancelled

    monkeypatch.setattr(gui.Worker, 'progress', cancel_on_the_fourth)
    window.output.setText(str(tmp_path / 'cancelled'))
    window.run()
    _wait(app, window)
    assert window.status.text() == 'Rewrite failed: cancelled'
    assert not (tmp_path / 'cancelled').exists()
    window.close()


def test_window_counts_a_file_without_summary(write_bag, small_events, tmp_path, monkeypatch):
    pytest.importorskip('PyQt5')
    monkeypatch.setenv('QT_QPA_PLATFORM', os.environ.get('QT_QPA_PLATFORM', 'offscreen'))
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    from rosbag2_dataset.gui import COUNT
    from rosbag2_dataset.gui import RewriteWindow
    from rosbag2_dataset.gui import SIZE

    fastwrite = write_bag(small_events, options='noChunking: true\nnoSummaryCRC: true\n')
    unfinished = tmp_path / 'unfinished.mcap'
    unfinished.write_bytes(next(fastwrite.glob('*.mcap')).read_bytes()[:-8])
    app = QApplication.instance() or QApplication([])
    window = RewriteWindow()
    window.load(str(unfinished))
    assert window.stats is None and window.table.rowCount() == 0
    _wait(app, window)
    rows = {window.table.item(row, 1).text(): row for row in range(window.table.rowCount())}
    assert {name: window.table.item(row, COUNT).text() for name, row in rows.items()} == {
        '/lidar': '10',
        '/cam': '20',
    }
    assert window.table.item(rows['/cam'], SIZE).data(Qt.UserRole) > 20 * 192
    window.ranged.setChecked(True)
    window.end_s.setValue(0.5)
    out = tmp_path / 'out'
    window.output.setText(str(out))
    assert window.command().endswith(f'--start-ns {START} --end-ns {START + STEP * 5}')
    window.run()
    _wait(app, window)
    assert window.status.text().startswith('Wrote 2 topics'), window.status.text()
    assert window.progress.format() == 'done'
    with rd.open(out) as result:
        assert {name: info.count for name, info in result.topics.items()} == {
            '/cam/compressed': 10,
            '/lidar': 5,
        }
    window.close()
