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

"""
Rewrite a bag from a window: it shows the topics and runs rosbag2_dataset.rewrite.

python -m rosbag2_dataset.gui [bag]
"""

import argparse
from dataclasses import replace
from pathlib import Path
import shutil
import sys
import time

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QCheckBox
from PyQt5.QtWidgets import QComboBox
from PyQt5.QtWidgets import QDialog
from PyQt5.QtWidgets import QDoubleSpinBox
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QHeaderView
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QPlainTextEdit
from PyQt5.QtWidgets import QProgressBar
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QSpinBox
from PyQt5.QtWidgets import QTableWidget
from PyQt5.QtWidgets import QTableWidgetItem
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QWidget
import yaml

from ._mcap import file_layout
from ._scene import IMAGE
from .rewrite import build_spec
from .rewrite import Choice
from .rewrite import head_sizes
from .rewrite import metadata_stats
from .rewrite import recorded_preset
from .rewrite import rewrite
from .rewrite import summary_stats
from .rewrite import walk_stats

COLUMNS = ('Keep', 'Topic', 'Type', 'Count', 'Hz', 'Est. size', 'JPEG quality', 'Thin')
KEEP, TOPIC, TYPE, COUNT, HZ, SIZE, QUALITY, THIN = range(len(COLUMNS))
DEFAULT_QUALITY = 95


class Cancelled(Exception):
    """Raised from a progress callback to stop the work."""


class Worker(QThread):
    """Run work(progress) off the GUI thread; progress arrives at most every 0.1 s."""

    progressed = pyqtSignal(str, float, float)
    completed = pyqtSignal(object, str)

    def __init__(self, work):
        super().__init__()
        self.work = work
        self.cancelled = False
        self.last = 0.0

    def progress(self, stage, done, total):
        if self.cancelled:
            raise Cancelled
        now = time.monotonic()
        if now - self.last >= 0.1:
            self.last = now
            self.progressed.emit(stage, done, total)

    def run(self):
        try:
            self.completed.emit(self.work(self.progress), '')
        except Cancelled:
            self.completed.emit(None, 'cancelled')
        except Exception as error:
            self.completed.emit(None, f'{type(error).__name__}: {error}')


def _size(value):
    if value is None:
        return '-'
    for unit in ('B', 'KB', 'MB', 'GB'):
        if value < 1000 or unit == 'GB':
            return f'{value:.0f} {unit}' if unit == 'B' else f'{value:.1f} {unit}'
        value /= 1000


def _bag_dir(path):
    # Older recorder sessions hold the bag in <session>/bag; accept the session folder too.
    if path.is_dir() and not any(path.glob('*.mcap')) and any((path / 'bag').glob('*.mcap')):
        return path / 'bag'
    return path


def _mcap_files(path):
    return sorted(path.glob('*.mcap')) if path.is_dir() else [path]


def _written(path):
    """Describe the first MCAP file's layout and the preset that the session names."""
    files = _mcap_files(path)
    if not files or not files[0].is_file():
        return ''
    try:
        text = f'layout: {file_layout(files[0])}'
    except ValueError:
        text = 'layout: unreadable'
    preset = recorded_preset(path)
    return f'{text}   preset: {preset}' if preset else text


def _listed(path):
    """Return (stats, first log time) from metadata.yaml or the MCAP summaries, or ({}, None)."""
    stats, start_ns = metadata_stats(path)
    if stats:
        return stats, start_ns
    files = _mcap_files(path)
    if not all(file.is_file() for file in files):
        return {}, None
    return summary_stats(files)


def _estimate(path, stats, start_ns):
    """Count by a header walk when nothing lists the topics, then estimate sizes from the head."""

    def work(progress):
        files = _mcap_files(path)
        if not files:
            raise FileNotFoundError(f'No .mcap files found in {path}')
        counted, first = (stats, start_ns) if stats else walk_stats(files, progress)
        busy = [name for name, info in counted.items() if info.count]
        means = head_sizes(files[0], busy, progress)
        sized = {
            name: replace(info, bytes=round(means[name] * info.count) if name in means else None)
            for name, info in counted.items()
        }
        return sized, first

    return work


class RewriteWindow(QWidget):
    """Choose a bag, topics, JPEG qualities, thinning, a time range, and an output."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle('rosbag2_dataset rewrite')
        self.stats = None
        self.start_ns = None
        self.path = None
        self.worker = None
        self.build()

    def build(self):
        self.bag_path = QLineEdit()
        self.bag_path.setReadOnly(True)
        self.written = QLabel('')
        open_dir = QPushButton('Open bag folder')
        open_dir.clicked.connect(
            lambda: self.load(QFileDialog.getExistingDirectory(self, 'Bag folder'))
        )
        open_file = QPushButton('Open .mcap')
        open_file.clicked.connect(
            lambda: self.load(QFileDialog.getOpenFileName(self, 'MCAP file', '', '*.mcap')[0])
        )
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(TOPIC, QHeaderView.Stretch)
        self.table.itemChanged.connect(self.show_total)
        keep_all = QPushButton('Keep all')
        keep_all.clicked.connect(lambda: self.keep_all(True))
        keep_none = QPushButton('Keep none')
        keep_none.clicked.connect(lambda: self.keep_all(False))
        self.reference = QComboBox()
        self.reference.setMinimumContentsLength(30)
        self.tolerance = QSpinBox()
        self.tolerance.setRange(1, 60_000)
        self.tolerance.setValue(50)
        self.tolerance.setSuffix(' ms')
        self.ranged = QCheckBox('Time range')
        self.start_s = QDoubleSpinBox()
        self.end_s = QDoubleSpinBox()
        for box in (self.start_s, self.end_s):
            box.setRange(0, 1e7)
            box.setDecimals(1)
            box.setSuffix(' s')
        self.output = QLineEdit()
        self.output.setPlaceholderText('output bag directory; it must not exist')
        browse = QPushButton('Browse')
        browse.clicked.connect(self.browse_output)
        self.total = QLabel('')
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(True)
        self.status = QLabel('Open a bag')
        self.status.setWordWrap(True)
        buttons = {}
        for title, slot in (
            ('Show spec', self.show_spec),
            ('Save spec', self.save_spec),
            ('Run', self.run),
            ('Cancel', self.cancel),
        ):
            buttons[title] = QPushButton(title)
            buttons[title].clicked.connect(slot)
        self.run_button = buttons['Run']
        layout = QVBoxLayout(self)
        layout.addLayout(
            _row(QLabel('Bag'), self.bag_path, self.written, open_dir, open_file)
        )
        layout.addLayout(_row(keep_all, keep_none, None))
        layout.addWidget(self.table)
        layout.addWidget(self.total)
        layout.addLayout(
            _row(QLabel('Thin checked topics to the nearest message of'), self.reference,
                 QLabel('within'), self.tolerance, None)
        )
        layout.addLayout(
            _row(self.ranged, QLabel('from'), self.start_s, QLabel('to'), self.end_s,
                 QLabel('after the bag start'), None)
        )
        layout.addLayout(_row(QLabel('Output'), self.output, browse))
        layout.addLayout(_row(*buttons.values(), None))
        layout.addWidget(self.progress)
        layout.addWidget(self.status)
        self.resize(1100, 700)

    def load(self, path):
        if not path or self.busy():
            return
        self.path = _bag_dir(Path(path))
        self.bag_path.setText(str(self.path))
        self.written.setText(_written(self.path))
        stats, start_ns = _listed(self.path)
        self.stats, self.start_ns = stats or None, start_ns
        self.fill(stats)
        if stats:
            text = f'{len(stats)} topics; estimating sizes from the start of the first file'
        else:
            text = 'No metadata.yaml or MCAP summary: counting messages from the record headers'
        self.status.setText(f'{self.path}: {text}')
        self.start(_estimate(self.path, stats, start_ns), self.estimated)

    def estimated(self, result, error):
        if error:
            self.status.setText(f'Reading topics failed: {error}')
            QMessageBox.warning(self, 'Reading topics failed', str(error))
            return
        self.stats, self.start_ns = result
        self.fill(self.stats)
        self.status.setText(f'{len(self.stats)} topics in {self.path}')

    def fill(self, stats):
        previous = {choice.name: choice for choice in self.choices()}
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for name in sorted(stats):
            self.add_row(name, stats[name], previous.get(name))
        self.table.blockSignals(False)
        reference = self.reference.currentText()
        self.reference.clear()
        self.reference.addItems([''] + sorted(stats))
        self.reference.setCurrentText(reference)
        self.show_total()

    def add_row(self, name, stats, choice):
        row = self.table.rowCount()
        self.table.insertRow(row)
        texts = {
            TOPIC: name,
            TYPE: stats.type,
            COUNT: str(stats.count),
            HZ: '-' if stats.hz is None else f'{stats.hz:.1f}',
            SIZE: _size(stats.bytes),
        }
        for column, text in texts.items():
            item = QTableWidgetItem(text)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            item.setData(Qt.UserRole, stats.bytes)
            self.table.setItem(row, column, item)
        for column, checked in ((KEEP, choice.keep if choice else True),
                                (THIN, choice.thin if choice else False)):
            item = QTableWidgetItem()
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            self.table.setItem(row, column, item)
        if stats.type == IMAGE:
            box = QSpinBox()
            box.setRange(0, 100)
            box.setSpecialValueText('raw')
            box.setToolTip('JPEG quality; raw copies the image unchanged')
            box.setValue(DEFAULT_QUALITY if choice is None else choice.quality or 0)
            self.table.setCellWidget(row, QUALITY, box)

    def choices(self):
        out = []
        for row in range(self.table.rowCount()):
            box = self.table.cellWidget(row, QUALITY)
            out.append(
                Choice(
                    self.table.item(row, TOPIC).text(),
                    self.table.item(row, KEEP).checkState() == Qt.Checked,
                    box.value() or None if box is not None else None,
                    self.table.item(row, THIN).checkState() == Qt.Checked,
                )
            )
        return out

    def keep_all(self, keep):
        self.table.blockSignals(True)
        for row in range(self.table.rowCount()):
            self.table.item(row, KEEP).setCheckState(Qt.Checked if keep else Qt.Unchecked)
        self.table.blockSignals(False)
        self.show_total()

    def show_total(self, *_):
        sizes = [
            self.table.item(row, SIZE).data(Qt.UserRole)
            for row in range(self.table.rowCount())
            if self.table.item(row, KEEP).checkState() == Qt.Checked
        ]
        known = sum(size for size in sizes if size is not None)
        self.total.setText(f'Kept: {len(sizes)} topics, {_size(known)} before JPEG and thinning')

    def spec(self):
        return build_spec(self.choices(), self.reference.currentText(), self.tolerance.value())

    def window_ns(self):
        if not self.ranged.isChecked() or self.start_ns is None:
            return None, None
        start = self.start_ns
        return (start + round(self.start_s.value() * 1e9), start + round(self.end_s.value() * 1e9))

    def command(self):
        start_ns, end_ns = self.window_ns()
        words = ['python -m rosbag2_dataset.rewrite spec.yaml', str(self.path), self.output.text()]
        if start_ns is not None:
            words += [f'--start-ns {start_ns}', f'--end-ns {end_ns}']
        return ' '.join(words)

    def show_spec(self):
        dialog = QDialog(self)
        dialog.setWindowTitle('Rewrite spec')
        text = QPlainTextEdit(yaml.safe_dump(self.spec(), sort_keys=False))
        text.setReadOnly(True)
        command = QLabel(self.command())
        command.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout = QVBoxLayout(dialog)
        layout.addWidget(text)
        layout.addWidget(command)
        dialog.resize(700, 500)
        dialog.show()
        return dialog

    def save_spec(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Save spec', 'rewrite.yaml', '*.yaml')
        if path:
            Path(path).write_text(yaml.safe_dump(self.spec(), sort_keys=False))
            self.status.setText(f'Saved {path}; run it with: {self.command()}')

    def browse_output(self):
        parent = QFileDialog.getExistingDirectory(self, 'Folder for the output bag')
        if parent and self.path is not None:
            self.output.setText(str(Path(parent) / f'{self.path.stem}_rewrite'))

    def run(self):
        out = Path(self.output.text()).expanduser()
        if self.stats is None or self.busy():
            self.status.setText('Wait until the topics are read')
            return
        if not self.output.text() or out.exists():
            self.status.setText('Choose an output directory that does not exist')
            return
        spec = self.spec()
        start_ns, end_ns = self.window_ns()
        bag = self.path

        def work(progress):
            try:
                return rewrite(spec, bag, out, start_ns=start_ns, end_ns=end_ns, progress=progress)
            except Cancelled:
                if out.exists():
                    shutil.rmtree(out)
                raise

        self.status.setText(f'Writing {out}')
        self.start(work, lambda result, error: self.rewritten(out, result, error))

    def rewritten(self, out, counts, error):
        if error:
            self.status.setText(f'Rewrite failed: {error}')
            return
        messages = sum(row.output_count for row in counts)
        written = sum(row.output_bytes for row in counts)
        self.progress.setValue(1000)
        self.progress.setFormat('done')
        self.status.setText(
            f'Wrote {len(counts)} topics, {messages} messages, {_size(written)} to {out}'
        )

    def start(self, work, done):
        self.worker = Worker(work)
        self.worker.progressed.connect(self.show_progress)
        self.worker.completed.connect(done)
        self.progress.setValue(0)
        self.run_button.setEnabled(False)
        self.worker.finished.connect(lambda: self.run_button.setEnabled(True))
        self.worker.start()

    def show_progress(self, stage, done, total):
        self.progress.setValue(round(1000 * done / total) if total else 0)
        self.progress.setFormat(f'{stage}: {done / 1e9:.1f} / {total / 1e9:.1f} GB')

    def busy(self):
        return self.worker is not None and self.worker.isRunning()

    def cancel(self):
        if self.busy():
            self.worker.cancelled = True

    def closeEvent(self, event):  # noqa: N802
        self.cancel()
        if self.worker is not None:
            self.worker.wait()
        super().closeEvent(event)


def _row(*widgets):
    """Lay out widgets in a row; None adds a stretch."""
    row = QHBoxLayout()
    for widget in widgets:
        if widget is None:
            row.addStretch()
        else:
            row.addWidget(widget)
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(prog='rosbag2_dataset', description=__doc__.split('\n')[1])
    parser.add_argument('bag', nargs='?', help='bag directory or .mcap file to open')
    args = parser.parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = RewriteWindow()
    window.show()
    window.load(args.bag)
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
