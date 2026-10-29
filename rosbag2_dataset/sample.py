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

"""Choose the messages of one sample, from a whole bag or from live messages."""

from bisect import bisect_left, bisect_right
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


def nearest(times, time):
    """Return the first index of the time nearest to time; ties pick the earlier time."""
    if not len(times):
        raise ValueError('Cannot find the nearest message in an empty topic')
    index = int(np.searchsorted(times, time, side='left'))
    if index == len(times):
        index -= 1
    elif index and int(times[index]) - time >= time - int(times[index - 1]):
        index -= 1
    return int(np.searchsorted(times, times[index], side='left'))


@dataclass(frozen=True)
class Window:
    """Messages every every_ns before (past) and after (future) the reference time."""

    every_ns: int
    past: int = 0
    future: int = 0
    tolerance_ns: int = 0

    def __post_init__(self):
        if self.every_ns <= 0 or self.past < 0 or self.future < 0 or self.tolerance_ns < 0:
            raise ValueError('every_ns must be positive; past, future and tolerance_ns '
                             'nonnegative')

    @property
    def offsets(self):
        return [-k * self.every_ns for k in range(self.past, 0, -1)] + [
            k * self.every_ns for k in range(1, self.future + 1)]


class SampleRule:
    """
    Align topics to each reference message; the one rule for bags and live input.

    A row is dropped when a topic is beyond tolerance, unless the topic is optional;
    then its index is -1. Window positions beyond their tolerance are -1.
    """

    def __init__(self, topics, reference=None, tolerance_ns=0, *, optional=(),
                 windows=None):
        self.topics = tuple(topics)
        if not self.topics or len(set(self.topics)) != len(self.topics):
            raise ValueError('topics must contain distinct topic names')
        self.reference = self.topics[0] if reference is None else reference
        if self.reference not in self.topics:
            raise ValueError('reference must be one of topics')
        self.optional = frozenset(optional)
        if not self.optional <= set(self.topics) - {self.reference}:
            raise ValueError('optional must name non-reference topics')
        self.tolerance_ns = {
            name: int(tolerance_ns[name] if isinstance(tolerance_ns, Mapping) else tolerance_ns)
            for name in self.topics if name != self.reference}
        if any(limit < 0 for limit in self.tolerance_ns.values()):
            raise ValueError('tolerance_ns must be nonnegative')
        self.windows = {name: window if isinstance(window, Window) else Window(**window)
                        for name, window in (windows or {}).items()}
        if not set(self.windows) <= set(self.topics):
            raise ValueError('windows must name topics of the rule')

    def horizon_ns(self, name):
        """Return how far past the reference time a message of name can still be chosen."""
        limit = self.tolerance_ns.get(name, 0)
        window = self.windows.get(name)
        if window is not None:
            limit = max(limit, window.future * window.every_ns + window.tolerance_ns)
        return limit

    def history_ns(self, name):
        """Return how far before the reference time a message of name can be chosen."""
        limit = self.tolerance_ns.get(name, 0)
        window = self.windows.get(name)
        if window is not None:
            limit = max(limit, window.past * window.every_ns + window.tolerance_ns)
        return limit

    def select(self, clocks, ref_index):
        """
        Choose indices into the sorted clocks for the reference message at ref_index.

        Returns (indices, times, windows) or None for a dropped row. windows maps a
        topic to an int64 array of indices with -1 beyond tolerance.
        """
        ref_time = int(clocks[self.reference][ref_index])
        indices = []
        times = []
        for name in self.topics:
            if name == self.reference:
                indices.append(ref_index)
                times.append(ref_time)
                continue
            clock = clocks[name]
            time = ref_time
            if len(clock):
                index = nearest(clock, ref_time)
                time = int(clock[index])
                if abs(time - ref_time) <= self.tolerance_ns[name]:
                    indices.append(index)
                    times.append(time)
                    continue
            if name not in self.optional:
                return None
            indices.append(-1)
            times.append(time)
        windows = {}
        for name, window in self.windows.items():
            clock = clocks[name]
            chosen = np.full(len(window.offsets), -1, dtype=np.int64)
            for slot, offset in enumerate(window.offsets):
                if len(clock):
                    index = nearest(clock, ref_time + offset)
                    if abs(int(clock[index]) - ref_time - offset) <= window.tolerance_ns:
                        chosen[slot] = index
            windows[name] = chosen
        return indices, times, windows


class LiveSampler:
    """
    Build samples from messages as they arrive, choosing what a whole bag would.

    add() takes each topic's messages in clock order. A reference message becomes a
    sample once every topic holds a later message than it could still choose, so the
    choice matches SampleRule over the full recording. ready(now_ns, max_wait_ns)
    also releases references older than max_wait_ns with the messages held so far.
    """

    def __init__(self, rule):
        self.rule = rule
        self._times = {name: [] for name in rule.topics}
        self._items = {name: [] for name in rule.topics}
        self._next = 0

    def add(self, name, clock_ns, item, *, log_ns=None):
        """Hold one message; samples carry (log_ns or clock_ns, item)."""
        times = self._times[name]
        position = bisect_right(times, clock_ns)
        if name == self.rule.reference and position < self._next:
            raise ValueError('reference messages must arrive in clock order')
        times.insert(position, int(clock_ns))
        self._items[name].insert(position, (clock_ns if log_ns is None else log_ns, item))

    def _final(self, ref_time, now_ns, max_wait_ns):
        if now_ns is not None and max_wait_ns is not None and now_ns - ref_time > max_wait_ns:
            return True
        for name in self.rule.topics:
            if name == self.rule.reference:
                continue
            times = self._times[name]
            if not times or times[-1] <= ref_time + self.rule.horizon_ns(name):
                return False
        return True

    def ready(self, now_ns=None, max_wait_ns=None):
        """Return the samples that became final, oldest first, and drop unneeded messages."""
        samples = []
        reference = self._times[self.rule.reference]
        while self._next < len(reference) and self._final(reference[self._next], now_ns,
                                                          max_wait_ns):
            clocks = {name: np.asarray(times, dtype=np.int64)
                      for name, times in self._times.items()}
            chosen = self.rule.select(clocks, self._next)
            if chosen is not None:
                samples.append(_sample(self.rule, chosen, self._items))
            self._next += 1
        if samples or self._next:
            self._prune()
        return samples

    def _prune(self):
        reference = self._times[self.rule.reference]
        if not reference:
            return
        pending = reference[min(self._next, len(reference) - 1)]
        for name in self.rule.topics:
            keep_from = pending - self.rule.history_ns(name)
            times = self._times[name]
            # The last time before keep_from stays, with all its messages: it can still
            # be the nearest, and ties choose the first of them.
            before = bisect_right(times, keep_from - 1)
            cut = bisect_left(times, times[before - 1]) if before else 0
            if name == self.rule.reference:
                cut = min(cut, self._next)
                self._next -= cut
            del times[:cut]
            del self._items[name][:cut]


def _sample(rule, chosen, items):
    indices, times, windows = chosen
    sample = {'@time': times[rule.topics.index(rule.reference)]}
    for name, index in zip(rule.topics, indices):
        sample[name] = items[name][index] if index >= 0 else None
    for name, positions in windows.items():
        window = rule.windows[name]
        entries = [items[name][int(i)] if i >= 0 else None for i in positions]
        sample[name + '@past'] = entries[:window.past]
        sample[name + '@future'] = entries[window.past:]
    return sample
