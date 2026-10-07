"""Incremental versions of the core features, for the event-driven engine.

These are what a live system would run: O(1) work per event, no look-back
over arrays. A test replays a synthetic day through both this class and the
batch kernel and requires identical values, so the research numbers and the
live numbers come from the same definitions.
"""

from __future__ import annotations

import time
from collections import deque

from halftick.replay.engine import BookUpdate, Trade
from halftick.schema import BID


class StreamingFeatures:
    def __init__(self, ofi_events: int = 50, tfi_window_ms: int = 1000) -> None:
        self.ofi_events = ofi_events
        self.tfi_window_ns = tfi_window_ms * 1_000_000
        self._ofi: deque[float] = deque()
        self._ofi_sum = 0.0
        self._trades: deque[tuple[int, int, int]] = deque()  # (ts, buy, sell)
        self._buy = 0
        self._sell = 0
        self._prev: BookUpdate | None = None
        self.imbalance = 0.0
        self.microprice = 0.0
        self.ofi = 0.0
        self.tfi = 0.0
        self.last_compute_seconds = 0.0

    def on_trade(self, trade: Trade) -> None:
        buy = trade.size if trade.side == BID else 0
        sell = trade.size if trade.side != BID else 0
        self._trades.append((trade.ts_event, buy, sell))
        self._buy += buy
        self._sell += sell

    def on_book_update(self, u: BookUpdate) -> None:
        t0 = time.perf_counter()
        # Trade-flow window: drop trades at or before ts - window.
        lo = u.ts_event - self.tfi_window_ns
        while self._trades and self._trades[0][0] <= lo:
            _, b, s = self._trades.popleft()
            self._buy -= b
            self._sell -= s
        tot = self._buy + self._sell
        self.tfi = (self._buy - self._sell) / tot if tot > 0 else 0.0

        e = 0.0
        p = self._prev
        if p is not None and p.valid and u.valid:
            if u.bid_px >= p.bid_px:
                e += u.bid_sz
            if u.bid_px <= p.bid_px:
                e -= p.bid_sz
            if u.ask_px <= p.ask_px:
                e -= u.ask_sz
            if u.ask_px >= p.ask_px:
                e += p.ask_sz
        self._ofi.append(e)
        self._ofi_sum += e
        if len(self._ofi) > self.ofi_events:
            self._ofi_sum -= self._ofi.popleft()
        self.ofi = self._ofi_sum

        if u.valid:
            q = u.bid_sz + u.ask_sz
            self.imbalance = (u.bid_sz - u.ask_sz) / q if q > 0 else 0.0
            self.microprice = (
                (u.bid_px * u.ask_sz + u.ask_px * u.bid_sz) / q - u.mid if q > 0 else 0.0
            )
        self._prev = u
        self.last_compute_seconds = time.perf_counter() - t0
