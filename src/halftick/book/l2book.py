"""Price-level (L2) order book.

The book keeps one dense array of sizes per side, indexed by price in ticks
relative to a base price. Large-tick futures trade inside a narrow band each
day, so a fixed window of a few thousand ticks is plenty, and dense arrays make
every update O(1) except when the best level empties (then we scan outward to
the next non-empty level, which is usually one step away).

The update logic is a numba function so the same code runs in the batch
pipeline (millions of events per day) and inside the event-driven ReplayEngine.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import njit

from halftick.schema import ADD, ASK, BID, CANCEL, CLEAR, MODIFY, NO_ASK, NO_BID, TRADE

# Bit flags returned by apply_event.
FLAG_OUT_OF_RANGE = 1
FLAG_NEGATIVE_SIZE = 2
FLAG_CROSSED = 4
FLAG_LOCKED = 8
FLAG_BAD_SIZE = 16

# meta layout
_BASE, _BEST_BID, _BEST_ASK = 0, 1, 2

I64 = npt.NDArray[np.int64]


@njit(cache=True)
def _rescan_bid(bids: I64, meta: I64, start_idx: int) -> None:
    i = start_idx
    while i >= 0:
        if bids[i] > 0:
            meta[_BEST_BID] = meta[_BASE] + i
            return
        i -= 1
    meta[_BEST_BID] = NO_BID


@njit(cache=True)
def _rescan_ask(asks: I64, meta: I64, start_idx: int) -> None:
    i = start_idx
    n = asks.shape[0]
    while i < n:
        if asks[i] > 0:
            meta[_BEST_ASK] = meta[_BASE] + i
            return
        i += 1
    meta[_BEST_ASK] = NO_ASK


@njit(cache=True)
def _remove(arr: I64, meta: I64, idx: int, size: int, is_bid: bool) -> int:
    flags = 0
    arr[idx] -= size
    if arr[idx] < 0:
        arr[idx] = 0
        flags |= FLAG_NEGATIVE_SIZE
    if arr[idx] == 0:
        price = meta[_BASE] + idx
        if is_bid and price == meta[_BEST_BID]:
            _rescan_bid(arr, meta, idx - 1)
        elif (not is_bid) and price == meta[_BEST_ASK]:
            _rescan_ask(arr, meta, idx + 1)
    return flags


@njit(cache=True)
def apply_event(
    bids: I64, asks: I64, meta: I64, action: int, side: int, price: int, size: int
) -> int:
    """Apply one delta to the book in place. Returns bit flags describing problems."""
    if action == CLEAR:
        bids[:] = 0
        asks[:] = 0
        meta[_BEST_BID] = NO_BID
        meta[_BEST_ASK] = NO_ASK
        return 0
    idx = price - meta[_BASE]
    if idx < 0 or idx >= bids.shape[0]:
        return FLAG_OUT_OF_RANGE
    flags = 0
    if size < 0 or (size == 0 and action != MODIFY):
        return FLAG_BAD_SIZE

    if action == ADD:
        if side == BID:
            bids[idx] += size
            if price > meta[_BEST_BID]:
                meta[_BEST_BID] = price
        else:
            asks[idx] += size
            if price < meta[_BEST_ASK]:
                meta[_BEST_ASK] = price
    elif action == CANCEL:
        if side == BID:
            flags |= _remove(bids, meta, idx, size, True)
        else:
            flags |= _remove(asks, meta, idx, size, False)
    elif action == MODIFY:
        if side == BID:
            bids[idx] = size
            if size > 0 and price > meta[_BEST_BID]:
                meta[_BEST_BID] = price
            elif size == 0 and price == meta[_BEST_BID]:
                _rescan_bid(bids, meta, idx - 1)
        else:
            asks[idx] = size
            if size > 0 and price < meta[_BEST_ASK]:
                meta[_BEST_ASK] = price
            elif size == 0 and price == meta[_BEST_ASK]:
                _rescan_ask(asks, meta, idx + 1)
    elif action == TRADE:
        # A buy aggressor (side BID) consumes resting asks, and vice versa.
        if side == BID:
            flags |= _remove(asks, meta, idx, size, False)
        else:
            flags |= _remove(bids, meta, idx, size, True)

    bb = meta[_BEST_BID]
    ba = meta[_BEST_ASK]
    if bb != NO_BID and ba != NO_ASK:
        if bb > ba:
            flags |= FLAG_CROSSED
        elif bb == ba:
            flags |= FLAG_LOCKED
    return flags


@njit(cache=True)
def _fill_depth(bids: I64, asks: I64, meta: I64, out_bid: I64, out_ask: I64) -> None:
    levels = out_bid.shape[0]
    n = bids.shape[0]
    bb = meta[_BEST_BID]
    ba = meta[_BEST_ASK]
    for i in range(levels):
        if bb != NO_BID:
            j = bb - meta[_BASE] - i
            out_bid[i] = bids[j] if j >= 0 else 0
        else:
            out_bid[i] = 0
        if ba != NO_ASK:
            j = ba - meta[_BASE] + i
            out_ask[i] = asks[j] if j < n else 0
        else:
            out_ask[i] = 0


@njit(cache=True)
def build_book_frame(
    actions: npt.NDArray[np.int8],
    sides: npt.NDArray[np.int8],
    prices: I64,
    sizes: I64,
    levels: int,
    window: int,
) -> tuple[I64, I64, I64, I64, npt.NDArray[np.int32]]:
    """Replay a whole day of deltas and return the book after every event.

    Returns (bid_px, ask_px, bid_depth[n, levels], ask_depth[n, levels], flags).
    Depth is on a dense price grid: column i is the size i ticks away from the touch.
    """
    n = actions.shape[0]
    bids = np.zeros(window, dtype=np.int64)
    asks = np.zeros(window, dtype=np.int64)
    meta = np.empty(3, dtype=np.int64)
    meta[_BASE] = (prices[0] if n > 0 else 0) - window // 2
    meta[_BEST_BID] = NO_BID
    meta[_BEST_ASK] = NO_ASK
    bid_px = np.empty(n, dtype=np.int64)
    ask_px = np.empty(n, dtype=np.int64)
    bid_depth = np.zeros((n, levels), dtype=np.int64)
    ask_depth = np.zeros((n, levels), dtype=np.int64)
    flags = np.zeros(n, dtype=np.int32)
    for k in range(n):
        flags[k] = apply_event(
            bids, asks, meta, int(actions[k]), int(sides[k]), int(prices[k]), int(sizes[k])
        )
        bid_px[k] = meta[_BEST_BID]
        ask_px[k] = meta[_BEST_ASK]
        _fill_depth(bids, asks, meta, bid_depth[k], ask_depth[k])
    return bid_px, ask_px, bid_depth, ask_depth, flags


class OrderBook:
    """Object wrapper around the numba book for tests and the event-driven engine."""

    def __init__(self, base_price: int, window: int = 16384) -> None:
        self.window = window
        self.bids: I64 = np.zeros(window, dtype=np.int64)
        self.asks: I64 = np.zeros(window, dtype=np.int64)
        self.meta: I64 = np.array([base_price - window // 2, NO_BID, NO_ASK], dtype=np.int64)

    def apply(self, action: int, side: int, price: int, size: int) -> int:
        return int(apply_event(self.bids, self.asks, self.meta, action, side, price, size))

    @property
    def best_bid(self) -> int | None:
        v = int(self.meta[_BEST_BID])
        return None if v == NO_BID else v

    @property
    def best_ask(self) -> int | None:
        v = int(self.meta[_BEST_ASK])
        return None if v == NO_ASK else v

    def size_at(self, side: int, price: int) -> int:
        idx = price - int(self.meta[_BASE])
        if idx < 0 or idx >= self.window:
            return 0
        return int(self.bids[idx] if side == BID else self.asks[idx])

    def depth(self, levels: int) -> tuple[I64, I64]:
        out_bid = np.zeros(levels, dtype=np.int64)
        out_ask = np.zeros(levels, dtype=np.int64)
        _fill_depth(self.bids, self.asks, self.meta, out_bid, out_ask)
        return out_bid, out_ask


__all__ = [
    "ASK",
    "BID",
    "FLAG_BAD_SIZE",
    "FLAG_CROSSED",
    "FLAG_LOCKED",
    "FLAG_NEGATIVE_SIZE",
    "FLAG_OUT_OF_RANGE",
    "OrderBook",
    "apply_event",
    "build_book_frame",
]
