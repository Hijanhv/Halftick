"""Event-driven replay engine.

The engine walks a MarketDataSource row by row and calls subscribers through
the same two hooks a live feed handler would use: on_trade and
on_book_update. In --realtime mode it sleeps so that one second of market
time takes 1/speed seconds of wall time, which is what the Grafana demo uses.

The research pipeline does not go through this loop (it runs numba kernels over
whole days, which is about 1000x faster). Tests check that the streaming
features computed here match the batch features exactly, so the two paths
cannot drift apart silently.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from halftick.book.l2book import FLAG_CROSSED, FLAG_LOCKED
from halftick.log import get_logger
from halftick.replay.sources import MarketDataSource
from halftick.schema import NO_ASK, NO_BID, TRADE

log = get_logger(__name__)
NS = 1_000_000_000


@dataclass(frozen=True, slots=True)
class BookUpdate:
    ts_event: int
    sequence: int
    action: int
    side: int
    price: int
    size: int
    bid_px: int
    ask_px: int
    bid_sz: int
    ask_sz: int
    flags: int

    @property
    def valid(self) -> bool:
        return self.bid_px != NO_BID and self.ask_px != NO_ASK

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid_px + self.ask_px)


@dataclass(frozen=True, slots=True)
class Trade:
    ts_event: int
    side: int  # aggressor side
    price: int
    size: int


class Subscriber(Protocol):
    def on_trade(self, trade: Trade) -> None: ...

    def on_book_update(self, update: BookUpdate) -> None: ...


class EngineObserver(Protocol):
    """Receives engine-level telemetry (implemented by the Prometheus exporter)."""

    def event_processed(self, update: BookUpdate, wall_lag_s: float) -> None: ...

    def data_quality_error(self, check: str) -> None: ...


@dataclass
class ReplayStats:
    events: int = 0
    trades: int = 0
    crossed: int = 0
    locked: int = 0
    wall_seconds: float = 0.0


class ReplayEngine:
    def __init__(
        self,
        source: MarketDataSource,
        subscribers: list[Subscriber] | None = None,
        realtime: bool = False,
        speed: float = 1.0,
        observer: EngineObserver | None = None,
        start_ns: int | None = None,
        end_ns: int | None = None,
    ) -> None:
        if speed <= 0:
            raise ValueError("speed must be positive")
        self.source = source
        self.subscribers: list[Subscriber] = list(subscribers or [])
        self.realtime = realtime
        self.speed = speed
        self.observer = observer
        self.start_ns = start_ns
        self.end_ns = end_ns
        self.stats = ReplayStats()

    def subscribe(self, sub: Subscriber) -> None:
        self.subscribers.append(sub)

    def run(self) -> ReplayStats:
        wall0 = time.perf_counter()
        market0: int | None = None
        flags_present = None
        for batch in self.source.batches():
            if flags_present is None:
                flags_present = "book_flags" in batch.columns
            cols = {
                c: batch[c].to_numpy()
                for c in (
                    "ts_event",
                    "sequence",
                    "action",
                    "side",
                    "price",
                    "size",
                    "bid_px",
                    "ask_px",
                    "bid_sz_0",
                    "ask_sz_0",
                )
            }
            flags = (
                batch["book_flags"].to_numpy()
                if flags_present
                else np.zeros(batch.height, dtype=np.int32)
            )
            for i in range(batch.height):
                ts = int(cols["ts_event"][i])
                if self.start_ns is not None and ts < self.start_ns:
                    continue
                if self.end_ns is not None and ts >= self.end_ns:
                    self.stats.wall_seconds = time.perf_counter() - wall0
                    return self.stats
                if market0 is None:
                    market0 = ts
                    wall0 = time.perf_counter()
                lag = 0.0
                if self.realtime:
                    due = wall0 + (ts - market0) / NS / self.speed
                    now = time.perf_counter()
                    if due > now:
                        time.sleep(due - now)
                    lag = max(0.0, time.perf_counter() - due)
                upd = BookUpdate(
                    ts_event=ts,
                    sequence=int(cols["sequence"][i]),
                    action=int(cols["action"][i]),
                    side=int(cols["side"][i]),
                    price=int(cols["price"][i]),
                    size=int(cols["size"][i]),
                    bid_px=int(cols["bid_px"][i]),
                    ask_px=int(cols["ask_px"][i]),
                    bid_sz=int(cols["bid_sz_0"][i]),
                    ask_sz=int(cols["ask_sz_0"][i]),
                    flags=int(flags[i]),
                )
                self._dispatch(upd, lag)
        self.stats.wall_seconds = time.perf_counter() - wall0
        return self.stats

    def _dispatch(self, upd: BookUpdate, lag: float) -> None:
        self.stats.events += 1
        if upd.flags & FLAG_CROSSED:
            self.stats.crossed += 1
            if self.observer:
                self.observer.data_quality_error("crossed_book")
        if upd.flags & FLAG_LOCKED:
            self.stats.locked += 1
            if self.observer:
                self.observer.data_quality_error("locked_book")
        if upd.action == TRADE:
            self.stats.trades += 1
            trade = Trade(upd.ts_event, upd.side, upd.price, upd.size)
            for s in self.subscribers:
                s.on_trade(trade)
        for s in self.subscribers:
            s.on_book_update(upd)
        if self.observer:
            self.observer.event_processed(upd, lag)
