"""Market data sources for the ReplayEngine.

A source yields the book frame in event order, in batches. The engine does
not care where rows come from, so adding a DatabentoLiveSource later means
writing one more class with the same `batches()` method: subscribe to the
live gateway, convert each record with mbp_to_book_frame, and yield.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, runtime_checkable

import polars as pl

from halftick.config import Settings


@runtime_checkable
class MarketDataSource(Protocol):
    instrument: str

    def batches(self) -> Iterator[pl.DataFrame]:
        """Yield book-frame batches (see halftick.schema) in event order."""
        ...


class SyntheticSource:
    """Generates a synthetic day in memory. Used by tests and the live demo."""

    def __init__(
        self,
        cfg: Settings,
        instrument: str,
        date: dt.date,
        day_index: int = 0,
        start_bid_ticks: int | None = None,
        batch_size: int = 20_000,
    ) -> None:
        self.cfg = cfg
        self.instrument = instrument
        self.date = date
        self.day_index = day_index
        sp = cfg.synthetic.instruments[instrument]
        tick = cfg.instrument(instrument).tick_size
        self.start_bid = (
            start_bid_ticks if start_bid_ticks is not None else round(sp.start_price / tick)
        )
        self.batch_size = batch_size

    def frame(self) -> pl.DataFrame:
        from halftick.data import quality
        from halftick.data.pipeline import book_from_events
        from halftick.data.synthetic import generate_day

        day = generate_day(self.cfg, self.instrument, self.day_index, self.date, self.start_bid)
        report = quality.QualityReport(self.instrument, self.date.isoformat(), "synthetic")
        events = quality.clean_events(day.events, report)
        frame, flags = book_from_events(self.cfg, events)
        return frame.with_columns(pl.Series("book_flags", flags))

    def batches(self) -> Iterator[pl.DataFrame]:
        yield from self.frame().iter_slices(self.batch_size)


class ParquetSource:
    """Replays a processed day (features.parquet) from disk."""

    def __init__(self, path: Path, instrument: str, batch_size: int = 20_000) -> None:
        self.path = path
        self.instrument = instrument
        self.batch_size = batch_size

    def batches(self) -> Iterator[pl.DataFrame]:
        yield from pl.read_parquet(self.path).iter_slices(self.batch_size)


class DatabentoHistoricalSource:
    """Replays a cached Databento mbp-1 / mbp-10 DBN file."""

    def __init__(
        self, cfg: Settings, instrument: str, path: Path, batch_size: int = 20_000
    ) -> None:
        self.cfg = cfg
        self.instrument = instrument
        self.path = path
        self.batch_size = batch_size

    def batches(self) -> Iterator[pl.DataFrame]:
        from halftick.data.databento_io import mbp_to_book_frame, read_dbn

        frame = mbp_to_book_frame(
            read_dbn(self.path),
            self.cfg.instrument(self.instrument).tick_size,
            self.cfg.book.depth_levels,
        )
        yield from frame.iter_slices(self.batch_size)
