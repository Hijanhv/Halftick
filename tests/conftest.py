"""Shared fixtures. Tests never use the network or an API key: all market data
comes from hand-built event lists or the synthetic generator."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from halftick.config import Settings, load_config
from halftick.data.pipeline import book_from_events
from halftick.features.compute import compute_features
from halftick.schema import ADD, ASK, BID, CANCEL, TRADE  # noqa: F401  (re-exported for tests)
from halftick.targets.compute import compute_targets

ROOT = Path(__file__).resolve().parents[1]
NS = 1_000_000_000


def _test_config(tmp_path: Path) -> Settings:
    out = [
        f"paths.raw={tmp_path}/raw",
        f"paths.processed={tmp_path}/processed",
        f"paths.calendar={tmp_path}/calendar",
        f"paths.reports={tmp_path}/reports",
        f"paths.figures={tmp_path}/reports/figures",
        f"paths.tables={tmp_path}/reports/tables",
        f"paths.trade_logs={tmp_path}/reports/trade_logs",
        f"paths.data_quality={tmp_path}/reports/data_quality",
        f"paths.readme={tmp_path}/README.md",
        f"databento.spend_ledger={tmp_path}/raw/spend_ledger.json",
        "synthetic.generate_from='08:25'",
        "synthetic.generate_to='08:45'",
    ]
    return load_config(ROOT / "config.yaml", out)


@pytest.fixture
def cfg(tmp_path: Path) -> Settings:
    """Repo config with every output under tmp_path and short synthetic days."""
    return _test_config(tmp_path)


@pytest.fixture(scope="session")
def base_cfg(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    """Session-wide config for property-based tests that must not reuse function fixtures."""
    return _test_config(tmp_path_factory.mktemp("base"))


Event = tuple[int, int, int, int, int, int]  # (ts_ns, action, side, price, size, queue_pos)


def events_frame(events: Sequence[Event]) -> pl.DataFrame:
    arr = np.array(events, dtype=np.int64)
    n = len(events)
    return pl.DataFrame(
        {
            "ts_event": arr[:, 0],
            "sequence": np.arange(1, n + 1, dtype=np.int64),
            "instrument_id": np.ones(n, dtype=np.int32),
            "action": arr[:, 1].astype(np.int8),
            "side": arr[:, 2].astype(np.int8),
            "price": arr[:, 3],
            "size": arr[:, 4],
            "queue_pos": arr[:, 5],
        }
    )


def book_frame(cfg: Settings, events: Sequence[Event]) -> pl.DataFrame:
    frame, flags = book_from_events(cfg, events_frame(events))
    return frame.with_columns(pl.Series("book_flags", flags))


def full_frame(cfg: Settings, events: Sequence[Event], open_ns: int = 0) -> pl.DataFrame:
    book = book_frame(cfg, events)
    feats = compute_features(book, cfg.features, cfg.book.depth_levels, open_ns)
    return book.hstack(feats).hstack(compute_targets(book, cfg.targets.horizons_s))
