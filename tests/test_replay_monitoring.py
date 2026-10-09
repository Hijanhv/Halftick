"""ReplayEngine dispatch order and pacing, metrics export, trade-log playback, CLI."""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from prometheus_client import generate_latest
from typer.testing import CliRunner

from halftick.cli import app
from halftick.config import Settings
from halftick.monitoring.metrics import FeatureMonitor, ReplayMetrics, TradeLogPlayback
from halftick.replay.engine import BookUpdate, ReplayEngine, Trade
from halftick.replay.sources import MarketDataSource, ParquetSource, SyntheticSource
from halftick.schema import ADD, ASK, BID, TRADE
from tests.conftest import NS, ROOT, book_frame

EVENTS = [
    (0, ADD, BID, 100, 10, -1),
    (0, ADD, ASK, 101, 10, -1),
    (NS // 10, TRADE, ASK, 100, 4, 0),
    (NS // 5, ADD, BID, 101, 2, -1),  # locks the book: a data-quality event
]


class FrameSource:
    instrument = "ZN"

    def __init__(self, frame: pl.DataFrame) -> None:
        self.frame = frame

    def batches(self):  # type: ignore[no-untyped-def]
        yield from self.frame.iter_slices(2)


class Recorder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def on_trade(self, t: Trade) -> None:
        self.calls.append(f"trade:{t.side}:{t.size}")

    def on_book_update(self, u: BookUpdate) -> None:
        self.calls.append(f"book:{u.sequence}")


def test_engine_dispatches_trade_before_book_update(cfg: Settings) -> None:
    rec = Recorder()
    stats = ReplayEngine(FrameSource(book_frame(cfg, EVENTS)), [rec]).run()
    assert rec.calls == ["book:1", "book:2", f"trade:{ASK}:4", "book:3", "book:4"]
    assert (stats.events, stats.trades, stats.locked) == (4, 1, 1)


def test_sources_satisfy_the_protocol(cfg: Settings, tmp_path: Path) -> None:
    assert isinstance(FrameSource(book_frame(cfg, EVENTS)), MarketDataSource)
    assert isinstance(SyntheticSource(cfg, "ZN", dt.date(2026, 8, 17)), MarketDataSource)
    p = tmp_path / "f.parquet"
    book_frame(cfg, EVENTS).write_parquet(p)
    assert isinstance(ParquetSource(p, "ZN"), MarketDataSource)
    assert sum(b.height for b in ParquetSource(p, "ZN", batch_size=3).batches()) == 4


def test_realtime_mode_paces_by_market_time(cfg: Settings) -> None:
    t0 = time.perf_counter()
    ReplayEngine(FrameSource(book_frame(cfg, EVENTS)), [], realtime=True, speed=1.0).run()
    elapsed = time.perf_counter() - t0
    assert 0.18 <= elapsed < 1.0  # last event is 0.2 s of market time after the first
    with pytest.raises(ValueError):
        ReplayEngine(FrameSource(book_frame(cfg, EVENTS)), speed=0)


def test_metrics_and_trade_log_playback(cfg: Settings, tmp_path: Path) -> None:
    frame = book_frame(cfg, EVENTS)
    log = pl.DataFrame(
        {
            "queue_model": ["proportional"] * 2,
            "latency_ms": [10] * 2,
            "deadline_s": [15] * 2,
            "strategy": ["always_cross", "model_cost"],
            "decision_ts": [0, 0],
            "exec_ts": [NS // 10, NS],
            "filled_passive": [False, True],
            "cost_ticks": [0.5, -0.5],
            "cost_usd": [10.3, -5.3],
        }
    )
    lp = tmp_path / "log.parquet"
    log.write_parquet(lp)
    m = ReplayMetrics("ZN", session_bounds_ns=(0, NS))
    ReplayEngine(
        FrameSource(frame),
        [FeatureMonitor(m), TradeLogPlayback(m, lp, "proportional", 10, 15)],
        observer=m,
    ).run()
    text = generate_latest(m.registry).decode()
    assert 'halftick_events_total{instrument="ZN"} 4.0' in text
    assert 'halftick_data_quality_errors_total{check="locked_book",instrument="ZN"} 1.0' in text
    assert 'halftick_sim_orders_total{instrument="ZN",strategy="model_cost"} 1.0' in text
    # always_cross finished at 0.1 s (replayed); model_cost finishes at 1 s (after the data ends)
    assert 'halftick_sim_avg_cost_ticks{instrument="ZN",strategy="always_cross"} 0.5' in text
    assert 'halftick_sim_completed_total{instrument="ZN",strategy="model_cost"}' not in text
    assert "halftick_feature_latency_seconds_count" in text
    assert 'halftick_in_session{instrument="ZN"} 1.0' in text


def test_cli_synth_and_build(tmp_path: Path) -> None:
    runner = CliRunner()
    sets = [
        "--set",
        f"paths.processed={tmp_path}/p",
        "--set",
        f"paths.calendar={tmp_path}/c",
        "--set",
        f"paths.data_quality={tmp_path}/dq",
        "--set",
        "synthetic.generate_from='08:25'",
        "--set",
        "synthetic.generate_to='08:40'",
    ]
    base = ["--config", str(ROOT / "config.yaml"), *sets]
    r = runner.invoke(app, [*base, "synth", "-i", "ES", "--days", "1"])
    assert r.exit_code == 0, r.output
    r = runner.invoke(app, [*base, "build-features", "-i", "ES"])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "dq" / "SUMMARY.md").exists()
    assert len(list((tmp_path / "p" / "ES").glob("*/features.parquet"))) == 1
    assert np.isfinite(
        pl.read_parquet(next((tmp_path / "p" / "ES").glob("*/features.parquet")))["mid"]
        .drop_nans()
        .mean()
    )


def test_playback_ignores_orders_before_a_mid_day_start(cfg: Settings, tmp_path: Path) -> None:
    frame = book_frame(cfg, EVENTS)
    log = pl.DataFrame(
        {
            "queue_model": ["proportional"] * 3,
            "latency_ms": [10] * 3,
            "deadline_s": [15] * 3,
            "strategy": ["always_cross"] * 3,
            "decision_ts": [0, 0, NS // 5],  # two orders before the replay starts, one after
            "exec_ts": [NS // 20, NS // 5, NS // 5],
            "filled_passive": [False] * 3,
            "cost_ticks": [0.5] * 3,
            "cost_usd": [10.3] * 3,
        }
    )
    lp = tmp_path / "log.parquet"
    log.write_parquet(lp)
    m = ReplayMetrics("ZN")
    pb = TradeLogPlayback(m, lp, "proportional", 10, 15)
    ReplayEngine(FrameSource(frame), [pb], observer=m, start_ns=NS // 10).run()
    text = generate_latest(m.registry).decode()
    assert 'halftick_sim_orders_total{instrument="ZN",strategy="always_cross"} 1.0' in text
    assert 'halftick_sim_completed_total{instrument="ZN",strategy="always_cross"} 1.0' in text
