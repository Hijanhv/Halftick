"""Per-day pipeline: raw events -> clean events -> book frame -> features + targets.

The output of one day is data/processed/<instrument>/<date>/features.parquet:
book state, session flags, every feature and every target, one row per event.
Downstream steps (models, simulator, diagnostics) only read these files, so
they work the same whether the day came from Databento or the generator.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from halftick.book.l2book import build_book_frame
from halftick.config import Settings
from halftick.data import quality
from halftick.data.roll import detect_roll_days, roll_dates
from halftick.data.sessions import load_releases, release_window_mask, session_bounds_ns
from halftick.features.compute import compute_features
from halftick.log import get_logger
from halftick.targets.compute import compute_targets

log = get_logger(__name__)


def day_dir(cfg: Settings, instrument: str, date: dt.date) -> Path:
    return cfg.paths.processed / instrument / date.isoformat()


def available_days(
    cfg: Settings, instrument: str, artifact: str = "events.parquet"
) -> list[dt.date]:
    root = cfg.paths.processed / instrument
    if not root.exists():
        return []
    return sorted(dt.date.fromisoformat(p.parent.name) for p in root.glob(f"*/{artifact}"))


def roll_table(cfg: Settings, instrument: str) -> pl.DataFrame | None:
    path = cfg.paths.processed / instrument / "daily_volume.parquet"
    if not path.exists():
        return None
    return detect_roll_days(
        pl.read_parquet(path), cfg.roll.min_front_share, cfg.roll.roll_buffer_days
    )


def analysis_days(cfg: Settings, instrument: str) -> list[dt.date]:
    """Days with features, excluding roll days detected from volume."""
    days = available_days(cfg, instrument, "features.parquet")
    table = roll_table(cfg, instrument)
    excluded = roll_dates(table) if table is not None else set()
    return [d for d in days if d not in excluded]


def book_from_events(cfg: Settings, events: pl.DataFrame) -> tuple[pl.DataFrame, np.ndarray]:
    levels = cfg.book.depth_levels
    bid_px, ask_px, bid_d, ask_d, flags = build_book_frame(
        events["action"].to_numpy().astype(np.int8),
        events["side"].to_numpy().astype(np.int8),
        events["price"].to_numpy().astype(np.int64),
        events["size"].to_numpy().astype(np.int64),
        levels,
        cfg.book.price_window_ticks,
    )
    cols: dict[str, np.ndarray] = {"bid_px": bid_px, "ask_px": ask_px}
    for i in range(levels):
        cols[f"bid_sz_{i}"] = bid_d[:, i]
    for i in range(levels):
        cols[f"ask_sz_{i}"] = ask_d[:, i]
    return events.hstack(pl.DataFrame(cols)), flags


def build_day(cfg: Settings, instrument: str, date: dt.date) -> Path:
    """Run the full per-day pipeline and write features.parquet plus a quality report."""
    t_start = time.perf_counter()
    out_dir = day_dir(cfg, instrument, date)
    meta = (
        json.loads((out_dir / "meta.json").read_text()) if (out_dir / "meta.json").exists() else {}
    )
    source = meta.get("source", cfg.data_source)
    report = quality.QualityReport(instrument=instrument, date=date.isoformat(), source=source)

    if source == "synthetic":
        events = quality.clean_events(pl.read_parquet(out_dir / "events.parquet"), report)
        frame, flags = book_from_events(cfg, events)
        quality.check_book_flags(flags, report)
    else:
        from halftick.data.databento_io import make_request, mbp_to_book_frame, read_dbn

        req = make_request(cfg, instrument, "mbp-10", date)
        path = req.path(cfg.paths.raw)
        if not path.exists():
            path = make_request(cfg, instrument, "mbp-1", date).path(cfg.paths.raw)
        raw = mbp_to_book_frame(
            read_dbn(path), cfg.instrument(instrument).tick_size, cfg.book.depth_levels
        )
        frame = quality.clean_events(raw, report)
        bp, ap = frame["bid_px"].to_numpy(), frame["ask_px"].to_numpy()
        flags = np.where(bp > ap, 4, np.where(bp == ap, 8, 0)).astype(np.int32)
        quality.check_book_flags(flags, report)

    ts = frame["ts_event"].to_numpy()
    open_ns, close_ns = session_bounds_ns(cfg, date)
    in_session = (ts >= open_ns) & (ts < close_ns)
    releases = load_releases(cfg)
    rel_ns = releases.filter(pl.col("date") == date.isoformat())["ts_ns"].to_numpy()
    in_release = release_window_mask(ts, rel_ns, cfg.session.release_window_minutes)
    quality.check_session_gaps(ts, in_session, report, cfg.quality.max_gap_seconds)
    quality.enforce(report)
    report.write(cfg.paths.data_quality)

    t_feat = time.perf_counter()
    feats = compute_features(frame, cfg.features, cfg.book.depth_levels, open_ns)
    targets = compute_targets(frame, cfg.targets.horizons_s)
    feat_seconds = time.perf_counter() - t_feat

    keep = [
        "ts_event",
        "sequence",
        "instrument_id",
        "action",
        "side",
        "price",
        "size",
        "queue_pos",
        "bid_px",
        "ask_px",
    ]
    keep += [f"bid_sz_{i}" for i in range(cfg.book.depth_levels)]
    keep += [f"ask_sz_{i}" for i in range(cfg.book.depth_levels)]
    if "signal_x" in frame.columns:
        keep.append("signal_x")
    out = (
        frame.select(keep)
        .hstack(feats)
        .hstack(targets)
        .with_columns(
            pl.Series("book_flags", flags),
            pl.Series("in_session", in_session),
            pl.Series("in_release_window", in_release),
        )
    )
    path = out_dir / "features.parquet"
    out.write_parquet(path)
    log.info(
        "day_built",
        instrument=instrument,
        date=date.isoformat(),
        rows=out.height,
        session_rows=int(in_session.sum()),
        feature_seconds=round(feat_seconds, 2),
        total_seconds=round(time.perf_counter() - t_start, 2),
    )
    return path


def load_features(
    cfg: Settings, instrument: str, date: dt.date, columns: list[str] | None = None
) -> pl.DataFrame:
    return pl.read_parquet(day_dir(cfg, instrument, date) / "features.parquet", columns=columns)
