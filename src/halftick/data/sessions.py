"""Session windows and the macro-release calendar.

Timestamps arrive in UTC nanoseconds. The U.S. day session is defined in New
York local time, and New York switches between UTC-4 and UTC-5 twice a year, so
we never hard-code an offset: zoneinfo converts each date's local open and
close to UTC, which is correct on both sides of a DST change.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import numpy.typing as npt
import polars as pl

from halftick.config import Settings

NS = 1_000_000_000


def local_to_utc_ns(date: dt.date, hhmm: str, tz_name: str) -> int:
    h, m = (int(x) for x in hhmm.split(":"))
    local = dt.datetime(date.year, date.month, date.day, h, m, tzinfo=ZoneInfo(tz_name))
    return int(local.timestamp()) * NS


def session_bounds_ns(cfg: Settings, date: dt.date) -> tuple[int, int]:
    s = cfg.session
    return local_to_utc_ns(date, s.start, s.timezone), local_to_utc_ns(date, s.end, s.timezone)


def releases_path(cfg: Settings) -> Path:
    name = "releases_synthetic.csv" if cfg.data_source == "synthetic" else "releases.csv"
    return cfg.paths.calendar / name


def load_releases(cfg: Settings, path: Path | None = None) -> pl.DataFrame:
    """Return releases as (date, time_et, event, ts_ns) with ts_ns in UTC nanoseconds."""
    path = path or releases_path(cfg)
    if not path.exists():
        return pl.DataFrame(
            schema={"date": pl.String, "time_et": pl.String, "event": pl.String, "ts_ns": pl.Int64}
        )
    df = pl.read_csv(
        path, comment_prefix="#", schema_overrides={"date": pl.String, "time_et": pl.String}
    )
    ts = [
        local_to_utc_ns(dt.date.fromisoformat(d), t, cfg.session.timezone)
        for d, t in zip(df["date"], df["time_et"], strict=True)
    ]
    return df.with_columns(pl.Series("ts_ns", ts, dtype=pl.Int64))


def release_window_mask(
    ts: npt.NDArray[np.int64], release_ns: npt.NDArray[np.int64], window_minutes: int
) -> npt.NDArray[np.bool_]:
    """True where ts is within +/- window of any release time."""
    mask = np.zeros(ts.shape[0], dtype=bool)
    w = window_minutes * 60 * NS
    for r in release_ns:
        lo, hi = np.searchsorted(ts, [r - w, r + w], side="left")
        mask[lo:hi] = True
    return mask


def time_of_day_bucket(
    minutes: npt.NDArray[np.float64], buckets: dict[str, tuple[int, int]]
) -> npt.NDArray[np.object_]:
    out = np.full(minutes.shape[0], "outside", dtype=object)
    for name, (lo, hi) in buckets.items():
        out[(minutes >= lo) & (minutes < hi)] = name
    return out
