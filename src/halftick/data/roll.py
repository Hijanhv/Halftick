"""Roll-week detection from daily volume by contract.

Quarterly futures roll: for a few days volume migrates from the expiring
contract to the next one, and the front-by-volume book can be thin or jump
between contracts. We find those days from the data (the front contract's
share of volume and whether the front contract changed) instead of
hard-coding calendar dates.
"""

from __future__ import annotations

import datetime as dt

import polars as pl


def detect_roll_days(
    daily_volume: pl.DataFrame, min_front_share: float, buffer_days: int
) -> pl.DataFrame:
    """Return per-day front contract, its volume share and an is_roll flag.

    daily_volume needs columns: date, instrument_id, volume.
    """
    per_day = (
        daily_volume.group_by("date")
        .agg(
            pl.col("volume").sum().alias("total"),
            pl.col("instrument_id").sort_by("volume").last().alias("front_id"),
            pl.col("volume").max().alias("front_volume"),
        )
        .sort("date")
        .with_columns((pl.col("front_volume") / pl.col("total")).alias("front_share"))
    )
    changed = (pl.col("front_id") != pl.col("front_id").shift(1)).fill_null(False)
    per_day = per_day.with_columns(changed.alias("front_changed"))
    flags = per_day["front_changed"].to_list()
    n = len(flags)
    near_change = [
        any(flags[j] for j in range(max(0, i - buffer_days), min(n, i + buffer_days + 1)))
        for i in range(n)
    ]
    return per_day.with_columns(
        ((pl.col("front_share") < min_front_share) | pl.Series(near_change)).alias("is_roll")
    ).select("date", "front_id", "front_share", "front_changed", "is_roll")


def roll_dates(table: pl.DataFrame) -> set[dt.date]:
    return set(table.filter(pl.col("is_roll"))["date"].to_list())
