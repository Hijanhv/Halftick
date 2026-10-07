"""Prediction targets. Every target at row i looks only at rows after i.

* next_dir: direction of the next mid-price change in event time
  (1 = up, 0 = down, NaN when the day ends first). The Gould-Bonart target.
* ret_{h}s: mid change in ticks over the next h seconds of clock time, using
  the last book state at or before ts + h. NaN if the day ends first.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import njit

from halftick.schema import NO_ASK, NO_BID

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]


def mid_ticks(bid_px: I64, ask_px: I64) -> F64:
    valid = (bid_px != NO_BID) & (ask_px != NO_ASK)
    return np.where(valid, 0.5 * (bid_px + ask_px), np.nan)


@njit(cache=True)
def next_change(mid: F64, ts: I64) -> tuple[F64, F64]:
    """Direction of and time (ms) to the next mid change, scanning backwards."""
    n = mid.shape[0]
    direction = np.full(n, np.nan)
    wait_ms = np.full(n, np.nan)
    nxt_mid = np.nan
    nxt_ts = 0
    for i in range(n - 1, -1, -1):
        if np.isnan(mid[i]):
            continue
        if not np.isnan(nxt_mid) and nxt_mid != mid[i]:
            direction[i] = 1.0 if nxt_mid > mid[i] else 0.0
            wait_ms[i] = (nxt_ts - ts[i]) / 1e6
        # Row i becomes the "next change" reference for earlier rows only if it
        # differs from what came before it; otherwise keep the later reference.
        if i == 0 or np.isnan(mid[i - 1]) or mid[i - 1] != mid[i]:
            nxt_mid = mid[i]
            nxt_ts = ts[i]
    return direction, wait_ms


def forward_return(mid: F64, ts: I64, horizon_ns: int) -> F64:
    idx = np.searchsorted(ts, ts + horizon_ns, side="right") - 1
    out = mid[idx] - mid
    out[ts + horizon_ns > ts[-1]] = np.nan
    return out


def compute_targets(frame: pl.DataFrame, horizons_s: list[int]) -> pl.DataFrame:
    ts = frame["ts_event"].to_numpy()
    mid = mid_ticks(frame["bid_px"].to_numpy(), frame["ask_px"].to_numpy())
    direction, wait = next_change(mid, ts)
    cols: dict[str, F64] = {"mid": mid, "next_dir": direction, "next_change_ms": wait}
    for h in horizons_s:
        cols[f"ret_{h}s"] = forward_return(mid, ts, h * 1_000_000_000)
    return pl.DataFrame(cols)
