"""Order-book features, computed on every event in a single backward-looking pass.

Every value at row i uses rows 0..i only. Time windows are "the last T
milliseconds up to and including this event", found with a two-pointer sweep,
so a whole day costs O(n) per window.

References:
* queue imbalance: Gould & Bonart (2016)
* order flow imbalance (OFI): Cont, Kukanov & Stoikov (2014)
* microprice: Stoikov (2018); here the simple size-weighted version, which is
  the first-order term of Stoikov's estimator
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import njit

from halftick.config import FeaturesCfg
from halftick.schema import BID, NO_ASK, NO_BID, TRADE

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]


@dataclass(frozen=True)
class FeatureLayout:
    names: list[str]
    windows_ms: list[int]
    windows_events: list[int]
    depth_levels: list[int]
    vol_windows_s: list[int]


def layout(cfg: FeaturesCfg, available_levels: int) -> FeatureLayout:
    depth = [k for k in cfg.depth_levels if k <= available_levels]
    names = ["imbalance", "microprice", "spread", "wide_spread", "bid_queue", "ask_queue"]
    names += [f"depth_imb_{k}" for k in depth]
    names += [f"ofi_ev{n}" for n in cfg.windows_events]
    names += [f"ofi_ms{t}" for t in cfg.windows_ms]
    names += [f"tfi_ms{t}" for t in cfg.windows_ms]
    names += [f"bid_depl_ms{t}" for t in cfg.windows_ms]
    names += [f"ask_depl_ms{t}" for t in cfg.windows_ms]
    names += ["since_mid_change_ms"]
    names += [f"vol_{w}s" for w in cfg.vol_windows_s]
    names += ["minutes_since_open"]
    return FeatureLayout(
        names, list(cfg.windows_ms), list(cfg.windows_events), depth, list(cfg.vol_windows_s)
    )


@njit(cache=True)
def _window_start(ts: I64, width_ns: int) -> I64:
    """For each i, the first index j with ts[j] > ts[i] - width (window is (ts-w, ts])."""
    n = ts.shape[0]
    out = np.empty(n, dtype=np.int64)
    j = 0
    for i in range(n):
        lo = ts[i] - width_ns
        while ts[j] <= lo:
            j += 1
        out[i] = j
    return out


@njit(cache=True)
def _prefix(x: F64) -> F64:
    out = np.empty(x.shape[0] + 1)
    out[0] = 0.0
    acc = 0.0
    for i in range(x.shape[0]):
        acc += x[i]
        out[i + 1] = acc
    return out


@njit(cache=True)
def compute_features_kernel(
    ts: I64,
    action: npt.NDArray[np.int8],
    side: npt.NDArray[np.int8],
    size: I64,
    bid_px: I64,
    ask_px: I64,
    bid_depth: I64,
    ask_depth: I64,
    depth_levels: I64,
    windows_ms: I64,
    windows_events: I64,
    vol_windows_s: I64,
    open_ns: int,
) -> F64:
    n = ts.shape[0]
    n_feat = (
        6
        + depth_levels.shape[0]
        + windows_events.shape[0]
        + 4 * windows_ms.shape[0]
        + 1
        + vol_windows_s.shape[0]
        + 1
    )
    out = np.full((n, n_feat), np.nan)

    valid = np.empty(n, dtype=np.bool_)
    mid = np.full(n, np.nan)
    for i in range(n):
        valid[i] = bid_px[i] != NO_BID and ask_px[i] != NO_ASK
        if valid[i]:
            mid[i] = 0.5 * (bid_px[i] + ask_px[i])

    # Per-event building blocks.
    ofi_e = np.zeros(n)
    buy_v = np.zeros(n)
    sell_v = np.zeros(n)
    mid_chg = np.zeros(n)
    for i in range(n):
        if action[i] == TRADE:
            if side[i] == BID:
                buy_v[i] = size[i]
            else:
                sell_v[i] = size[i]
        if i > 0 and valid[i] and valid[i - 1]:
            qb, qb0 = bid_depth[i, 0], bid_depth[i - 1, 0]
            qa, qa0 = ask_depth[i, 0], ask_depth[i - 1, 0]
            e = 0.0
            if bid_px[i] >= bid_px[i - 1]:
                e += qb
            if bid_px[i] <= bid_px[i - 1]:
                e -= qb0
            if ask_px[i] <= ask_px[i - 1]:
                e -= qa
            if ask_px[i] >= ask_px[i - 1]:
                e += qa0
            ofi_e[i] = e
            if mid[i] != mid[i - 1]:
                mid_chg[i] = 1.0
    c_ofi = _prefix(ofi_e)
    c_buy = _prefix(buy_v)
    c_sell = _prefix(sell_v)
    c_chg = _prefix(mid_chg)

    # Index where the current best bid / ask price first became the touch.
    bid_lvl_start = np.zeros(n, dtype=np.int64)
    ask_lvl_start = np.zeros(n, dtype=np.int64)
    last_mid_change_ts = np.empty(n, dtype=np.int64)
    lb = 0
    la = 0
    lm = ts[0]
    for i in range(n):
        if i > 0 and bid_px[i] != bid_px[i - 1]:
            lb = i
        if i > 0 and ask_px[i] != ask_px[i - 1]:
            la = i
        if mid_chg[i] > 0:
            lm = ts[i]
        bid_lvl_start[i] = lb
        ask_lvl_start[i] = la
        last_mid_change_ts[i] = lm

    col = 0
    for i in range(n):
        if not valid[i]:
            continue
        qb = float(bid_depth[i, 0])
        qa = float(ask_depth[i, 0])
        tot = qb + qa
        out[i, 0] = (qb - qa) / tot if tot > 0 else 0.0
        out[i, 1] = ((bid_px[i] * qa + ask_px[i] * qb) / tot - mid[i]) if tot > 0 else 0.0
        out[i, 2] = float(ask_px[i] - bid_px[i])
        out[i, 3] = 1.0 if ask_px[i] - bid_px[i] > 1 else 0.0
        out[i, 4] = qb
        out[i, 5] = qa
    col = 6
    for k in range(depth_levels.shape[0]):
        lv = depth_levels[k]
        for i in range(n):
            if not valid[i]:
                continue
            sb = 0.0
            sa = 0.0
            for j in range(lv):
                sb += bid_depth[i, j]
                sa += ask_depth[i, j]
            out[i, col] = (sb - sa) / (sb + sa) if sb + sa > 0 else 0.0
        col += 1
    for k in range(windows_events.shape[0]):
        w = windows_events[k]
        for i in range(n):
            if valid[i]:
                j = max(0, i - w + 1)
                out[i, col] = c_ofi[i + 1] - c_ofi[j]
        col += 1
    starts = np.empty((windows_ms.shape[0], n), dtype=np.int64)
    for k in range(windows_ms.shape[0]):
        starts[k] = _window_start(ts, windows_ms[k] * 1_000_000)
    for k in range(windows_ms.shape[0]):
        for i in range(n):
            if valid[i]:
                out[i, col] = c_ofi[i + 1] - c_ofi[starts[k, i]]
        col += 1
    for k in range(windows_ms.shape[0]):
        for i in range(n):
            if valid[i]:
                j = starts[k, i]
                b = c_buy[i + 1] - c_buy[j]
                s = c_sell[i + 1] - c_sell[j]
                out[i, col] = (b - s) / (b + s) if b + s > 0 else 0.0
        col += 1
    for side_k in range(2):
        for k in range(windows_ms.shape[0]):
            secs = windows_ms[k] / 1000.0
            for i in range(n):
                if not valid[i]:
                    continue
                if side_k == 0:
                    r = max(starts[k, i], bid_lvl_start[i])
                    out[i, col] = (bid_depth[r, 0] - bid_depth[i, 0]) / secs
                else:
                    r = max(starts[k, i], ask_lvl_start[i])
                    out[i, col] = (ask_depth[r, 0] - ask_depth[i, 0]) / secs
            col += 1
    for i in range(n):
        if valid[i]:
            out[i, col] = (ts[i] - last_mid_change_ts[i]) / 1e6
    col += 1
    for k in range(vol_windows_s.shape[0]):
        st = _window_start(ts, vol_windows_s[k] * 1_000_000_000)
        for i in range(n):
            if valid[i]:
                out[i, col] = (c_chg[i + 1] - c_chg[st[i]]) / vol_windows_s[k]
        col += 1
    for i in range(n):
        out[i, col] = (ts[i] - open_ns) / 6e10
    return out


def compute_features(
    frame: pl.DataFrame, cfg: FeaturesCfg, levels: int, open_ns: int
) -> pl.DataFrame:
    """Compute all features for one day's book frame. Returns a frame of feature columns."""
    lay = layout(cfg, levels)
    bid_depth = np.ascontiguousarray(
        frame.select([f"bid_sz_{i}" for i in range(levels)]).to_numpy().astype(np.int64)
    )
    ask_depth = np.ascontiguousarray(
        frame.select([f"ask_sz_{i}" for i in range(levels)]).to_numpy().astype(np.int64)
    )
    mat = compute_features_kernel(
        frame["ts_event"].to_numpy(),
        frame["action"].to_numpy().astype(np.int8),
        frame["side"].to_numpy().astype(np.int8),
        frame["size"].to_numpy().astype(np.int64),
        frame["bid_px"].to_numpy(),
        frame["ask_px"].to_numpy(),
        bid_depth,
        ask_depth,
        np.array(lay.depth_levels, dtype=np.int64),
        np.array(lay.windows_ms, dtype=np.int64),
        np.array(lay.windows_events, dtype=np.int64),
        np.array(lay.vol_windows_s, dtype=np.int64),
        open_ns,
    )
    return pl.DataFrame(mat, schema=lay.names)


# Features whose sign flips with the side of a resting order. Used by the
# simulator to express everything from the order's point of view.
DIRECTIONAL_PREFIXES = ("imbalance", "microprice", "depth_imb_", "ofi_", "tfi_")


def is_directional(name: str) -> bool:
    return name.startswith(DIRECTIONAL_PREFIXES)
