"""Turning simulated paths into strategy outcomes, and the cost-to-go model.

Strategies (all trade exactly one contract per decision):

* always_cross     market order at t0 (fills after latency at the touch then)
* always_post      join the near touch, cross at the deadline if unfilled
* model_direction  post; at each checkpoint give up and cross if the calibrated
                   probability of an adverse next mid move exceeds a threshold
                   tuned on the calibration day
* model_cost       post; at each checkpoint give up and cross if a regression
                   model predicts that waiting costs more than crossing now
* new_level        wait for a fresh level to open on our side and join it near
                   the front of the queue; cross at the deadline otherwise
* oracle           UPPER BOUND using future information: the cheapest of
                   crossing at any checkpoint or waiting to the end

Costs are in ticks relative to the arrival mid (the mid at t0), signed so that
a positive number is money paid: buy at 1 tick above mid costs +1.

The cost-to-go label at checkpoint j is

    (price paid if we keep waiting until fill or deadline) - (price paid if we cross at j)

multiplied by the side sign. A positive prediction means waiting is worse, so
the policy crosses. Comparing "cross now" against "never give up" one step at
a time is a one-step policy improvement over always_post: if the model were
exact it could only lower the expected cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import polars as pl

from halftick.features.compute import is_directional

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]

STRATEGIES = ("always_cross", "always_post", "model_direction", "model_cost", "new_level", "oracle")

# Market features used by the cost model, from the order's point of view.
COST_MARKET_FEATURES = [
    "imbalance",
    "microprice",
    "depth_imb_3",
    "depth_imb_5",
    "depth_imb_10",
    "ofi_ev10",
    "ofi_ev50",
    "ofi_ev200",
    "ofi_ms100",
    "ofi_ms1000",
    "ofi_ms5000",
    "tfi_ms100",
    "tfi_ms1000",
    "tfi_ms5000",
    "spread",
    "since_mid_change_ms",
    "vol_30s",
    "vol_300s",
]
OWN_OPP = [
    "own_queue",
    "opp_queue",
    "own_depl_ms1000",
    "own_depl_ms5000",
    "opp_depl_ms1000",
    "opp_depl_ms5000",
]
ORDER_STATE = ["ahead", "ahead_frac", "tleft_s", "at_touch", "latency_ms"]
COST_FEATURES = COST_MARKET_FEATURES + OWN_OPP + ORDER_STATE


@dataclass
class SimResult:
    """Output of kernel.simulate_decisions plus decision metadata, for one day and one cell."""

    t0: I64
    side: npt.NDArray[np.int8]
    arrival_mid: F64
    entry_px: I64
    ahead0: F64
    filled: npt.NDArray[np.bool_]
    fill_row: I64
    fill_ts: I64
    fill_px: I64
    end_px: I64
    cp_row: I64
    cp_ahead: F64
    cp_tleft: F64
    cp_cross: I64
    cp_touch: npt.NDArray[np.int8]

    @property
    def passive_px(self) -> I64:
        """What waiting to the end pays: the fill price, or the deadline cross."""
        out: I64 = np.where(self.filled, self.fill_px, self.end_px)
        return out

    def cost(self, px: I64 | F64) -> F64:
        out: F64 = self.side * (px - self.arrival_mid)
        return out


def first_true(mask: npt.NDArray[np.bool_]) -> I64:
    """Column index of the first True per row, or -1."""
    any_ = mask.any(axis=1)
    idx: I64 = np.where(any_, mask.argmax(axis=1), -1)
    return idx


def outcome_with_abandon(
    r: SimResult, abandon: npt.NDArray[np.bool_]
) -> tuple[F64, npt.NDArray[np.bool_], I64]:
    """Cost when a policy gives up at the first checkpoint where `abandon` is True.

    Returns (cost, ended_passive_fill, abandon_checkpoint_index).
    """
    valid = r.cp_row >= 0
    j = first_true(abandon & valid)
    rows = np.arange(len(j))
    px = np.where(j >= 0, r.cp_cross[rows, np.maximum(j, 0)], r.passive_px)
    passive_fill = (j < 0) & r.filled
    return r.cost(px), passive_fill, j


def oracle_cost(r: SimResult) -> F64:
    valid = r.cp_row >= 0
    cross_costs = np.where(valid, r.side[:, None] * (r.cp_cross - r.arrival_mid[:, None]), np.inf)
    out: F64 = np.minimum(cross_costs.min(axis=1), r.cost(r.passive_px))
    return out


def cost_to_go_label(r: SimResult) -> F64:
    out: F64 = r.side[:, None] * (r.passive_px[:, None] - r.cp_cross)
    return out


def checkpoint_features(
    feats: pl.DataFrame, r: SimResult, latency_ms: int, max_rows: int | None = None, seed: int = 0
) -> tuple[pl.DataFrame, F64, npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Feature rows for every valid checkpoint (optionally subsampled).

    feats must contain the market features, bid/ask queue and depletion columns.
    Returns (X, label, decision_index, checkpoint_index).
    """
    valid = r.cp_row >= 0
    dec_idx, cp_idx = np.nonzero(valid)
    if max_rows is not None and dec_idx.size > max_rows:
        pick = np.sort(np.random.default_rng(seed).choice(dec_idx.size, max_rows, replace=False))
        dec_idx, cp_idx = dec_idx[pick], cp_idx[pick]
    rows = r.cp_row[dec_idx, cp_idx]
    s = r.side[dec_idx].astype(np.float64)
    sub = feats[rows]
    cols: dict[str, F64] = {}
    for name in COST_MARKET_FEATURES:
        v = sub[name].to_numpy().astype(np.float64)
        cols[name] = v * s if is_directional(name) else v
    buy = s > 0
    bq, aq = sub["bid_queue"].to_numpy(), sub["ask_queue"].to_numpy()
    cols["own_queue"] = np.where(buy, bq, aq)
    cols["opp_queue"] = np.where(buy, aq, bq)
    for w in (1000, 5000):
        b, a = sub[f"bid_depl_ms{w}"].to_numpy(), sub[f"ask_depl_ms{w}"].to_numpy()
        cols[f"own_depl_ms{w}"] = np.where(buy, b, a)
        cols[f"opp_depl_ms{w}"] = np.where(buy, a, b)
    ahead = r.cp_ahead[dec_idx, cp_idx]
    cols["ahead"] = ahead
    cols["ahead_frac"] = ahead / np.maximum(cols["own_queue"], 1.0)
    cols["tleft_s"] = r.cp_tleft[dec_idx, cp_idx]
    cols["at_touch"] = r.cp_touch[dec_idx, cp_idx].astype(np.float64)
    cols["latency_ms"] = np.full(dec_idx.size, float(latency_ms))
    label = cost_to_go_label(r)[dec_idx, cp_idx]
    return pl.DataFrame(cols).select(COST_FEATURES), label, dec_idx, cp_idx


class CostModel:
    """LightGBM regression of the cost-to-go label (ticks)."""

    def __init__(self, params: dict[str, Any], seed: int) -> None:
        self.model = lgb.LGBMRegressor(**params, random_state=seed)

    def fit(self, X: pl.DataFrame, y: F64) -> CostModel:
        self.model.fit(X.to_pandas(), y)
        return self

    def predict(self, X: pl.DataFrame) -> F64:
        out: F64 = np.asarray(self.model.predict(X.to_pandas()), dtype=np.float64)
        return out
