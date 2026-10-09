"""Walk-forward execution simulation across the full sensitivity grid.

For every fold (train days -> calibration day -> block of test days):

1. fit the direction model (LightGBM, calibrated) on the train days;
2. simulate every grid cell (queue model x latency x deadline) on the train
   days and fit one cost-to-go model per queue model on the checkpoint rows;
3. on the calibration day, pick the model_direction threshold per cell;
4. on the test days, run all strategies on the same decisions (a paired
   comparison) and write every order to the trade log.

Decision times are random but seeded by date, so every strategy and every cell
faces exactly the same set of orders.
"""

from __future__ import annotations

import datetime as dt
import itertools
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

from halftick.config import Settings
from halftick.data.pipeline import analysis_days, load_features
from halftick.data.sessions import load_releases, release_window_mask, session_bounds_ns
from halftick.log import get_logger
from halftick.models.direction import fit_direction, model_features, sample_day
from halftick.models.evaluate import DIAG_COLUMNS
from halftick.models.walk_forward import Fold, assert_no_leakage, walk_forward
from halftick.sim.kernel import MODE_JOIN_TOUCH, MODE_NEW_LEVEL, QUEUE_MODELS, simulate_decisions
from halftick.sim.policy import (
    COST_MARKET_FEATURES,
    CostModel,
    SimResult,
    checkpoint_features,
    oracle_cost,
    outcome_with_abandon,
)

log = get_logger(__name__)
F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
NS = 1_000_000_000
MS = 1_000_000


@dataclass(frozen=True)
class Cell:
    queue_model: str
    latency_ms: int
    deadline_s: int


def grid(cfg: Settings) -> list[Cell]:
    s = cfg.sim
    return [
        Cell(q, lat, d)
        for q, lat, d in itertools.product(s.queue_models, s.latencies_ms, s.deadlines_s)
    ]


@dataclass
class Day:
    date: dt.date
    ts: I64
    action: npt.NDArray[np.int8]
    side: npt.NDArray[np.int8]
    price: I64
    size: I64
    queue_pos: I64
    bid_px: I64
    ask_px: I64
    bid_depth: I64
    ask_depth: I64
    mid: F64
    feats: pl.DataFrame
    t0: I64
    sides: npt.NDArray[np.int8]
    in_release: npt.NDArray[np.bool_]
    new_level_max_q: int


def _feature_columns(cfg: Settings) -> list[str]:
    need = set(COST_MARKET_FEATURES) | set(model_features(cfg)) | set(DIAG_COLUMNS)
    need |= {
        "bid_queue",
        "ask_queue",
        "bid_depl_ms1000",
        "bid_depl_ms5000",
        "ask_depl_ms1000",
        "ask_depl_ms5000",
    }
    need -= {"ts_event", "next_dir", "ret_1s", "ret_5s", "ret_30s", "in_release_window"}
    return sorted(need)


def load_day(cfg: Settings, instrument: str, date: dt.date) -> Day:
    L = cfg.book.depth_levels
    book_cols = [
        "ts_event",
        "action",
        "side",
        "price",
        "size",
        "queue_pos",
        "bid_px",
        "ask_px",
        "mid",
        "in_session",
    ]
    depth_cols = [f"bid_sz_{i}" for i in range(L)] + [f"ask_sz_{i}" for i in range(L)]
    fcols = _feature_columns(cfg)
    df = load_features(cfg, instrument, date, columns=book_cols + depth_cols + fcols)
    ts = df["ts_event"].to_numpy()

    open_ns, close_ns = session_bounds_ns(cfg, date)
    max_d = max(cfg.sim.deadlines_s) * NS
    rng = np.random.default_rng(cfg.synthetic.seed + date.toordinal())
    t0 = np.sort(
        rng.integers(open_ns + NS, close_ns - max_d - NS, cfg.sim.decisions_per_day)
    ).astype(np.int64)
    sides = rng.choice(np.array([1, -1], dtype=np.int8), cfg.sim.decisions_per_day)
    rel = load_releases(cfg).filter(pl.col("date") == date.isoformat())["ts_ns"].to_numpy()
    in_release = release_window_mask(t0, rel, cfg.session.release_window_minutes)

    sess = df["in_session"].to_numpy()
    med_q = float(
        np.median(
            np.concatenate([df["bid_sz_0"].to_numpy()[sess], df["ask_sz_0"].to_numpy()[sess]])
        )
    )
    return Day(
        date=date,
        ts=ts,
        action=df["action"].to_numpy().astype(np.int8),
        side=df["side"].to_numpy().astype(np.int8),
        price=df["price"].to_numpy().astype(np.int64),
        size=df["size"].to_numpy().astype(np.int64),
        queue_pos=df["queue_pos"].to_numpy().astype(np.int64),
        bid_px=df["bid_px"].to_numpy().astype(np.int64),
        ask_px=df["ask_px"].to_numpy().astype(np.int64),
        bid_depth=np.ascontiguousarray(df.select(depth_cols[:L]).to_numpy().astype(np.int64)),
        ask_depth=np.ascontiguousarray(df.select(depth_cols[L:]).to_numpy().astype(np.int64)),
        mid=df["mid"].to_numpy().astype(np.float64),
        feats=df.select(fcols),
        t0=t0,
        sides=sides,
        in_release=in_release,
        new_level_max_q=max(1, int(cfg.sim.new_level_max_queue_frac * med_q)),
    )


def simulate(cfg: Settings, day: Day, cell: Cell, mode: int = MODE_JOIN_TOUCH) -> SimResult:
    out = simulate_decisions(
        day.ts,
        day.action,
        day.side,
        day.price,
        day.size,
        day.queue_pos,
        day.bid_px,
        day.ask_px,
        day.bid_depth,
        day.ask_depth,
        day.t0,
        day.sides,
        cell.deadline_s * NS,
        cell.latency_ms * MS,
        QUEUE_MODELS[cell.queue_model],
        cfg.sim.checkpoint_ms * MS,
        mode,
        day.new_level_max_q,
    )
    (
        _entry_row,
        entry_px,
        ahead0,
        filled,
        fill_row,
        fill_ts,
        fill_px,
        end_px,
        cp_row,
        cp_ahead,
        cp_tleft,
        cp_cross,
        cp_touch,
    ) = out
    r0 = np.searchsorted(day.ts, day.t0, side="right") - 1
    res = SimResult(
        t0=day.t0,
        side=day.sides,
        arrival_mid=day.mid[r0],
        entry_px=entry_px,
        ahead0=ahead0,
        filled=filled,
        fill_row=fill_row,
        fill_ts=fill_ts,
        fill_px=fill_px,
        end_px=end_px,
        cp_row=cp_row,
        cp_ahead=cp_ahead,
        cp_tleft=cp_tleft,
        cp_cross=cp_cross,
        cp_touch=cp_touch,
    )
    return res


def _adverse(day: Day, r: SimResult, passive_fill: npt.NDArray[np.bool_], h_s: int) -> F64:
    out = np.full(len(r.t0), np.nan)
    if not passive_fill.any():
        return out
    ft = r.fill_ts[passive_fill]
    i_fill = np.searchsorted(day.ts, ft, side="right") - 1
    i_after = np.searchsorted(day.ts, ft + h_s * NS, side="right") - 1
    s = r.side[passive_fill]
    out[passive_fill] = s * (day.mid[i_fill] - day.mid[i_after])
    return out


def _direction_abandon(r: SimResult, p_up: F64, theta: float) -> npt.NDArray[np.bool_]:
    rows = np.maximum(r.cp_row, 0)
    p = p_up[rows]
    p_adv = np.where(r.side[:, None] > 0, p, 1.0 - p)
    out: npt.NDArray[np.bool_] = (p_adv > theta) & (r.cp_row >= 0)
    return out


def _cost_abandon(
    cfg: Settings, day: Day, r: SimResult, cell: Cell, model: CostModel
) -> npt.NDArray[np.bool_]:
    X, _, di, ci = checkpoint_features(day.feats, r, cell.latency_ms)
    pred = model.predict(X)
    out = np.zeros(r.cp_row.shape, dtype=bool)
    out[di, ci] = pred > cfg.sim.cost_model_margin_ticks
    return out


def _exec_ts(r: SimResult, j: I64, cell: Cell, cp_ms: int) -> I64:
    """When the order finished: abandon cross, passive fill, or deadline cross."""
    lat = cell.latency_ms * MS
    deadline = r.t0 + cell.deadline_s * NS + lat
    passive_end = np.where(r.filled, r.fill_ts, deadline)
    out: I64 = np.where(j >= 0, r.t0 + j * cp_ms * MS + lat, passive_end)
    return out


def evaluate_day(
    cfg: Settings,
    instrument: str,
    fold_id: int,
    day: Day,
    cell: Cell,
    p_up: F64,
    theta: float,
    cost_model: CostModel,
) -> pl.DataFrame:
    """All strategies for one day and one cell, as trade-log rows."""
    spec = cfg.instrument(instrument)
    n = len(day.t0)
    r = simulate(cfg, day, cell)
    lat = cell.latency_ms * MS
    r0 = np.searchsorted(day.ts, day.t0, side="right") - 1
    cross_now = r.cp_cross[:, 0]

    rows: list[pl.DataFrame] = []
    base = {
        "instrument": instrument,
        "date": day.date,
        "fold": fold_id,
        "decision_id": np.arange(n),
        "decision_ts": day.t0,
        "side": day.sides,
        "queue_model": cell.queue_model,
        "latency_ms": cell.latency_ms,
        "deadline_s": cell.deadline_s,
        "arrival_mid_ticks": r.arrival_mid,
        "in_release_window": day.in_release,
        "minutes_since_open": day.feats["minutes_since_open"].to_numpy()[r0],
        "vol_300s": day.feats["vol_300s"].to_numpy()[r0],
        "own_queue": np.where(
            day.sides > 0,
            day.feats["bid_queue"].to_numpy()[r0],
            day.feats["ask_queue"].to_numpy()[r0],
        ),
        "spread_ticks": day.feats["spread"].to_numpy()[r0],
    }

    def add(
        strategy: str,
        cost: F64,
        order_type: list[str] | str,
        entry_px: F64,
        ahead: F64,
        passive_fill: npt.NDArray[np.bool_],
        fill_px: F64,
        exec_ts: I64,
        src: SimResult = r,
    ) -> None:
        rows.append(
            pl.DataFrame(
                {
                    **base,
                    "strategy": strategy,
                    "order_type": order_type if isinstance(order_type, list) else [order_type] * n,
                    "entry_px_ticks": entry_px,
                    "queue_ahead_at_entry": ahead,
                    "filled_passive": passive_fill,
                    "fill_px_ticks": fill_px,
                    "exec_ts": exec_ts,
                    "time_to_fill_ms": (exec_ts - day.t0) / 1e6,
                    "cost_ticks": cost,
                    "adverse_1s": _adverse(day, src, passive_fill, 1),
                    "adverse_5s": _adverse(day, src, passive_fill, 5),
                }
            )
        )

    nan = np.full(n, np.nan)
    # always_cross
    add(
        "always_cross",
        r.cost(cross_now),
        "market",
        nan,
        nan,
        np.zeros(n, bool),
        cross_now.astype(float),
        day.t0 + lat,
    )
    # always_post
    j_none = np.full(n, -1)
    add(
        "always_post",
        r.cost(r.passive_px),
        ["limit" if f else "limit_then_market" for f in r.filled],
        r.entry_px.astype(float),
        r.ahead0,
        r.filled,
        r.passive_px.astype(float),
        _exec_ts(r, j_none, cell, cfg.sim.checkpoint_ms),
    )
    # model_direction and model_cost
    for name, abandon in (
        ("model_direction", _direction_abandon(r, p_up, theta)),
        ("model_cost", _cost_abandon(cfg, day, r, cell, cost_model)),
    ):
        cost, pf, j = outcome_with_abandon(r, abandon)
        otype = [
            "market" if jj == 0 else ("limit_then_market" if (jj > 0 or not f) else "limit")
            for jj, f in zip(j, pf, strict=True)
        ]
        px = np.where(j >= 0, r.cp_cross[np.arange(n), np.maximum(j, 0)], r.passive_px).astype(
            float
        )
        add(
            name,
            cost,
            otype,
            np.where(j == 0, np.nan, r.entry_px),
            np.where(j == 0, np.nan, r.ahead0),
            pf,
            px,
            _exec_ts(r, j, cell, cfg.sim.checkpoint_ms),
        )
    # new_level
    rn = simulate(cfg, day, cell, MODE_NEW_LEVEL)
    joined = rn.entry_px != 0
    add(
        "new_level",
        rn.cost(rn.passive_px),
        [
            "limit" if f else "limit_then_market" if jn else "market"
            for f, jn in zip(rn.filled, joined, strict=True)
        ],
        np.where(joined, rn.entry_px, np.nan),
        np.where(joined, rn.ahead0, np.nan),
        rn.filled,
        rn.passive_px.astype(float),
        np.where(rn.filled, rn.fill_ts, day.t0 + cell.deadline_s * NS + lat),
        src=rn,
    )
    # oracle (UPPER BOUND: uses future information)
    add("oracle", oracle_cost(r), "oracle_upper_bound", nan, nan, np.zeros(n, bool), nan, day.t0)

    return finalize_trade_log(pl.concat(rows), spec.tick_value_usd, cfg.fees.per_side_usd)


# "Not applicable" values (no passive fill, no resting order) are stored as
# nulls, not NaN: SQL aggregates skip nulls but propagate NaN.
NOT_APPLICABLE = [
    "entry_px_ticks",
    "queue_ahead_at_entry",
    "fill_px_ticks",
    "adverse_1s",
    "adverse_5s",
]


def finalize_trade_log(
    log_rows: pl.DataFrame, tick_value_usd: float, fee_per_side_usd: float
) -> pl.DataFrame:
    """Add fees and dollar costs. One contract per order means exactly one fee per order."""
    return log_rows.with_columns(
        pl.lit(fee_per_side_usd).alias("fee_usd"),
        (pl.col("cost_ticks") * tick_value_usd + fee_per_side_usd).alias("cost_usd"),
        *[pl.col(c).fill_nan(None) for c in NOT_APPLICABLE],
    )


def _tune_theta(cfg: Settings, day: Day, cell: Cell, p_up: F64) -> float:
    r = simulate(cfg, day, cell)
    normal = ~day.in_release
    best, best_cost = cfg.sim.direction_thresholds[0], np.inf
    for theta in cfg.sim.direction_thresholds:
        cost, _, _ = outcome_with_abandon(r, _direction_abandon(r, p_up, theta))
        c = float(np.mean(cost[normal]))
        if c < best_cost:
            best, best_cost = theta, c
    return best


def run_simulation(cfg: Settings, instrument: str) -> Path:
    t_start = time.perf_counter()
    days = analysis_days(cfg, instrument)
    wf = cfg.sim.walk_forward
    folds = list(walk_forward(days, wf.min_train_days, 1, wf.test_block_days))
    if not folds:
        raise RuntimeError(
            f"{instrument}: not enough days ({len(days)}) for min_train_days={wf.min_train_days}"
        )
    cells = grid(cfg)
    out_dir = cfg.paths.trade_logs / instrument
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.parquet"):
        old.unlink()
    dir_cols = sorted(
        set(model_features(cfg)) | {"ts_event", "in_session", "in_release_window", "next_dir"}
    )
    thetas: list[dict[str, object]] = []
    per_cell_rows = max(
        1, cfg.sim.cost_model_rows_per_day // (len(cfg.sim.latencies_ms) * len(cfg.sim.deadlines_s))
    )

    for k, fold in enumerate(folds):
        assert_no_leakage(fold)
        seed = cfg.synthetic.seed + 1000 + k
        n = cfg.models.sample_rows_per_day
        train_s = pl.concat(
            [sample_day(cfg, instrument, d, n, seed + d.toordinal(), dir_cols) for d in fold.train]
        )
        cal_s = pl.concat(
            [
                sample_day(cfg, instrument, d, n, seed + d.toordinal(), dir_cols)
                for d in fold.calibrate
            ]
        )
        direction = fit_direction(cfg, train_s, cal_s, seed)
        del train_s, cal_s

        cost_models = _fit_cost_models(cfg, instrument, fold, cells, per_cell_rows, seed)

        theta_by_cell: dict[Cell, float] = {}
        for d in fold.calibrate:
            day = load_day(cfg, instrument, d)
            p_up = direction.predict(day.feats.select(direction.features))["lightgbm"]
            for cell in cells:
                theta_by_cell[cell] = _tune_theta(cfg, day, cell, p_up)
                thetas.append({"fold": k, **cell.__dict__, "theta": theta_by_cell[cell]})
        for d in fold.test:
            day = load_day(cfg, instrument, d)
            p_up = direction.predict(day.feats.select(direction.features))["lightgbm"]
            logs = [
                evaluate_day(
                    cfg,
                    instrument,
                    k,
                    day,
                    cell,
                    p_up,
                    theta_by_cell[cell],
                    cost_models[cell.queue_model],
                )
                for cell in cells
            ]
            pl.concat(logs).write_parquet(out_dir / f"{d.isoformat()}.parquet")
            log.info(
                "sim_test_day_done",
                instrument=instrument,
                fold=k,
                date=d.isoformat(),
                elapsed_s=round(time.perf_counter() - t_start, 1),
            )
    pl.DataFrame(thetas).write_csv(cfg.paths.tables / f"direction_thresholds_{instrument}.csv")
    return out_dir


def _fit_cost_models(
    cfg: Settings, instrument: str, fold: Fold, cells: list[Cell], per_cell_rows: int, seed: int
) -> dict[str, CostModel]:
    xs: dict[str, list[pl.DataFrame]] = {q: [] for q in cfg.sim.queue_models}
    ys: dict[str, list[F64]] = {q: [] for q in cfg.sim.queue_models}
    for d in fold.train:
        day = load_day(cfg, instrument, d)
        normal = ~day.in_release
        for i, cell in enumerate(cells):
            r = simulate(cfg, day, cell)
            X, y, di, _ = checkpoint_features(
                day.feats, r, cell.latency_ms, per_cell_rows, seed + i + d.toordinal()
            )
            keep = normal[di]
            xs[cell.queue_model].append(X.filter(pl.Series(keep)))
            ys[cell.queue_model].append(y[keep])
    models: dict[str, CostModel] = {}
    for q in cfg.sim.queue_models:
        models[q] = CostModel(cfg.sim.cost_model, seed).fit(pl.concat(xs[q]), np.concatenate(ys[q]))
    return models
