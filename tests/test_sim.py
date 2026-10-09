"""Queue-aware fill simulator: hand-crafted scenarios with known answers.

Scenario (buy 1 lot, decision at t = 1 s, zero latency):

    t=0     bid 100 x10, ask 101 x10
    t=1     we join the bid at 100 behind 10 lots            ahead = 10
    t=1.5   +10 lots join behind us                         level 20
    t=2     4 lots cancel from BEHIND us (queue_pos 14)     level 16
    t=3     seller trades 7 at 100                          level 9
    t=3.5   2 lots cancel from the FRONT (queue_pos 0)      level 7
    t=4     seller trades 2 at 100                          level 5
    t=5     seller trades 2 at 100                          level 3
    t=6     seller trades 2 at 100                          level 1

Queue ahead of us, by cancellation assumption:

    pessimistic   10 -> 10 -> 3 -> 3 -> 1 -> fills at t=5 (2 > 1)
    proportional  10 -> 8  -> 1 -> 0.78 -> fills at t=4
    optimistic    10 -> 6  -> fills at t=3 (7 > 6)
    exact         10 -> 10 -> 3 -> 1 -> fills at t=4
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from halftick.config import Settings
from halftick.schema import ADD, ASK, BID, CANCEL, TRADE
from halftick.sim.kernel import (
    MODE_JOIN_TOUCH,
    MODE_NEW_LEVEL,
    QUEUE_MODELS,
    simulate_decisions,
)
from halftick.sim.policy import SimResult, cost_to_go_label, oracle_cost, outcome_with_abandon
from halftick.sim.runner import finalize_trade_log
from tests.conftest import NS, book_frame

S = NS
SCENARIO = [
    (0, ADD, BID, 100, 10, -1),
    (0, ADD, ASK, 101, 10, -1),
    (int(1.5 * S), ADD, BID, 100, 10, -1),
    (2 * S, CANCEL, BID, 100, 4, 14),
    (3 * S, TRADE, ASK, 100, 7, 0),
    (int(3.5 * S), CANCEL, BID, 100, 2, 0),
    (4 * S, TRADE, ASK, 100, 2, 0),
    (5 * S, TRADE, ASK, 100, 2, 0),
    (6 * S, TRADE, ASK, 100, 2, 0),
]


def run(
    cfg: Settings,
    events: list,  # type: ignore[type-arg]
    queue_model: str,
    *,
    t0: int = 1 * S,
    side: int = 1,
    deadline_s: float = 10,
    latency_ms: int = 0,
    checkpoint_ms: int = 250,
    mode: int = MODE_JOIN_TOUCH,
    new_level_max_q: int = 5,
) -> SimResult:
    f = book_frame(cfg, events)
    L = cfg.book.depth_levels
    out = simulate_decisions(
        f["ts_event"].to_numpy(),
        f["action"].to_numpy().astype(np.int8),
        f["side"].to_numpy().astype(np.int8),
        f["price"].to_numpy(),
        f["size"].to_numpy(),
        f["queue_pos"].to_numpy(),
        f["bid_px"].to_numpy(),
        f["ask_px"].to_numpy(),
        np.ascontiguousarray(f.select([f"bid_sz_{i}" for i in range(L)]).to_numpy()),
        np.ascontiguousarray(f.select([f"ask_sz_{i}" for i in range(L)]).to_numpy()),
        np.array([t0], dtype=np.int64),
        np.array([side], dtype=np.int8),
        int(deadline_s * S),
        latency_ms * 1_000_000,
        QUEUE_MODELS[queue_model],
        checkpoint_ms * 1_000_000,
        mode,
        new_level_max_q,
    )
    ts = f["ts_event"].to_numpy()
    mid = 0.5 * (f["bid_px"].to_numpy() + f["ask_px"].to_numpy())
    r0 = np.searchsorted(ts, [t0], side="right") - 1
    (
        _,
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
    return SimResult(
        t0=np.array([t0]),
        side=np.array([side], dtype=np.int8),
        arrival_mid=mid[r0],
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


@pytest.mark.parametrize(
    ("model", "fill_s"),
    [("optimistic", 3), ("proportional", 4), ("exact", 4), ("pessimistic", 5)],
)
def test_fill_time_per_cancellation_assumption(cfg: Settings, model: str, fill_s: int) -> None:
    r = run(cfg, SCENARIO, model)
    assert r.ahead0[0] == 10
    assert r.filled[0]
    assert r.fill_ts[0] == fill_s * S
    assert r.fill_px[0] == 100
    # cost vs arrival mid 100.5: bought at 100 -> -0.5 ticks
    assert r.cost(r.passive_px)[0] == pytest.approx(-0.5)


def test_queue_ahead_path_at_checkpoints(cfg: Settings) -> None:
    r = run(cfg, SCENARIO, "proportional", checkpoint_ms=1000)
    # Checkpoint 0 = the decision moment. A checkpoint at time t sees every event
    # stamped at or before t (same rule as "the book at time t" everywhere else):
    # t=2 includes the cancel from behind (10 -> 8), t=3 includes the 7-lot trade (8 -> 1).
    ahead = r.cp_ahead[0]
    assert ahead[0] == 10
    assert ahead[1] == pytest.approx(8)
    assert ahead[2] == pytest.approx(1)


def test_deadline_cross_when_unfilled(cfg: Settings) -> None:
    r = run(cfg, SCENARIO, "pessimistic", deadline_s=3.75)
    assert not r.filled[0]
    assert r.end_px[0] == 101  # ask at the deadline
    assert r.cost(r.passive_px)[0] == pytest.approx(0.5)


def test_trade_through_our_price_fills_us(cfg: Settings) -> None:
    ev = [
        (0, ADD, BID, 100, 10, -1),
        (0, ADD, BID, 99, 10, -1),
        (0, ADD, ASK, 101, 10, -1),
        (2 * S, TRADE, ASK, 99, 3, 0),  # trades below us: our level must be gone
    ]
    r = run(cfg, ev, "pessimistic")
    assert r.filled[0] and r.fill_ts[0] == 2 * S


def test_sell_side_mirror(cfg: Settings) -> None:
    ev = [
        (0, ADD, BID, 100, 10, -1),
        (0, ADD, ASK, 101, 6, -1),
        (2 * S, TRADE, BID, 101, 5, 0),
        (3 * S, TRADE, BID, 101, 2, 0),
    ]
    r = run(cfg, ev, "pessimistic", side=-1)
    assert r.ahead0[0] == 6 and r.fill_ts[0] == 3 * S and r.fill_px[0] == 101
    assert r.cost(r.passive_px)[0] == pytest.approx(-0.5)  # sold above the mid


def test_latency_uses_the_book_after_the_delay(cfg: Settings) -> None:
    ev = [
        (0, ADD, BID, 100, 10, -1),
        (0, ADD, ASK, 101, 3, -1),
        (0, ADD, ASK, 102, 9, -1),
        (S + 5_000_000, TRADE, BID, 101, 3, 0),  # 5 ms after the decision the ask moves up
    ]
    fast = run(cfg, ev, "proportional", latency_ms=1)
    slow = run(cfg, ev, "proportional", latency_ms=10)
    assert fast.cp_cross[0, 0] == 101 and fast.cost(fast.cp_cross[:, 0])[0] == pytest.approx(0.5)
    assert slow.cp_cross[0, 0] == 102 and slow.cost(slow.cp_cross[:, 0])[0] == pytest.approx(1.5)


def test_new_level_strategy_joins_a_fresh_small_level(cfg: Settings) -> None:
    ev = [
        (0, ADD, BID, 100, 50, -1),
        (0, ADD, ASK, 102, 50, -1),  # 2-tick spread
        (2 * S, ADD, BID, 101, 2, -1),  # fresh level inside the spread
        (3 * S, TRADE, ASK, 101, 3, 0),
    ]
    r = run(cfg, ev, "pessimistic", mode=MODE_NEW_LEVEL, new_level_max_q=5)
    assert r.entry_px[0] == 101 and r.ahead0[0] == 2
    assert r.filled[0] and r.fill_ts[0] == 3 * S


def test_policy_costs_abandon_oracle_and_labels() -> None:
    # Two buy orders, arrival mid 100.5; checkpoints cross at 101, 102.
    r = SimResult(
        t0=np.array([0, 0]),
        side=np.array([1, 1], dtype=np.int8),
        arrival_mid=np.array([100.5, 100.5]),
        entry_px=np.array([100, 100]),
        ahead0=np.array([5.0, 5.0]),
        filled=np.array([True, False]),
        fill_row=np.array([3, -1]),
        fill_ts=np.array([10, -1]),
        fill_px=np.array([100, 0]),
        end_px=np.array([0, 103]),
        cp_row=np.array([[0, 1], [0, 1]]),
        cp_ahead=np.ones((2, 2)),
        cp_tleft=np.ones((2, 2)),
        cp_cross=np.array([[101, 102], [101, 102]]),
        cp_touch=np.ones((2, 2), dtype=np.int8),
    )
    assert r.cost(r.passive_px).tolist() == [-0.5, 2.5]
    cost, pf, j = outcome_with_abandon(r, np.array([[False, False], [False, True]]))
    assert cost.tolist() == [-0.5, 1.5] and pf.tolist() == [True, False] and j.tolist() == [-1, 1]
    assert oracle_cost(r).tolist() == [-0.5, 0.5]
    # waiting minus crossing at each checkpoint, in ticks
    assert cost_to_go_label(r).tolist() == [[-1, -2], [2, 1]]


def test_fees_add_one_fee_per_order_and_cancel_in_savings() -> None:
    log = pl.DataFrame(
        {
            "strategy": ["always_cross", "model_cost"],
            "cost_ticks": [0.5, -0.25],
            "entry_px_ticks": [float("nan"), 100.0],
            "queue_ahead_at_entry": [float("nan"), 7.0],
            "fill_px_ticks": [101.0, 100.0],
            "adverse_1s": [float("nan"), 0.5],
            "adverse_5s": [float("nan"), 1.0],
        }
    )
    for fee in (1.0, 2.5, 4.0):
        out = finalize_trade_log(log, 15.625, fee)
        assert out["cost_usd"].to_list() == pytest.approx(
            [0.5 * 15.625 + fee, -0.25 * 15.625 + fee]
        )
        saving = out["cost_usd"][0] - out["cost_usd"][1]
        assert saving == pytest.approx(0.75 * 15.625)  # independent of the fee
        assert out["entry_px_ticks"].null_count() == 1  # NaN stored as null


events_at_one_level = st.lists(
    st.tuples(
        st.sampled_from(["add", "cancel", "trade"]),
        st.integers(min_value=1, max_value=15),
        st.floats(min_value=0, max_value=1),
    ),
    min_size=1,
    max_size=60,
)


@settings(max_examples=150, deadline=None)
@given(events_at_one_level, st.sampled_from(list(QUEUE_MODELS)))
def test_queue_ahead_never_increases_and_never_exceeds_the_level(
    base_cfg: Settings, flow, model: str
) -> None:  # type: ignore[no-untyped-def]
    cfg = base_cfg
    ev = [(0, ADD, BID, 100, 40, -1), (0, ADD, ASK, 101, 10_000, -1)]
    level = 40
    t = S + 1
    for kind, size, pos in flow:
        t += 300_000_000
        if kind == "add":
            ev.append((t, ADD, BID, 100, size, -1))
            level += size
        elif kind == "cancel" and level > size:
            ev.append((t, CANCEL, BID, 100, size, int(pos * (level - size))))
            level -= size
        elif kind == "trade" and level > size:
            ev.append((t, TRADE, ASK, 100, size, 0))
            level -= size
    r = run(cfg, ev, model, deadline_s=(t - S) / S + 1, checkpoint_ms=100)
    path = r.cp_ahead[0][r.cp_row[0] >= 0]
    assert np.all(np.diff(path) <= 1e-9), "queue ahead increased"
    f = book_frame(cfg, ev)
    sizes = f["bid_sz_0"].to_numpy()
    rows = r.cp_row[0][r.cp_row[0] >= 0]
    assert np.all(path[1:] <= sizes[rows[1:]] + 1e-9), "queue ahead exceeds the displayed level"
