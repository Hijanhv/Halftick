"""Queue-aware fill simulation (numba).

For each decision "buy (or sell) 1 contract within D seconds" the kernel
follows a passive limit order through the recorded event stream:

* the order is sent at t0 and arrives after the latency L; it joins the back of
  the queue at the near touch, so queue_ahead = displayed size at that moment;
* trades at our price reduce queue_ahead; the order fills when a trade at our
  price is larger than what is still ahead of us, or when the market trades
  through our price, or when the opposite side reaches our price;
* cancellations at our price reduce queue_ahead according to the queue model:
    pessimistic   cancels always come from behind us (ahead only shrinks via trades)
    proportional  cancels hit ahead/behind in proportion to volume
    optimistic    cancels come from ahead of us first
    exact         use the true position of the cancelled volume (MBO-style truth;
                  the synthetic generator records it, real mbp data does not)
* queue_ahead can never exceed the size displayed at our level.

While the order rests, the kernel records checkpoints every `checkpoint_ns`.
Each checkpoint stores the state a strategy would see (row index, queue ahead,
time left) and the price it would pay if it gave up and crossed right then
(the book after the latency). Strategies are then evaluated in numpy on these
arrays, so the expensive event walk happens once per decision.

Mode NEW_LEVEL instead waits for a fresh price level to open on our side (a
small queue at a better price, i.e. near the front of the line), joins it,
and otherwise crosses at the deadline.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import njit

from halftick.schema import ASK, BID, CANCEL, MODIFY, TRADE

QM_PESSIMISTIC, QM_PROPORTIONAL, QM_OPTIMISTIC, QM_EXACT = 0, 1, 2, 3
QUEUE_MODELS = {
    "pessimistic": QM_PESSIMISTIC,
    "proportional": QM_PROPORTIONAL,
    "optimistic": QM_OPTIMISTIC,
    "exact": QM_EXACT,
}
MODE_JOIN_TOUCH, MODE_NEW_LEVEL = 0, 1

I64 = npt.NDArray[np.int64]
F64 = npt.NDArray[np.float64]


@njit(cache=True)
def row_at(ts: I64, t: int) -> int:
    """Index of the last event at or before t (the book state in force at time t)."""
    return int(np.searchsorted(ts, t, side="right")) - 1


@njit(cache=True)
def _own_level_size(
    bid_px: I64, ask_px: I64, bid_depth: I64, ask_depth: I64, k: int, s: int, p: int
) -> int:
    """Displayed size at price p on our side at row k: -1 if beyond tracked depth,
    -2 if p is better than the current best (our level has emptied)."""
    L = bid_depth.shape[1]
    if s == 1:
        off = bid_px[k] - p
        if off < 0:
            return -2
        return int(bid_depth[k, off]) if off < L else -1
    off = p - ask_px[k]
    if off < 0:
        return -2
    return int(ask_depth[k, off]) if off < L else -1


@njit(cache=True)
def _apply_cancel(ahead: float, c: float, before: float, qpos: int, qm: int) -> float:
    if qm == QM_PESSIMISTIC:
        return ahead
    if qm == QM_OPTIMISTIC:
        return ahead - min(c, ahead)
    if qm == QM_EXACT and qpos >= 0:
        overlap = min(qpos + c, ahead) - max(float(qpos), 0.0)
        return ahead - max(overlap, 0.0)
    # proportional (also the fallback for exact when the position is unknown)
    if before > 0:
        return ahead - c * min(ahead, before) / before
    return ahead


@njit(cache=True)
def simulate_decisions(
    ts: I64,
    action: npt.NDArray[np.int8],
    side: npt.NDArray[np.int8],
    price: I64,
    size: I64,
    queue_pos: I64,
    bid_px: I64,
    ask_px: I64,
    bid_depth: I64,
    ask_depth: I64,
    t0s: I64,
    sides: npt.NDArray[np.int8],
    deadline_ns: int,
    latency_ns: int,
    queue_model: int,
    checkpoint_ns: int,
    mode: int,
    new_level_max_q: int,
) -> tuple[
    I64,
    I64,
    F64,
    npt.NDArray[np.bool_],
    I64,
    I64,
    I64,
    I64,
    I64,
    F64,
    F64,
    I64,
    npt.NDArray[np.int8],
]:
    """Simulate every decision. Returns, per decision:

    entry_row, entry_px, ahead_at_entry, filled, fill_row, fill_ts, fill_px,
    deadline_cross_px, and checkpoint matrices [n_decisions, max_checkpoints]:
    cp_row, cp_ahead, cp_tleft_s, cp_cross_px, cp_at_touch (-1 rows are unused).
    """
    n_dec = t0s.shape[0]
    max_cp = deadline_ns // checkpoint_ns + 1
    entry_row = np.full(n_dec, -1, dtype=np.int64)
    entry_px = np.zeros(n_dec, dtype=np.int64)
    ahead0 = np.full(n_dec, np.nan)
    filled = np.zeros(n_dec, dtype=np.bool_)
    fill_row = np.full(n_dec, -1, dtype=np.int64)
    fill_ts = np.full(n_dec, -1, dtype=np.int64)
    fill_px = np.zeros(n_dec, dtype=np.int64)
    end_px = np.zeros(n_dec, dtype=np.int64)
    cp_row = np.full((n_dec, max_cp), -1, dtype=np.int64)
    cp_ahead = np.full((n_dec, max_cp), np.nan)
    cp_tleft = np.full((n_dec, max_cp), np.nan)
    cp_cross = np.zeros((n_dec, max_cp), dtype=np.int64)
    cp_touch = np.zeros((n_dec, max_cp), dtype=np.int8)
    n = ts.shape[0]

    for d in range(n_dec):
        t0 = t0s[d]
        s = int(sides[d])
        t_end = t0 + deadline_ns
        r0 = row_at(ts, t0)
        if r0 < 0:
            continue
        # Checkpoint 0 is the decision moment itself: cross now, or post.
        n_cp = 0
        rc = row_at(ts, t0 + latency_ns)
        cp_row[d, 0] = r0
        cp_ahead[d, 0] = float(bid_depth[r0, 0] if s == 1 else ask_depth[r0, 0])
        cp_tleft[d, 0] = deadline_ns / 1e9
        cp_cross[d, 0] = ask_px[rc] if s == 1 else bid_px[rc]
        cp_touch[d, 0] = 1
        n_cp = 1
        next_cp = t0 + checkpoint_ns

        # --- find the entry ---------------------------------------------------
        if mode == MODE_JOIN_TOUCH:
            e = row_at(ts, t0 + latency_ns)
            p = bid_px[e] if s == 1 else ask_px[e]
            ahead = float(bid_depth[e, 0] if s == 1 else ask_depth[e, 0])
        else:
            e = -1
            p = 0
            ahead = 0.0
            k = row_at(ts, t0 + latency_ns) + 1
            while k < n and ts[k] <= t_end:
                if s == 1:
                    fresh = bid_px[k] > bid_px[k - 1] and bid_depth[k, 0] <= new_level_max_q
                else:
                    fresh = ask_px[k] < ask_px[k - 1] and ask_depth[k, 0] <= new_level_max_q
                if fresh:
                    pn = bid_px[k] if s == 1 else ask_px[k]
                    ej = row_at(ts, ts[k] + latency_ns)
                    if ts[k] + latency_ns <= t_end:
                        marketable = (s == 1 and pn >= ask_px[ej]) or (s == -1 and pn <= bid_px[ej])
                        if not marketable:
                            sz = _own_level_size(bid_px, ask_px, bid_depth, ask_depth, ej, s, pn)
                            e = ej
                            p = pn
                            ahead = 0.0 if sz == -2 else float(max(sz, 0))
                            break
                k += 1
            if e < 0:
                rd = row_at(ts, t_end + latency_ns)
                end_px[d] = ask_px[rd] if s == 1 else bid_px[rd]
                continue
        entry_row[d] = e
        entry_px[d] = p
        ahead0[d] = ahead

        # --- walk the event stream while the order rests ------------------------
        k = e + 1
        done = False
        while k < n and ts[k] <= t_end:
            if mode == MODE_JOIN_TOUCH:
                while next_cp < ts[k] and next_cp < t_end and n_cp < max_cp:
                    rcx = row_at(ts, next_cp + latency_ns)
                    cp_row[d, n_cp] = k - 1
                    cp_ahead[d, n_cp] = ahead
                    cp_tleft[d, n_cp] = (t_end - next_cp) / 1e9
                    cp_cross[d, n_cp] = ask_px[rcx] if s == 1 else bid_px[rcx]
                    own_best = bid_px[k - 1] if s == 1 else ask_px[k - 1]
                    cp_touch[d, n_cp] = 1 if own_best == p else 0
                    n_cp += 1
                    next_cp += checkpoint_ns
            a = action[k]
            if a == TRADE:
                # The aggressor on the other side hits resting orders on our side.
                hits_us = (s == 1 and side[k] == ASK) or (s == -1 and side[k] == BID)
                if hits_us:
                    through = (s == 1 and price[k] < p) or (s == -1 and price[k] > p)
                    if through:
                        done = True
                    elif price[k] == p:
                        if size[k] > ahead:
                            done = True
                        else:
                            ahead -= size[k]
            elif (
                (a == CANCEL or a == MODIFY)
                and price[k] == p
                and ((s == 1 and side[k] == BID) or (s == -1 and side[k] == ASK))
            ):
                before = _own_level_size(bid_px, ask_px, bid_depth, ask_depth, k - 1, s, p)
                if a == CANCEL:
                    ahead = _apply_cancel(
                        ahead, float(size[k]), float(before), int(queue_pos[k]), queue_model
                    )
                elif before > 0 and size[k] < before:
                    # MODIFY to a smaller size: a partial cancel at unknown position.
                    ahead = _apply_cancel(
                        ahead, float(before - size[k]), float(before), -1, queue_model
                    )
            if not done:
                # Opposite side reached our price: an incoming order would have matched us.
                if (s == 1 and ask_px[k] <= p) or (s == -1 and bid_px[k] >= p):
                    done = True
            if not done:
                now = _own_level_size(bid_px, ask_px, bid_depth, ask_depth, k, s, p)
                if now == -2:
                    ahead = 0.0  # everyone else at our price left; we are alone at the front
                elif now >= 0 and ahead > now:
                    ahead = float(now)
            if done:
                filled[d] = True
                fill_row[d] = k
                fill_ts[d] = ts[k]
                fill_px[d] = p
                break
            k += 1
        if not done:
            if mode == MODE_JOIN_TOUCH:
                last = min(k - 1, n - 1)
                while next_cp < t_end and n_cp < max_cp:
                    rcx = row_at(ts, next_cp + latency_ns)
                    cp_row[d, n_cp] = last
                    cp_ahead[d, n_cp] = ahead
                    cp_tleft[d, n_cp] = (t_end - next_cp) / 1e9
                    cp_cross[d, n_cp] = ask_px[rcx] if s == 1 else bid_px[rcx]
                    own_best = bid_px[last] if s == 1 else ask_px[last]
                    cp_touch[d, n_cp] = 1 if own_best == p else 0
                    n_cp += 1
                    next_cp += checkpoint_ns
            rd = row_at(ts, t_end + latency_ns)
            end_px[d] = ask_px[rd] if s == 1 else bid_px[rd]
    return (
        entry_row,
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
    )
