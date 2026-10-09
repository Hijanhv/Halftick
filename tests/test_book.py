"""Order book: each action type, edge cases, and property-based invariants."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from halftick.book.l2book import (
    FLAG_CROSSED,
    FLAG_LOCKED,
    FLAG_NEGATIVE_SIZE,
    FLAG_OUT_OF_RANGE,
    OrderBook,
    build_book_frame,
)
from halftick.schema import ADD, ASK, BID, CANCEL, CLEAR, MODIFY, TRADE


def two_sided() -> OrderBook:
    b = OrderBook(base_price=1000, window=256)
    b.apply(ADD, BID, 1000, 10)
    b.apply(ADD, BID, 999, 20)
    b.apply(ADD, ASK, 1001, 15)
    b.apply(ADD, ASK, 1002, 25)
    return b


def test_add_sets_best_prices_and_sizes() -> None:
    b = two_sided()
    assert (b.best_bid, b.best_ask) == (1000, 1001)
    assert b.size_at(BID, 1000) == 10
    assert b.size_at(ASK, 1002) == 25
    b.apply(ADD, BID, 1000, 5)
    assert b.size_at(BID, 1000) == 15


def test_cancel_reduces_and_empty_best_moves_outward() -> None:
    b = two_sided()
    b.apply(CANCEL, BID, 1000, 4)
    assert b.size_at(BID, 1000) == 6 and b.best_bid == 1000
    b.apply(CANCEL, BID, 1000, 6)
    assert b.size_at(BID, 1000) == 0 and b.best_bid == 999


def test_modify_sets_absolute_size() -> None:
    b = two_sided()
    b.apply(MODIFY, ASK, 1001, 3)
    assert b.size_at(ASK, 1001) == 3
    b.apply(MODIFY, ASK, 1001, 0)
    assert b.best_ask == 1002
    b.apply(MODIFY, BID, 1000, 0)
    assert b.best_bid == 999


def test_trade_consumes_the_opposite_side() -> None:
    b = two_sided()
    b.apply(TRADE, BID, 1001, 5)  # buyer lifts the ask
    assert b.size_at(ASK, 1001) == 10
    b.apply(TRADE, ASK, 1000, 10)  # seller hits the whole bid level
    assert b.best_bid == 999
    assert b.size_at(BID, 1000) == 0


def test_clear_empties_the_book() -> None:
    b = two_sided()
    b.apply(CLEAR, 0, 0, 0)
    assert b.best_bid is None and b.best_ask is None
    assert b.size_at(BID, 999) == 0


def test_problem_flags() -> None:
    b = two_sided()
    assert b.apply(CANCEL, BID, 999, 50) & FLAG_NEGATIVE_SIZE
    assert b.size_at(BID, 999) == 0
    assert b.apply(ADD, BID, 10_000, 1) & FLAG_OUT_OF_RANGE
    assert b.apply(ADD, BID, 1001, 1) & FLAG_LOCKED
    assert b.apply(ADD, BID, 1002, 1) & FLAG_CROSSED


def test_depth_is_on_a_dense_price_grid() -> None:
    b = OrderBook(base_price=1000, window=256)
    b.apply(ADD, BID, 1000, 10)
    b.apply(ADD, BID, 998, 7)  # one empty tick between levels
    b.apply(ADD, ASK, 1001, 4)
    bid, ask = b.depth(3)
    assert bid.tolist() == [10, 0, 7]
    assert ask.tolist() == [4, 0, 0]


def test_batch_replay_matches_object_book() -> None:
    events = [
        (ADD, BID, 1000, 10),
        (ADD, ASK, 1001, 8),
        (TRADE, ASK, 1000, 10),
        (ADD, ASK, 1000, 3),
    ]
    a = np.array([e[0] for e in events], dtype=np.int8)
    s = np.array([e[1] for e in events], dtype=np.int8)
    p = np.array([e[2] for e in events], dtype=np.int64)
    z = np.array([e[3] for e in events], dtype=np.int64)
    bid_px, ask_px, _bid_d, ask_d, _ = build_book_frame(a, s, p, z, 2, 256)
    ob = OrderBook(base_price=1000, window=256)
    for k, e in enumerate(events):
        ob.apply(*e)
        assert bid_px[k] == (ob.best_bid if ob.best_bid is not None else bid_px[k])
        assert ask_px[k] == (ob.best_ask if ob.best_ask is not None else ask_px[k])
    assert ob.best_ask == 1000 and ob.best_bid is None
    assert ask_d[-1].tolist() == [3, 8]


# Property-based: random valid order flow against a dictionary reference model.
op = st.tuples(
    st.sampled_from(["add", "cancel", "trade"]),
    st.sampled_from([BID, ASK]),
    st.integers(min_value=0, max_value=9),  # distance from the mid area, in ticks
    st.integers(min_value=1, max_value=50),
)


@settings(max_examples=200, deadline=None)
@given(st.lists(op, min_size=1, max_size=120))
def test_book_matches_reference_and_never_crosses(ops: list[tuple[str, int, int, int]]) -> None:
    mid = 5000
    book = OrderBook(base_price=mid, window=512)
    ref: dict[int, dict[int, int]] = {BID: {}, ASK: {}}

    def best(side: int) -> int | None:
        live = [p for p, q in ref[side].items() if q > 0]
        if not live:
            return None
        return max(live) if side == BID else min(live)

    for kind, side, dist, size in ops:
        price = mid - 1 - dist if side == BID else mid + dist
        other = ASK if side == BID else BID
        ob = best(other)
        # keep the flow valid: never add through the opposite touch
        if kind == "add":
            if ob is not None and ((side == BID and price >= ob) or (side == ASK and price <= ob)):
                continue
            ref[side][price] = ref[side].get(price, 0) + size
            flags = book.apply(ADD, side, price, size)
        elif kind == "cancel":
            have = ref[side].get(price, 0)
            if have == 0:
                continue
            c = min(size, have)
            ref[side][price] = have - c
            flags = book.apply(CANCEL, side, price, c)
        else:
            target = best(other)
            if target is None:
                continue
            have = ref[other][target]
            v = min(size, have)
            ref[other][target] = have - v
            flags = book.apply(TRADE, side, target, v)
        assert flags == 0
        assert book.best_bid == best(BID)
        assert book.best_ask == best(ASK)
        if book.best_bid is not None and book.best_ask is not None:
            assert book.best_bid < book.best_ask
        for s in (BID, ASK):
            for p, q in ref[s].items():
                assert book.size_at(s, p) == q >= 0
