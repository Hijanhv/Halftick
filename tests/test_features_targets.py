"""Features on hand-built books, targets, no-look-ahead, and streaming = batch."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from halftick.config import Settings
from halftick.data.pipeline import book_from_events
from halftick.data.synthetic import generate_day
from halftick.features.compute import compute_features
from halftick.features.streaming import StreamingFeatures
from halftick.replay.engine import ReplayEngine
from halftick.schema import ADD, ASK, BID, CANCEL, TRADE
from halftick.targets.compute import compute_targets
from tests.conftest import NS, book_frame, full_frame

MS = 1_000_000

# A tiny book: bid 100 x10 (99 x8 behind it), ask 101 x20.
HAND = [
    (0 * MS, ADD, BID, 99, 8, -1),
    (1 * MS, ADD, BID, 100, 10, -1),
    (2 * MS, ADD, ASK, 101, 20, -1),  # row 2: first two-sided row
    (500 * MS, ADD, BID, 100, 5, -1),  # row 3: bid queue 15
    (700 * MS, TRADE, ASK, 100, 15, 0),  # row 4: seller takes the whole bid level
    (900 * MS, ADD, ASK, 101, 5, -1),  # row 5: ask queue 25
]


@pytest.fixture
def hand(cfg: Settings) -> pl.DataFrame:
    return full_frame(cfg, HAND)


def test_imbalance_and_microprice(hand: pl.DataFrame) -> None:
    r = hand.row(3, named=True)
    assert r["imbalance"] == pytest.approx((15 - 20) / 35)
    expected_micro = (100 * 20 + 101 * 15) / 35 - 100.5
    assert r["microprice"] == pytest.approx(expected_micro)
    assert r["spread"] == 1 and r["wide_spread"] == 0
    # one-sided rows have no defined features
    assert np.isnan(hand["imbalance"][0])


def test_ofi_matches_cont_kukanov_stoikov(hand: pl.DataFrame) -> None:
    # Row 3: bid queue grows by 5 at the same price -> e = +5.
    # Row 4: best bid drops from 100 to 99 -> e = -(previous bid queue 15) = -15.
    # Row 5: ask queue grows by 5 at the same price -> e = -5.
    ofi10 = hand["ofi_ev10"].to_list()
    assert ofi10[3] == pytest.approx(5)
    assert ofi10[4] == pytest.approx(5 - 15)
    assert ofi10[5] == pytest.approx(5 - 15 - 5)
    # 100 ms window at row 5 (t = 900 ms) only contains row 5 itself.
    assert hand["ofi_ms100"][5] == pytest.approx(-5)


def test_trade_flow_and_depletion(hand: pl.DataFrame) -> None:
    assert hand["tfi_ms1000"][4] == pytest.approx(-1.0)  # only a sell
    assert hand["tfi_ms1000"][3] == pytest.approx(0.0)  # no trades yet
    # Row 3: bid queue at 100 went 10 -> 15 within the last second: negative depletion.
    assert hand["bid_depl_ms1000"][3] == pytest.approx((10 - 15) / 1.0)


def test_time_since_mid_change_and_targets(hand: pl.DataFrame) -> None:
    # The mid changes at row 4 (bid 100 -> 99): 100.5 -> 100.0
    assert hand["since_mid_change_ms"][5] == pytest.approx(200.0)
    assert hand["next_dir"][3] == 0.0  # next change is down
    assert hand["next_change_ms"][3] == pytest.approx(200.0)
    assert np.isnan(hand["next_dir"][5])  # no further change in the data


def test_forward_returns_use_last_state_at_or_before_horizon(cfg: Settings) -> None:
    ev = [
        (0, ADD, BID, 100, 10, -1),
        (0, ADD, ASK, 101, 10, -1),
        (int(0.5 * NS), ADD, ASK, 100 + 1, 1, -1),
        (int(1.5 * NS), TRADE, BID, 101, 11, 0),  # ask level gone; no asks left
        (int(1.6 * NS), ADD, ASK, 102, 5, -1),
        (int(40 * NS), ADD, BID, 100, 1, -1),
    ]
    f = full_frame(cfg, ev)
    # From row 1 (t=0, mid 100.5), 1 s later the mid is still 100.5; 5 s later it is 101.
    assert f["ret_1s"][1] == pytest.approx(0.0)
    assert f["ret_5s"][1] == pytest.approx(0.5)
    # 30 s horizon from t=40 s runs past the end of the data -> NaN
    assert np.isnan(f["ret_30s"][5])


def _synthetic_book(cfg: Settings) -> pl.DataFrame:
    day = generate_day(cfg, "ZN", 0, dt.date(2026, 8, 17), 7168)
    frame, _ = book_from_events(
        cfg, day.events.unique(subset=["sequence"], keep="first", maintain_order=True)
    )
    return frame.head(60_000)


def test_features_do_not_look_ahead(cfg: Settings) -> None:
    book = _synthetic_book(cfg)
    full = compute_features(book, cfg.features, cfg.book.depth_levels, 0)
    for cut in (1_000, 17_777, 45_000):
        part = compute_features(book.head(cut), cfg.features, cfg.book.depth_levels, 0)
        a = full.head(cut).to_numpy()
        b = part.to_numpy()
        assert np.array_equal(a, b, equal_nan=True), (
            f"features changed when data after row {cut} was removed"
        )


def test_targets_only_use_the_future(cfg: Settings) -> None:
    book = _synthetic_book(cfg)
    full = compute_targets(book, cfg.targets.horizons_s)
    for start in (5_000, 30_000):
        part = compute_targets(book.slice(start), cfg.targets.horizons_s)
        a = full.slice(start).to_numpy()
        b = part.to_numpy()
        assert np.array_equal(a, b, equal_nan=True), (
            f"targets changed when data before row {start} was removed"
        )
    # and they genuinely depend on the future: the next_dir of a row equals the
    # sign of the first later mid that differs
    mid = full["mid"].to_numpy()
    nd = full["next_dir"].to_numpy()
    i = 1_000
    j = i + 1 + int(np.argmax(mid[i + 1 :] != mid[i]))
    assert nd[i] == (1.0 if mid[j] > mid[i] else 0.0)


def test_streaming_features_equal_batch(cfg: Settings) -> None:
    book = _synthetic_book(cfg)
    batch = compute_features(book, cfg.features, cfg.book.depth_levels, 0)
    flags = pl.Series("book_flags", np.zeros(book.height, dtype=np.int32))

    class Src:
        instrument = "ZN"

        def batches(self):  # type: ignore[no-untyped-def]
            yield from book.with_columns(flags).iter_slices(7_000)

    class Rec:
        def __init__(self) -> None:
            self.f = StreamingFeatures(ofi_events=50, tfi_window_ms=1000)
            self.rows: list[tuple[float, float, float, float]] = []

        def on_trade(self, t):  # type: ignore[no-untyped-def]
            self.f.on_trade(t)

        def on_book_update(self, u):  # type: ignore[no-untyped-def]
            self.f.on_book_update(u)
            self.rows.append((self.f.imbalance, self.f.microprice, self.f.ofi, self.f.tfi))

    rec = Rec()
    ReplayEngine(Src(), [rec]).run()
    got = np.array(rec.rows)
    valid = ~np.isnan(batch["imbalance"].to_numpy())
    for k, name in enumerate(["imbalance", "microprice", "ofi_ev50", "tfi_ms1000"]):
        np.testing.assert_allclose(
            got[valid, k], batch[name].to_numpy()[valid], rtol=0, atol=1e-12, err_msg=name
        )


def test_cancel_only_book_has_no_trade_flow(cfg: Settings) -> None:
    ev = [(0, ADD, BID, 100, 10, -1), (0, ADD, ASK, 101, 10, -1), (MS, CANCEL, BID, 100, 3, 0)]
    f = book_frame(cfg, ev)
    feats = compute_features(f, cfg.features, cfg.book.depth_levels, 0)
    assert feats["tfi_ms1000"][2] == 0.0
