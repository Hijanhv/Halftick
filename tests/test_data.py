"""Sessions (DST), calendar, roll detection, quality checks, synthetic data,
the per-day pipeline, config validation and the Databento guards (no network)."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pydantic
import pytest

from halftick.config import Settings, load_config
from halftick.data import databento_io as dbio
from halftick.data import quality
from halftick.data.pipeline import analysis_days, available_days, build_day, load_features
from halftick.data.roll import detect_roll_days, roll_dates
from halftick.data.sessions import (
    load_releases,
    local_to_utc_ns,
    release_window_mask,
    session_bounds_ns,
)
from halftick.data.synthetic import generate_dataset, generate_day, trading_days
from halftick.schema import ADD, ASK, BID, MODIFY, NO_BID, TRADE
from tests.conftest import NS, ROOT, events_frame


def test_session_bounds_follow_daylight_saving(cfg: Settings) -> None:
    summer = session_bounds_ns(cfg, dt.date(2026, 7, 15))
    winter = session_bounds_ns(cfg, dt.date(2026, 12, 15))
    as_utc = lambda ns: dt.datetime.fromtimestamp(ns / NS, dt.UTC).strftime("%H:%M")  # noqa: E731
    assert (as_utc(summer[0]), as_utc(summer[1])) == ("12:30", "19:00")  # EDT, UTC-4
    assert (as_utc(winter[0]), as_utc(winter[1])) == ("13:30", "20:00")  # EST, UTC-5
    # the Monday after the November change already uses the winter offset
    assert as_utc(local_to_utc_ns(dt.date(2026, 11, 2), "08:30", "America/New_York")) == "13:30"


def test_release_window_mask() -> None:
    ts = np.arange(0, 20 * 60 * NS, 30 * NS, dtype=np.int64)
    mask = release_window_mask(ts, np.array([10 * 60 * NS]), 5)
    inside = ts[mask]
    assert inside.min() == 5 * 60 * NS and inside.max() < 15 * 60 * NS


def test_releases_csv_round_trip(cfg: Settings, tmp_path: Path) -> None:
    p = tmp_path / "rel.csv"
    p.write_text("# date,time_et,event\ndate,time_et,event\n2026-09-11,08:30,CPI\n")
    df = load_releases(cfg, p)
    assert df["ts_ns"][0] == local_to_utc_ns(dt.date(2026, 9, 11), "08:30", "America/New_York")


def test_roll_detection_from_volume() -> None:
    days = [dt.date(2026, 8, d) for d in (17, 18, 19, 20, 21, 24, 25)]
    old = [97, 96, 70, 45, 20, 3, 3]
    rows = []
    for d, v in zip(days, old, strict=True):
        rows += [
            {"date": d, "instrument_id": 1, "volume": v},
            {"date": d, "instrument_id": 2, "volume": 100 - v},
        ]
    table = detect_roll_days(pl.DataFrame(rows), 0.85, 1)
    assert roll_dates(table) == {dt.date(2026, 8, 19), dt.date(2026, 8, 20), dt.date(2026, 8, 21)}


def test_quality_counts_and_repairs() -> None:
    ev = events_frame(
        [
            (0, ADD, BID, 100, 5, -1),
            (1, ADD, ASK, 101, 5, -1),
            (2, ADD, BID, 100, 0, -1),  # zero size
        ]
    )
    dup = pl.concat([ev, ev.head(1)])
    dup = dup.with_columns(pl.Series("sequence", [1, 2, 5, 1]))  # gap 2 -> 5 and a duplicate
    rep = quality.QualityReport("ZN", "2026-01-01", "test")
    out = quality.clean_events(dup, rep)
    assert rep.duplicates_removed == 1
    assert rep.sequence_gaps == 1 and rep.missing_sequence_numbers == 2
    assert rep.zero_or_negative_sizes == 1
    assert out["sequence"].to_list() == [1, 2, 5]


def test_quality_enforce_fails_loudly_on_crossed_books() -> None:
    rep = quality.QualityReport("ZN", "2026-01-01", "test", n_clean=100, crossed_book_events=5)
    with pytest.raises(quality.DataQualityError):
        quality.enforce(rep)
    ok = quality.QualityReport("ZN", "2026-01-01", "test", n_clean=100)
    quality.enforce(ok)


def test_session_gap_check() -> None:
    rep = quality.QualityReport("ZN", "2026-01-01", "test")
    ts = np.array([0, 1, 2, 6, 7], dtype=np.int64) * NS
    quality.check_session_gaps(ts, np.ones(5, bool), rep, 2.0)
    assert rep.session_gaps_over_threshold == 1 and rep.longest_session_gap_s == 4.0


def test_trading_days_skip_weekends_and_holidays() -> None:
    days = trading_days("2026-09-03", 4, ["2026-09-07"])
    assert days == [
        dt.date(2026, 9, 3),
        dt.date(2026, 9, 4),
        dt.date(2026, 9, 8),
        dt.date(2026, 9, 9),
    ]


def test_synthetic_day_is_deterministic_and_large_tick(cfg: Settings) -> None:
    a = generate_day(cfg, "ZN", 0, dt.date(2026, 8, 17), 7168)
    b = generate_day(cfg, "ZN", 0, dt.date(2026, 8, 17), 7168)
    assert a.events.equals(b.events)
    assert a.events.height > 10_000
    assert set(a.events["action"].unique().to_list()) <= {0, 1, 3}
    trades = a.events.filter(pl.col("action") == TRADE)
    assert trades["queue_pos"].max() == 0  # trades always take the front of the queue


def test_pipeline_end_to_end(cfg: Settings) -> None:
    generate_dataset(cfg, "ZN", n_days=2)
    days = available_days(cfg, "ZN")
    assert len(days) == 2
    for d in days:
        build_day(cfg, "ZN", d)
    f = load_features(cfg, "ZN", days[0])
    for col in (
        "imbalance",
        "ofi_ev50",
        "next_dir",
        "ret_5s",
        "in_session",
        "in_release_window",
        "book_flags",
    ):
        assert col in f.columns
    sess = f.filter("in_session")
    assert sess.height > 0
    assert (sess["spread"] >= 1).all()
    assert (sess["book_flags"] & 4).sum() == 0  # no crossed books
    rep = json.loads((cfg.paths.data_quality / f"ZN_{days[0].isoformat()}.json").read_text())
    assert rep["n_clean"] == f.height
    # no roll in the first two days, so both are analysis days
    assert analysis_days(cfg, "ZN") == days


def test_config_rejects_bad_values(tmp_path: Path) -> None:
    with pytest.raises(pydantic.ValidationError):
        load_config(ROOT / "config.yaml", ["fees.per_side_usd=-1"])
    with pytest.raises(pydantic.ValidationError):
        load_config(ROOT / "config.yaml", ["sim.queue_models=[telepathic]"])
    with pytest.raises(pydantic.ValidationError):
        load_config(ROOT / "config.yaml", ["unknown_key=1"])
    with pytest.raises(ValueError, match="key=value"):
        load_config(ROOT / "config.yaml", ["no_equals_sign"])


def test_config_tick_values() -> None:
    c = load_config(ROOT / "config.yaml")
    assert c.instrument("ZN").tick_size == 1 / 64
    assert c.instrument("ZN").tick_value_usd == pytest.approx(1000 / 64)
    assert c.instrument("ES").tick_value_usd == pytest.approx(0.25 * 50)


# --- Databento guards, exercised with a fake client ------------------------------------


class FakeClient:
    def __init__(self, cost: float) -> None:
        self.cost = cost
        self.downloads: list[str] = []
        self.metadata = SimpleNamespace(get_cost=lambda **kw: self.cost)
        self.timeseries = SimpleNamespace(get_range=self._get_range)

    def _get_range(self, **kw: object) -> None:
        self.downloads.append(str(kw["path"]))
        Path(str(kw["path"])).write_bytes(b"dbn")


def test_downloader_dry_run_buys_nothing(cfg: Settings) -> None:
    req = dbio.make_request(cfg, "ZN", "mbp-1", dt.date(2026, 9, 1))
    client = FakeClient(1.25)
    assert dbio.download(cfg, req, confirm=False, client=client) is None
    assert client.downloads == []
    assert dbio.SpendLedger(cfg.databento.spend_ledger, 100).spent == 0


def test_downloader_caches_and_records_spend(cfg: Settings) -> None:
    req = dbio.make_request(cfg, "ZN", "mbp-1", dt.date(2026, 9, 1))
    client = FakeClient(1.25)
    path = dbio.download(cfg, req, confirm=True, client=client)
    assert path is not None and path.exists()
    assert dbio.SpendLedger(cfg.databento.spend_ledger, 100).spent == pytest.approx(1.25)
    again = dbio.download(cfg, req, confirm=True, client=client)
    assert again == path and len(client.downloads) == 1  # cache hit, no second purchase


def test_budget_guard_refuses_overspend(cfg: Settings) -> None:
    req = dbio.make_request(cfg, "ZN", "mbo", dt.date(2026, 9, 1))
    with pytest.raises(dbio.BudgetExceededError):
        dbio.download(cfg, req, confirm=True, client=FakeClient(cfg.databento.budget_usd + 0.01))


def test_missing_api_key_is_a_clear_error(cfg: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(cfg.databento.api_key_env, raising=False)
    monkeypatch.setattr(dbio, "load_dotenv", lambda: None)
    req = dbio.make_request(cfg, "ZN", "mbp-1", dt.date(2026, 9, 2))
    with pytest.raises(dbio.MissingApiKeyError):
        dbio.download(cfg, req, confirm=False)


def test_mbp_records_convert_to_book_frame() -> None:
    tick = 1 / 64
    px = lambda p: round(p / 1e-9)  # noqa: E731
    dtype = [
        ("ts_event", "u8"),
        ("sequence", "u4"),
        ("instrument_id", "u4"),
        ("action", "S1"),
        ("side", "S1"),
        ("price", "i8"),
        ("size", "u4"),
    ]
    for lvl in range(2):
        dtype += [
            (f"bid_px_{lvl:02d}", "i8"),
            (f"ask_px_{lvl:02d}", "i8"),
            (f"bid_sz_{lvl:02d}", "u4"),
            (f"ask_sz_{lvl:02d}", "u4"),
        ]
    rec = np.zeros(2, dtype=dtype)
    rec["ts_event"] = [10, 20]
    rec["sequence"] = [1, 2]
    rec["action"] = [b"A", b"T"]
    rec["side"] = [b"B", b"A"]
    rec["price"] = [px(112.0), px(112.0)]
    rec["size"] = [5, 3]
    rec["bid_px_00"] = px(112.0)
    rec["ask_px_00"] = [px(112.0 + tick), dbio.UNDEF_PRICE]
    rec["bid_sz_00"] = [500, 497]
    rec["ask_sz_00"] = [400, 0]
    rec["bid_px_01"] = px(112.0 - 2 * tick)  # two ticks below: grid column 2
    rec["bid_sz_01"] = 900
    rec["ask_px_01"] = dbio.UNDEF_PRICE
    f = dbio.mbp_to_book_frame(rec, tick, 3)
    assert f["bid_px"].to_list() == [7168, 7168]
    assert f["ask_px"][0] == 7169
    assert f["action"].to_list() == [ADD, TRADE]
    assert f["side"].to_list() == [BID, ASK]
    assert f["bid_sz_0"].to_list() == [500, 497] and f["bid_sz_2"].to_list() == [900, 900]
    assert f["bid_sz_1"].to_list() == [0, 0]
    assert (f["queue_pos"] == -1).all()
    assert dbio.to_ticks(np.array([dbio.UNDEF_PRICE]), tick, NO_BID)[0] == NO_BID
    assert dbio.ACTION_MAP["N"] == MODIFY
