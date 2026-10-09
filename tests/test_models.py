"""Walk-forward splitter, baselines, calibration and metrics."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from halftick.models.direction import BaselineA, BaselineB, Calibrator
from halftick.models.evaluate import hac_mean_test, metrics
from halftick.models.walk_forward import Fold, assert_no_leakage, walk_forward

D = [dt.date(2026, 9, 1) + dt.timedelta(days=i) for i in range(12)]


def test_walk_forward_basic_layout() -> None:
    folds = list(walk_forward(D[:8], min_train_days=5))
    assert [len(f.train) for f in folds] == [5, 6]
    assert folds[0].calibrate == (D[5],) and folds[0].test == (D[6],)
    assert folds[1].calibrate == (D[6],) and folds[1].test == (D[7],)


def test_walk_forward_blocks_and_rolling_window() -> None:
    folds = list(walk_forward(D, min_train_days=4, test_days=3, expanding=False))
    assert all(len(f.train) == 4 for f in folds)
    assert folds[0].test == tuple(D[5:8])
    assert folds[-1].test == tuple(D[11:12])  # last block may be short


@given(
    st.integers(min_value=3, max_value=30),
    st.integers(min_value=1, max_value=6),
    st.integers(min_value=1, max_value=4),
    st.integers(min_value=1, max_value=3),
    st.booleans(),
)
def test_walk_forward_never_leaks(
    n: int, min_train: int, test_days: int, cal_days: int, expanding: bool
) -> None:
    days = [dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    seen_test: set[dt.date] = set()
    for fold in walk_forward(days, min_train, cal_days, test_days, expanding):
        assert_no_leakage(fold)
        assert not (set(fold.train) | set(fold.calibrate)) & set(fold.test)
        assert not seen_test & set(fold.test), "a day was tested twice"
        seen_test |= set(fold.test)


def test_leakage_check_catches_bad_folds() -> None:
    with pytest.raises(AssertionError):
        assert_no_leakage(Fold(train=(D[3],), calibrate=(D[2],), test=(D[4],)))
    with pytest.raises(ValueError):
        list(walk_forward([D[2], D[1]], 1))


def test_baseline_a_is_a_decile_frequency_table() -> None:
    rng = np.random.default_rng(0)
    imb = rng.uniform(-1, 1, 20_000)
    y = (rng.uniform(0, 1, imb.size) < 0.5 + 0.3 * imb).astype(float)
    X = pl.DataFrame({"imbalance": imb})
    m = BaselineA(bins=10, alpha=1.0).fit(X, y)
    p = m.raw(pl.DataFrame({"imbalance": [-0.95, 0.0, 0.95]}))
    assert p[0] < p[1] < p[2]
    assert p[2] == pytest.approx(0.5 + 0.3 * 0.9, abs=0.05)
    assert len(m.freq) == 10


def test_baseline_b_and_calibration() -> None:
    rng = np.random.default_rng(1)
    X = pl.DataFrame(
        {
            "imbalance": rng.normal(size=5000),
            "microprice": rng.normal(size=5000),
            "ofi_ev50": rng.normal(size=5000),
        }
    )
    y = (X["imbalance"].to_numpy() + 0.5 * rng.normal(size=5000) > 0).astype(float)
    b = BaselineB(["imbalance", "microprice", "ofi_ev50"]).fit(X, y)
    p = b.raw(X)
    assert metrics(y, p)["auc"] > 0.85
    # An over-confident score becomes honest after isotonic calibration.
    raw = np.clip(p * 1.6 - 0.3, 0.001, 0.999)
    cal = Calibrator("isotonic").fit(raw, y)
    assert metrics(y, cal.transform(raw))["brier"] < metrics(y, raw)["brier"]
    platt = Calibrator("platt").fit(raw, y)
    assert np.all((platt.transform(raw) > 0) & (platt.transform(raw) < 1))


def test_metrics_values() -> None:
    y = np.array([0, 0, 1, 1], dtype=float)
    p = np.array([0.1, 0.4, 0.35, 0.8])
    m = metrics(y, p)
    assert m["auc"] == pytest.approx(0.75)
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["brier"] == pytest.approx(np.mean((p - y) ** 2))


def test_hac_standard_error_accounts_for_autocorrelation() -> None:
    rng = np.random.default_rng(2)
    noise = np.convolve(rng.normal(size=20_000), np.ones(10) / 10, mode="same")  # autocorrelated
    x = 0.2 + noise
    mean, se, t = hac_mean_test(x, lags=30)
    naive_se = x.std(ddof=1) / np.sqrt(x.size)
    assert mean == pytest.approx(0.2, abs=0.05)
    assert se > 2 * naive_se  # naive errors would overstate significance
    assert t > 3
