"""Next-mid-change direction models.

* Baseline A (Gould & Bonart): bin queue imbalance into deciles and predict
  the historical up-frequency in each bin. No parameters beyond the bins.
* Baseline B: logistic regression on imbalance, microprice and OFI.
* Main model: LightGBM on the full feature set.

Every model's raw output is then calibrated on a held-out day with isotonic
regression (or Platt scaling), because the simulator uses the probabilities
as numbers in an expected-cost formula, not just for ranking.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Protocol

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from halftick.config import Settings
from halftick.data.pipeline import load_features
from halftick.features.compute import layout

F64 = npt.NDArray[np.float64]
MODEL_NAMES = ("baseline_a", "baseline_b", "lightgbm")
EXCLUDED_FROM_MODEL = {"minutes_since_open"}  # diagnostics only


def model_features(cfg: Settings) -> list[str]:
    names = layout(cfg.features, cfg.book.depth_levels).names
    return [n for n in names if n not in EXCLUDED_FROM_MODEL]


class _Scorer(Protocol):
    def raw(self, X: pl.DataFrame) -> F64: ...


class BaselineA:
    """Imbalance-decile lookup table."""

    def __init__(self, bins: int, alpha: float) -> None:
        self.bins = bins
        self.alpha = alpha
        self.edges: F64 = np.array([])
        self.freq: F64 = np.array([])

    def fit(self, X: pl.DataFrame, y: F64) -> BaselineA:
        imb = X["imbalance"].to_numpy()
        self.edges = np.unique(np.quantile(imb, np.linspace(0, 1, self.bins + 1)[1:-1]))
        idx = np.searchsorted(self.edges, imb, side="right")
        k = len(self.edges) + 1
        ups = np.bincount(idx, weights=y, minlength=k)
        cnt = np.bincount(idx, minlength=k)
        self.freq = (ups + self.alpha) / (cnt + 2 * self.alpha)
        return self

    def raw(self, X: pl.DataFrame) -> F64:
        idx = np.searchsorted(self.edges, X["imbalance"].to_numpy(), side="right")
        out: F64 = self.freq[idx]
        return out


class BaselineB:
    def __init__(self, features: list[str]) -> None:
        self.features = features
        self.model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

    def fit(self, X: pl.DataFrame, y: F64) -> BaselineB:
        self.model.fit(X.select(self.features).to_numpy(), y)
        return self

    def raw(self, X: pl.DataFrame) -> F64:
        out: F64 = self.model.predict_proba(X.select(self.features).to_numpy())[:, 1]
        return out


class LightGBMModel:
    def __init__(self, features: list[str], params: dict[str, Any], seed: int) -> None:
        self.features = features
        self.model = lgb.LGBMClassifier(**params, random_state=seed)

    def fit(self, X: pl.DataFrame, y: F64) -> LightGBMModel:
        self.model.fit(X.select(self.features).to_pandas(), y)
        return self

    def raw(self, X: pl.DataFrame) -> F64:
        out: F64 = self.model.predict_proba(X.select(self.features).to_pandas())[:, 1]
        return out


class Calibrator:
    def __init__(self, method: str) -> None:
        self.method = method
        self._iso: IsotonicRegression | None = None
        self._platt: LogisticRegression | None = None

    def fit(self, scores: F64, y: F64) -> Calibrator:
        if self.method == "isotonic":
            self._iso = IsotonicRegression(y_min=1e-4, y_max=1 - 1e-4, out_of_bounds="clip").fit(
                scores, y
            )
        else:
            self._platt = LogisticRegression().fit(_logit(scores)[:, None], y)
        return self

    def transform(self, scores: F64) -> F64:
        if self._iso is not None:
            out: F64 = self._iso.predict(scores)
            return out
        assert self._platt is not None
        p: F64 = self._platt.predict_proba(_logit(scores)[:, None])[:, 1]
        return p


def _logit(p: F64) -> F64:
    q = np.clip(p, 1e-6, 1 - 1e-6)
    out: F64 = np.log(q / (1 - q))
    return out


@dataclass
class FittedDirection:
    features: list[str]
    models: dict[str, _Scorer] = field(default_factory=dict)
    calibrators: dict[str, Calibrator] = field(default_factory=dict)

    def predict(self, X: pl.DataFrame, calibrated: bool = True) -> dict[str, F64]:
        """Probabilities for every row. Rows with a missing feature (e.g. before
        both sides of the book exist) get a neutral 0.5 instead of an error."""
        complete = ~np.isnan(X.select(self.features).to_numpy()).any(axis=1)
        Xc = X.filter(pl.Series(complete)) if not complete.all() else X
        out = {}
        for name, m in self.models.items():
            s = m.raw(Xc)
            p = self.calibrators[name].transform(s) if calibrated else s
            full = np.full(X.height, 0.5)
            full[complete] = p
            out[name] = full
        return out


def modelling_rows(frame: pl.DataFrame, include_release: bool = False) -> pl.DataFrame:
    """Rows usable for the direction target: in session, valid book, known target."""
    cond = pl.col("in_session") & pl.col("next_dir").is_not_null() & pl.col("next_dir").is_not_nan()
    cond = cond & pl.col("imbalance").is_not_nan()
    if not include_release:
        cond = cond & ~pl.col("in_release_window")
    return frame.filter(cond)


def sample_day(
    cfg: Settings,
    instrument: str,
    date: dt.date,
    n: int,
    seed: int,
    columns: list[str],
    include_release: bool = False,
) -> pl.DataFrame:
    frame = load_features(cfg, instrument, date, columns=columns)
    rows = modelling_rows(frame, include_release)
    if rows.height > n:
        rows = rows.sample(n=n, seed=seed).sort("ts_event")
    return rows.with_columns(pl.lit(date).alias("date"))


def fit_direction(
    cfg: Settings, train: pl.DataFrame, cal: pl.DataFrame, seed: int
) -> FittedDirection:
    feats = model_features(cfg)
    m = cfg.models
    y_tr = train["next_dir"].to_numpy()
    y_cal = cal["next_dir"].to_numpy()
    fitted = FittedDirection(features=feats)
    fitted.models["baseline_a"] = BaselineA(m.imbalance_bins, m.laplace_alpha).fit(train, y_tr)
    fitted.models["baseline_b"] = BaselineB(m.logistic_features).fit(train, y_tr)
    fitted.models["lightgbm"] = LightGBMModel(feats, m.lightgbm, seed).fit(train, y_tr)
    for name, model in fitted.models.items():
        fitted.calibrators[name] = Calibrator(m.calibration).fit(model.raw(cal), y_cal)
    return fitted
