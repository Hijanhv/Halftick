"""Walk-forward evaluation of the direction models.

Outputs (under reports/):
* tables/direction_folds_<inst>.csv    per-fold, per-model metrics (out of sample)
* tables/direction_summary_<inst>.csv  pooled out-of-sample metrics, plus in-sample for contrast
* tables/direction_tests_<inst>.csv    HAC (Newey-West) tests of log-loss differences
* tables/direction_predictions_<inst>.parquet  test-day predictions for diagnostics
* figures/reliability_<inst>.png       reliability diagrams
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import polars as pl
import statsmodels.api as sm
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from halftick import plots
from halftick.config import Settings
from halftick.data.pipeline import analysis_days
from halftick.log import get_logger
from halftick.models.direction import MODEL_NAMES, fit_direction, model_features, sample_day
from halftick.models.walk_forward import assert_no_leakage, walk_forward

log = get_logger(__name__)
F64 = npt.NDArray[np.float64]

DIAG_COLUMNS = [
    "ts_event",
    "minutes_since_open",
    "vol_300s",
    "bid_queue",
    "ask_queue",
    "spread",
    "in_release_window",
    "next_dir",
    "ret_1s",
    "ret_5s",
    "ret_30s",
]


def metrics(y: F64, p: F64) -> dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "n": float(len(y)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "accuracy": float(np.mean((p > 0.5) == (y > 0.5))),
    }


def _columns(cfg: Settings) -> list[str]:
    return sorted(set(model_features(cfg) + DIAG_COLUMNS + ["in_session", "next_dir"]))


def hac_mean_test(diff: F64, lags: int) -> tuple[float, float, float]:
    """Mean of a loss difference with a Newey-West standard error. Returns (mean, se, t)."""
    res = sm.OLS(diff, np.ones_like(diff)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    # rounded: multi-threaded BLAS leaves ~1e-12 run-to-run noise
    return (
        round(float(res.params[0]), 9),
        round(float(res.bse[0]), 9),
        round(float(res.tvalues[0]), 6),
    )


def run_direction_walk_forward(cfg: Settings, instrument: str) -> pl.DataFrame:
    days = analysis_days(cfg, instrument)
    if len(days) < cfg.models.min_train_days + 2:
        raise RuntimeError(
            f"{instrument}: need at least {cfg.models.min_train_days + 2} days, have {len(days)}"
        )
    cols = _columns(cfg)
    n = cfg.models.sample_rows_per_day
    cache: dict = {}

    def get(d, include_release=False):  # type: ignore[no-untyped-def]
        key = (d, include_release)
        if key not in cache:
            cache[key] = sample_day(
                cfg, instrument, d, n, cfg.synthetic.seed + d.toordinal(), cols, include_release
            )
        return cache[key]

    fold_rows = []
    preds = []
    insample = []
    for k, fold in enumerate(walk_forward(days, cfg.models.min_train_days)):
        assert_no_leakage(fold)
        train = pl.concat([get(d) for d in fold.train])
        cal = pl.concat([get(d) for d in fold.calibrate])
        test = pl.concat([get(d, include_release=True) for d in fold.test])
        fitted = fit_direction(cfg, train, cal, seed=cfg.synthetic.seed + k)
        p_test = fitted.predict(test)
        p_train = fitted.predict(train)
        normal = ~test["in_release_window"].to_numpy()
        y = test["next_dir"].to_numpy()
        for name in MODEL_NAMES:
            m = metrics(y[normal], p_test[name][normal])
            fold_rows.append(
                {
                    "fold": k,
                    "test_day": str(fold.test[0]),
                    "train_days": len(fold.train),
                    "model": name,
                    **m,
                }
            )
            insample.append({"model": name, **metrics(train["next_dir"].to_numpy(), p_train[name])})
        preds.append(
            test.select(["date", *DIAG_COLUMNS, "imbalance"]).with_columns(
                pl.lit(k).alias("fold"), *[pl.Series(f"p_{nm}", p_test[nm]) for nm in MODEL_NAMES]
            )
        )
        log.info(
            "direction_fold_done",
            instrument=instrument,
            fold=k,
            test_day=str(fold.test[0]),
            **{
                f"auc_{nm}": round(r["auc"], 4)
                for nm, r in zip(MODEL_NAMES, fold_rows[-3:], strict=True)
            },
        )

    folds = pl.DataFrame(fold_rows)
    pred = pl.concat(preds)
    tables = cfg.paths.tables
    tables.mkdir(parents=True, exist_ok=True)
    folds.write_csv(tables / f"direction_folds_{instrument}.csv")
    pred.write_parquet(tables / f"direction_predictions_{instrument}.parquet")

    normal = pred.filter(~pl.col("in_release_window"))
    release = pred.filter(pl.col("in_release_window"))
    y = normal["next_dir"].to_numpy()
    summary = []
    ins = pl.DataFrame(insample).group_by("model").mean()
    for name in MODEL_NAMES:
        oos = metrics(y, normal[f"p_{name}"].to_numpy())
        summary.append({"model": name, "sample": "out_of_sample", **oos})
        if release.height > 100:
            summary.append(
                {
                    "model": name,
                    "sample": "release_windows_oos",
                    **metrics(release["next_dir"].to_numpy(), release[f"p_{name}"].to_numpy()),
                }
            )
        row = ins.filter(pl.col("model") == name).to_dicts()[0]
        summary.append(
            {
                "model": name,
                "sample": "in_sample",
                **{k: row[k] for k in ("n", "log_loss", "brier", "auc", "accuracy")},
            }
        )
    summary_df = pl.DataFrame(summary)
    summary_df.write_csv(tables / f"direction_summary_{instrument}.csv")

    # HAC tests of per-row log-loss differences, rows in time order within each day.
    ordered = normal.sort(["date", "ts_event"])
    yo = ordered["next_dir"].to_numpy()

    def row_loss(p: F64) -> F64:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        out: F64 = -(yo * np.log(p) + (1 - yo) * np.log(1 - p))
        return out

    loss = {nm: row_loss(ordered[f"p_{nm}"].to_numpy()) for nm in MODEL_NAMES}
    tests = []
    for a, b in (
        ("lightgbm", "baseline_a"),
        ("lightgbm", "baseline_b"),
        ("baseline_b", "baseline_a"),
    ):
        mean, se, t = hac_mean_test(loss[a] - loss[b], cfg.models.hac_lags)
        tests.append({"model": a, "vs": b, "mean_logloss_diff": mean, "hac_se": se, "t_stat": t})
    pl.DataFrame(tests).write_csv(tables / f"direction_tests_{instrument}.csv")

    plot_reliability(cfg, instrument, normal)
    return summary_df


def plot_reliability(cfg: Settings, instrument: str, pred: pl.DataFrame) -> None:
    fig, axes = plots.plt.subplots(
        1, 2, figsize=(10.5, 4.4), gridspec_kw={"width_ratios": [1.3, 1]}
    )
    ax, hx = axes
    y = pred["next_dir"].to_numpy()
    ax.plot([0, 1], [0, 1], color=plots.TEXT_2, lw=1, ls=(0, (4, 3)), label="perfect calibration")
    for name in MODEL_NAMES:
        p = pred[f"p_{name}"].to_numpy()
        frac, mean = calibration_curve(
            y, p, n_bins=cfg.models.reliability_bins, strategy="quantile"
        )
        ax.plot(mean, frac, marker="o", ms=4, color=plots.color(name), label=name.replace("_", " "))
        hx.hist(
            p,
            bins=40,
            histtype="step",
            lw=1.6,
            color=plots.color(name),
            label=name.replace("_", " "),
        )
    ax.set_xlabel("predicted P(next mid move is up)")
    ax.set_ylabel("observed frequency of up moves")
    ax.set_title(f"{instrument}: reliability, out of sample")
    ax.legend(loc="upper left")
    hx.set_xlabel("predicted probability")
    hx.set_ylabel("rows")
    hx.set_title("Prediction spread")
    hx.legend(loc="upper right")
    plots.save(fig, cfg.paths.figures / f"reliability_{instrument}.png")
