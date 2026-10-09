"""Where does it work and where does it fail?

* direction-model accuracy by time of day, volatility tercile, queue size,
  spread state and release windows;
* execution savings (headline cell) by the same splits;
* signal decay: predictive power against prediction horizon;
* stability: per-day AUC and per-day savings;
* queue-model validation: each cancellation assumption against the true
  queue position (the role MBO data would play with real data);
* descriptive statistics of the book.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from halftick import plots
from halftick.config import Settings
from halftick.data.pipeline import analysis_days, load_features
from halftick.sim.summary import paired, trade_log_glob

BUCKET_ORDER = {
    "tod": ["open", "midday", "close"],
    "vol": ["low", "mid", "high"],
    "queue": ["small", "large"],
    "spread": ["1 tick", "wider"],
}


def _tod(minutes: pl.Expr, cfg: Settings) -> pl.Expr:
    b = cfg.session.time_of_day_buckets
    expr = pl.lit("outside")
    for name, (lo, hi) in reversed(list(b.items())):
        expr = pl.when((minutes >= lo) & (minutes < hi)).then(pl.lit(name)).otherwise(expr)
    return expr


def _add_buckets(
    df: pl.DataFrame, cfg: Settings, vol: str, queue: str, spread: str, minutes: str
) -> pl.DataFrame:
    v = df[vol].to_numpy()
    lo, hi = np.nanquantile(v, [1 / 3, 2 / 3])
    qmed = float(np.nanmedian(df[queue].to_numpy()))
    return df.with_columns(
        _tod(pl.col(minutes), cfg).alias("tod"),
        pl.when(pl.col(vol) <= lo)
        .then(pl.lit("low"))
        .when(pl.col(vol) <= hi)
        .then(pl.lit("mid"))
        .otherwise(pl.lit("high"))
        .alias("vol"),
        pl.when(pl.col(queue) <= qmed)
        .then(pl.lit("small"))
        .otherwise(pl.lit("large"))
        .alias("queue"),
        pl.when(pl.col(spread) <= 1)
        .then(pl.lit("1 tick"))
        .otherwise(pl.lit("wider"))
        .alias("spread_state"),
    )


def model_breakdown(cfg: Settings, instrument: str) -> pl.DataFrame:
    pred = pl.read_parquet(cfg.paths.tables / f"direction_predictions_{instrument}.parquet")
    pred = pred.with_columns((pl.col("bid_queue") + pl.col("ask_queue")).alias("total_queue"))
    pred = _add_buckets(pred, cfg, "vol_300s", "total_queue", "spread", "minutes_since_open")
    pred = pred.with_columns(
        pl.when(pl.col("in_release_window"))
        .then(pl.lit("release"))
        .otherwise(pl.lit("normal"))
        .alias("period")
    )
    rows = []
    for dim in ("tod", "vol", "queue", "spread_state", "period"):
        for key, g in pred.group_by(dim):
            val = key[0]
            if dim != "period":
                g = g.filter(~pl.col("in_release_window"))
            y = g["next_dir"].to_numpy()
            if len(np.unique(y)) < 2 or len(y) < 200:
                continue
            for m in ("baseline_a", "lightgbm"):
                p = g[f"p_{m}"].to_numpy()
                rows.append(
                    {
                        "dimension": dim,
                        "bucket": val,
                        "model": m,
                        "n": len(y),
                        "auc": roc_auc_score(y, p),
                        "accuracy": float(np.mean((p > 0.5) == (y > 0.5))),
                    }
                )
    out = pl.DataFrame(rows).sort(["dimension", "bucket", "model"])
    out.write_csv(cfg.paths.tables / f"diag_model_{instrument}.csv")
    return out


def execution_breakdown(cfg: Settings, instrument: str) -> pl.DataFrame:
    h = cfg.sim.headline
    wide = paired(cfg, instrument).filter(
        (pl.col("queue_model") == h.queue_model)
        & (pl.col("latency_ms") == h.latency_ms)
        & (pl.col("deadline_s") == h.deadline_s)
    )
    wide = _add_buckets(wide, cfg, "vol_300s", "own_queue", "spread_ticks", "minutes_since_open")
    rel = paired(cfg, instrument, release=True).filter(
        (pl.col("queue_model") == h.queue_model)
        & (pl.col("latency_ms") == h.latency_ms)
        & (pl.col("deadline_s") == h.deadline_s)
    )
    rows = []

    def emit(dim: str, val: str, g: pl.DataFrame) -> None:
        if g.height < 30:
            return
        rows.append(
            {
                "dimension": dim,
                "bucket": val,
                "n": g.height,
                "always_cross": g["always_cross"].mean(),
                "always_post": g["always_post"].mean(),
                "model_cost": g["model_cost"].mean(),
                "model_direction": g["model_direction"].mean(),
                "new_level": g["new_level"].mean(),
                "model_cost_saving_vs_cross": (g["always_cross"] - g["model_cost"]).mean(),
                "model_cost_saving_vs_post": (g["always_post"] - g["model_cost"]).mean(),
            }
        )

    for dim in ("tod", "vol", "queue", "spread_state"):
        for key, g in wide.group_by(dim):
            emit(dim, str(key[0]), g)
    emit("period", "normal", wide)
    if rel.height:
        emit("period", "release", rel)
    out = pl.DataFrame(rows).sort(["dimension", "bucket"])
    out.write_csv(cfg.paths.tables / f"diag_execution_{instrument}.csv")
    return out


def signal_decay(cfg: Settings, instrument: str) -> pl.DataFrame:
    pred = pl.read_parquet(cfg.paths.tables / f"direction_predictions_{instrument}.parquet").filter(
        ~pl.col("in_release_window")
    )
    rows = []
    horizons = [("next move", "next_dir")] + [
        (f"{h}s", f"ret_{h}s") for h in cfg.targets.horizons_s
    ]
    for label, col in horizons:
        g = pred.filter(pl.col(col).is_not_null() & pl.col(col).is_not_nan())
        y = g[col].to_numpy()
        if col != "next_dir":
            keep = y != 0
            g, y = g.filter(pl.Series(keep)), (y[keep] > 0).astype(float)
        for name, score in (
            ("imbalance", g["imbalance"].to_numpy()),
            ("lightgbm", g["p_lightgbm"].to_numpy()),
        ):
            rows.append(
                {"horizon": label, "signal": name, "n": len(y), "auc": roc_auc_score(y, score)}
            )
    out = pl.DataFrame(rows)
    out.write_csv(cfg.paths.tables / f"diag_signal_decay_{instrument}.csv")
    fig, ax = plots.plt.subplots(figsize=(6.5, 4))
    labels = [h[0] for h in horizons]
    for name in ("imbalance", "lightgbm"):
        v = [
            out.filter((pl.col("horizon") == lab) & (pl.col("signal") == name))["auc"][0]
            for lab in labels
        ]
        ax.plot(
            labels,
            v,
            marker="o",
            color=plots.color("baseline_a" if name == "imbalance" else "lightgbm"),
            label="queue imbalance alone" if name == "imbalance" else "LightGBM (calibrated)",
        )
    ax.axhline(0.5, color=plots.TEXT_2, lw=1, ls=(0, (4, 3)))
    ax.set_ylabel("AUC (direction of mid move)")
    ax.set_xlabel("prediction horizon")
    ax.set_title(f"{instrument}: signal decay, out of sample")
    ax.legend()
    plots.save(fig, cfg.paths.figures / f"signal_decay_{instrument}.png")
    return out


def stability(cfg: Settings, instrument: str) -> pl.DataFrame:
    folds = pl.read_csv(cfg.paths.tables / f"direction_folds_{instrument}.csv")
    h = cfg.sim.headline
    q = f"""
        SELECT date, strategy, avg(cost_ticks) AS mean_cost
        FROM read_parquet('{trade_log_glob(cfg, instrument)}')
        WHERE NOT in_release_window AND queue_model = '{h.queue_model}'
          AND latency_ms = {h.latency_ms} AND deadline_s = {h.deadline_s}
        GROUP BY ALL ORDER BY ALL
    """
    daily = duckdb.sql(q).pl()
    daily.write_csv(cfg.paths.tables / f"diag_daily_costs_{instrument}.csv")
    fig, axes = plots.plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    for m in ("baseline_a", "baseline_b", "lightgbm"):
        g = folds.filter(pl.col("model") == m)
        ax.plot(
            g["test_day"].to_list(),
            g["auc"].to_list(),
            marker="o",
            ms=4,
            color=plots.color(m),
            label=m.replace("_", " "),
        )
    ax.set_title(f"{instrument}: direction AUC by test day")
    ax.set_ylabel("AUC")
    ax.tick_params(axis="x", rotation=60)
    ax.legend()
    ax = axes[1]
    for s in ("always_cross", "always_post", "model_direction", "model_cost", "new_level"):
        g = daily.filter(pl.col("strategy") == s).sort("date")
        ax.plot(
            [str(d) for d in g["date"].to_list()],
            g["mean_cost"].to_list(),
            marker="o",
            ms=4,
            color=plots.color(s),
            label=s.replace("_", " "),
        )
    ax.set_title("Mean cost by test day (headline cell)")
    ax.set_ylabel("ticks per contract")
    ax.tick_params(axis="x", rotation=60)
    ax.legend(fontsize=8)
    fig.tight_layout()
    plots.save(fig, cfg.paths.figures / f"stability_{instrument}.png")
    return daily


def queue_model_validation(cfg: Settings, instrument: str) -> pl.DataFrame:
    """Compare each queue assumption with the true queue position on the same orders."""
    q = f"""
        WITH p AS (
            SELECT date, decision_id, latency_ms, deadline_s, queue_model, filled_passive,
                   cost_ticks, time_to_fill_ms
            FROM read_parquet('{trade_log_glob(cfg, instrument)}')
            WHERE strategy = 'always_post' AND NOT in_release_window
        ), t AS (SELECT * FROM p WHERE queue_model = 'exact')
        SELECT p.queue_model, p.latency_ms, p.deadline_s,
               avg(p.filled_passive::INT) AS fill_rate,
               avg(t.filled_passive::INT) AS true_fill_rate,
               avg((p.filled_passive = t.filled_passive)::INT) AS fill_agreement,
               avg(p.cost_ticks) - avg(t.cost_ticks) AS cost_bias_ticks,
               avg(CASE WHEN p.filled_passive AND t.filled_passive
                        THEN abs(p.time_to_fill_ms - t.time_to_fill_ms) END) AS mae_fill_time_ms
        FROM p JOIN t USING (date, decision_id, latency_ms, deadline_s)
        GROUP BY ALL ORDER BY p.queue_model, p.latency_ms, p.deadline_s
    """
    out = duckdb.sql(q).pl()
    out.write_csv(cfg.paths.tables / f"queue_validation_{instrument}.csv")
    return out


def descriptive_instrument(cfg: Settings, instrument: str) -> pl.DataFrame:
    """Per-day book statistics for one instrument, plus a queue-size sample for the figure.

    Saved per instrument so the processed days can be deleted afterwards
    (they are reproducible from the seed) without losing the statistics.
    """
    rows = []
    queues = []
    cols = ["in_session", "spread", "bid_queue", "ask_queue", "mid", "action", "ts_event"]
    for d in analysis_days(cfg, instrument):
        f = load_features(cfg, instrument, d, columns=cols).filter("in_session")
        mid = f["mid"].to_numpy()
        changes = int(np.sum(np.diff(mid) != 0))
        hours = (f["ts_event"][-1] - f["ts_event"][0]) / 3.6e12
        both = np.concatenate([f["bid_queue"].to_numpy(), f["ask_queue"].to_numpy()])
        rows.append(
            {
                "instrument": instrument,
                "date": d,
                "events": f.height,
                "events_per_s": f.height / (hours * 3600),
                "spread_1tick_share": float((f["spread"] == 1).mean()),
                "median_best_queue": float(np.median(both)),
                "mid_changes": changes,
                "mid_changes_per_min": changes / (hours * 60),
                "trades": int((f["action"] == 3).sum()),
            }
        )
        queues.append(f["bid_queue"].to_numpy()[::50])
    out = pl.DataFrame(rows)
    cfg.paths.tables.mkdir(parents=True, exist_ok=True)
    out.write_csv(cfg.paths.tables / f"descriptive_{instrument}.csv")
    if queues:
        pl.DataFrame({"bid_queue": np.concatenate(queues)}).write_parquet(
            cfg.paths.tables / f"queue_sample_{instrument}.parquet"
        )
    return out


def descriptive(cfg: Settings, instruments: list[str]) -> pl.DataFrame:
    """Combine the per-instrument statistics and draw the queue-size figure."""
    frames = []
    fig, ax = plots.plt.subplots(figsize=(7.5, 4))
    for inst in instruments:
        table = cfg.paths.tables / f"descriptive_{inst}.csv"
        if not table.exists() and analysis_days(cfg, inst):
            descriptive_instrument(cfg, inst)
        if not table.exists():
            continue
        frames.append(pl.read_csv(table))
        sample = cfg.paths.tables / f"queue_sample_{inst}.parquet"
        if sample.exists():
            qq = pl.read_parquet(sample)["bid_queue"].to_numpy()
            med = float(np.median(qq))
            ax.hist(
                qq / med,
                bins=np.linspace(0, 4, 60),
                histtype="step",
                lw=2,
                density=True,
                color=plots.color(inst),
                label=f"{inst} (median {med:.0f} lots)",
            )
    out = pl.concat(frames) if frames else pl.DataFrame()
    out.write_csv(cfg.paths.tables / "descriptive_days.csv")
    ax.set_xlabel("best-bid queue size / median")
    ax.set_ylabel("density")
    ax.set_title("Queue size distributions, day session")
    ax.legend()
    plots.save(fig, cfg.paths.figures / "queue_sizes.png")
    return out


def run_all(cfg: Settings, instrument: str) -> dict[str, Path]:
    model_breakdown(cfg, instrument)
    execution_breakdown(cfg, instrument)
    signal_decay(cfg, instrument)
    stability(cfg, instrument)
    queue_model_validation(cfg, instrument)
    descriptive_instrument(cfg, instrument)
    return {"tables": cfg.paths.tables, "figures": cfg.paths.figures}
