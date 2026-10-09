"""Summaries of the trade logs: cost tables, savings with uncertainty, figures.

Savings are paired: every strategy faced the same decisions, so we difference
costs per decision before averaging. Uncertainty comes from two angles:

* a day-block bootstrap (resample whole test days), which respects the fact
  that decisions within a day share market conditions;
* a Newey-West (HAC) t-statistic on the time-ordered per-decision savings.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import statsmodels.api as sm

from halftick import plots
from halftick.config import Settings
from halftick.sim.policy import STRATEGIES


def query(sql: str) -> pl.DataFrame:
    """Run a DuckDB query single-threaded.

    Parallel aggregation adds floats in a varying order, so the last digit of a
    mean can change between runs. One thread makes every committed table
    reproducible bit for bit; on these data sizes it costs a second or two.
    """
    with duckdb.connect(config={"threads": 1}) as con:
        return con.sql(sql).pl()


COMPARE = ("model_cost", "model_direction", "always_post", "new_level", "oracle")


def trade_log_glob(cfg: Settings, instrument: str) -> str:
    return str(cfg.paths.trade_logs / instrument / "*.parquet")


def cost_table(cfg: Settings, instrument: str) -> pl.DataFrame:
    q = f"""
        SELECT queue_model, latency_ms, deadline_s, strategy,
               count(*) AS n,
               avg(cost_ticks) AS mean_cost_ticks,
               stddev_samp(cost_ticks) AS sd_cost_ticks,
               quantile_cont(cost_ticks, 0.1) AS p10_cost_ticks,
               median(cost_ticks) AS median_cost_ticks,
               quantile_cont(cost_ticks, 0.9) AS p90_cost_ticks,
               avg(cost_usd) AS mean_cost_usd,
               avg(CASE WHEN filled_passive THEN 1.0 ELSE 0.0 END) AS passive_fill_rate,
               avg(CASE WHEN filled_passive THEN time_to_fill_ms END) AS mean_ms_to_passive_fill,
               avg(time_to_fill_ms) AS mean_ms_to_done,
               avg(adverse_1s) AS adverse_1s_ticks,
               avg(adverse_5s) AS adverse_5s_ticks
        FROM read_parquet('{trade_log_glob(cfg, instrument)}')
        WHERE NOT in_release_window
        GROUP BY ALL ORDER BY ALL
    """
    return query(q)


def paired(cfg: Settings, instrument: str, release: bool = False) -> pl.DataFrame:
    """One row per decision and cell, one cost column per strategy."""
    cond = "in_release_window" if release else "NOT in_release_window"
    q = f"""
        PIVOT (SELECT date, decision_id, decision_ts, queue_model, latency_ms, deadline_s,
                      strategy, cost_ticks,
                      minutes_since_open, vol_300s, own_queue, spread_ticks
               FROM read_parquet('{trade_log_glob(cfg, instrument)}') WHERE {cond})
        ON strategy USING first(cost_ticks)
        GROUP BY date, decision_id, decision_ts, queue_model, latency_ms, deadline_s,
                 minutes_since_open, vol_300s, own_queue, spread_ticks
    """
    return query(q).sort(["queue_model", "latency_ms", "deadline_s", "decision_ts"])


def day_block_bootstrap(
    diff: np.ndarray, day: np.ndarray, reps: int, seed: int
) -> tuple[float, float]:
    days = np.unique(day)
    sums = np.array([diff[day == d].sum() for d in days])
    counts = np.array([(day == d).sum() for d in days])
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(days), size=(reps, len(days)))
    means = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def hac_t(diff: np.ndarray, lags: int) -> float:
    if len(diff) < 3 or np.allclose(diff, diff[0]):
        return float("nan")
    res = sm.OLS(diff, np.ones_like(diff)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    # Multi-threaded linear algebra leaves ~1e-12 noise; 6 decimals is plenty
    # for a t-statistic and keeps committed tables identical between runs.
    return round(float(res.tvalues[0]), 6)


def savings_table(cfg: Settings, instrument: str, wide: pl.DataFrame | None = None) -> pl.DataFrame:
    wide = wide if wide is not None else paired(cfg, instrument)
    out = []
    for (qm, lat, dl), g in wide.group_by(
        ["queue_model", "latency_ms", "deadline_s"], maintain_order=True
    ):
        day = g["date"].to_numpy()
        for strat in COMPARE:
            for base in ("always_cross", "always_post"):
                if strat == base:
                    continue
                diff = (g[base] - g[strat]).to_numpy()  # positive = strat is cheaper
                lo, hi = day_block_bootstrap(diff, day, cfg.sim.bootstrap_reps, cfg.synthetic.seed)
                out.append(
                    {
                        "queue_model": qm,
                        "latency_ms": lat,
                        "deadline_s": dl,
                        "strategy": strat,
                        "vs": base,
                        "n": len(diff),
                        "test_days": len(np.unique(day)),
                        "saving_ticks": float(diff.mean()),
                        "ci95_lo": lo,
                        "ci95_hi": hi,
                        "hac_t": hac_t(diff, cfg.models.hac_lags),
                        "saving_usd": float(diff.mean())
                        * cfg.instrument(instrument).tick_value_usd,
                    }
                )
    return pl.DataFrame(out)


def fee_sensitivity(cfg: Settings, instrument: str, costs: pl.DataFrame) -> pl.DataFrame:
    """Mean cost in USD at each fee level for the headline cell.

    Every strategy fills exactly one contract per decision and pays exactly one
    fee, so the savings column is identical across fees. The table shows that.
    """
    h = cfg.sim.headline
    tick = cfg.instrument(instrument).tick_value_usd
    cell = costs.filter(
        (pl.col("queue_model") == h.queue_model)
        & (pl.col("latency_ms") == h.latency_ms)
        & (pl.col("deadline_s") == h.deadline_s)
    )
    base = cell.filter(pl.col("strategy") == "always_cross")["mean_cost_ticks"][0]
    rows = []
    for fee in cfg.fees.sensitivity_usd:
        for r in cell.iter_rows(named=True):
            rows.append(
                {
                    "fee_per_side_usd": fee,
                    "strategy": r["strategy"],
                    "mean_cost_usd": r["mean_cost_ticks"] * tick + fee,
                    "saving_vs_always_cross_usd": (base - r["mean_cost_ticks"]) * tick,
                }
            )
    return pl.DataFrame(rows)


def write_tables(cfg: Settings, instrument: str) -> dict[str, pl.DataFrame]:
    tables = cfg.paths.tables
    tables.mkdir(parents=True, exist_ok=True)
    costs = cost_table(cfg, instrument)
    wide = paired(cfg, instrument)
    savings = savings_table(cfg, instrument, wide)
    fees = fee_sensitivity(cfg, instrument, costs)
    costs.write_csv(tables / f"sim_costs_{instrument}.csv")
    savings.write_csv(tables / f"sim_savings_{instrument}.csv")
    fees.write_csv(tables / f"sim_fee_sensitivity_{instrument}.csv")
    rel = paired(cfg, instrument, release=True)
    if rel.height:
        savings_table(cfg, instrument, rel).write_csv(
            tables / f"sim_savings_release_{instrument}.csv"
        )
    plot_cost_distribution(cfg, instrument, wide)
    plot_sensitivity(cfg, instrument, savings)
    return {"costs": costs, "savings": savings, "fees": fees}


def _headline(cfg: Settings, df: pl.DataFrame) -> pl.DataFrame:
    h = cfg.sim.headline
    return df.filter(
        (pl.col("queue_model") == h.queue_model)
        & (pl.col("latency_ms") == h.latency_ms)
        & (pl.col("deadline_s") == h.deadline_s)
    )


def plot_cost_distribution(cfg: Settings, instrument: str, wide: pl.DataFrame) -> Path:
    h = cfg.sim.headline
    g = _headline(cfg, wide)
    fig, ax = plots.plt.subplots(figsize=(8.5, 4.6))
    order = [s for s in STRATEGIES if s in g.columns]
    for s in order:
        v = np.sort(g[s].to_numpy())
        ax.step(
            v,
            np.arange(1, len(v) + 1) / len(v),
            where="post",
            color=plots.color(s),
            lw=1.4 if s == "oracle" else 2.0,
            ls=(0, (3, 2)) if s == "oracle" else "-",
            label=f"{s.replace('_', ' ')}  (mean {v.mean():+.3f})"
            + ("  [upper bound]" if s == "oracle" else ""),
        )
    ax.set_xlabel("cost per contract, ticks vs arrival mid (lower is better)")
    ax.set_ylabel("share of orders at or below")
    ax.set_title(
        f"{instrument}: cost distribution, {h.queue_model} queue, "
        f"{h.latency_ms} ms, {h.deadline_s} s deadline"
    )
    ax.legend(loc="lower right")
    return plots.save(fig, cfg.paths.figures / f"cost_distribution_{instrument}.png")


def plot_sensitivity(
    cfg: Settings, instrument: str, savings: pl.DataFrame, strategy: str = "model_cost"
) -> Path:
    qms = cfg.sim.queue_models
    lats = cfg.sim.latencies_ms
    dls = cfg.sim.deadlines_s
    fig, axes = plots.plt.subplots(
        2, len(qms), figsize=(3.2 * len(qms), 5.8), sharex=True, sharey=True
    )
    for row, base in enumerate(("always_cross", "always_post")):
        sub = savings.filter((pl.col("strategy") == strategy) & (pl.col("vs") == base))
        vmax = float(np.nanmax(np.abs(sub["saving_ticks"].to_numpy()))) or 1.0
        for col, qm in enumerate(qms):
            ax = axes[row, col]
            m = np.full((len(dls), len(lats)), np.nan)
            for r in sub.filter(pl.col("queue_model") == qm).iter_rows(named=True):
                m[dls.index(r["deadline_s"]), lats.index(r["latency_ms"])] = r["saving_ticks"]
            ax.imshow(
                m,
                cmap=plots.DIVERGING.reversed(),
                vmin=-vmax,
                vmax=vmax,
                aspect="auto",
                origin="lower",
            )
            for i in range(len(dls)):
                for j in range(len(lats)):
                    if not np.isnan(m[i, j]):
                        ax.text(
                            j,
                            i,
                            f"{m[i, j]:+.3f}",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color=plots.TEXT,
                        )
            ax.grid(False)
            ax.set_xticks(range(len(lats)), [str(x) for x in lats])
            ax.set_yticks(range(len(dls)), [f"{x}s" for x in dls])
            if row == 0:
                ax.set_title(qm, fontsize=10)
            if row == 1:
                ax.set_xlabel("latency (ms)")
            if col == 0:
                ax.set_ylabel(f"vs {base.replace('_', ' ')}\ndeadline")
    fig.suptitle(
        f"{instrument}: {strategy.replace('_', ' ')} saving in ticks per contract (blue = cheaper)",
        fontsize=12,
    )
    fig.tight_layout()
    return plots.save(fig, cfg.paths.figures / f"sensitivity_{instrument}.png")
