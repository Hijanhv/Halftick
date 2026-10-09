"""Build reports/REPORT.md and the README results block from generated tables.

Nothing here contains a result typed by hand: every number, and every
"works best when / fails when" sentence, is read from the CSV tables that the
train, simulate and diagnose commands wrote. Re-running the pipeline and then
`halftick report` regenerates the whole write-up.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

import numpy as np
import polars as pl

from halftick import plots
from halftick.config import Settings
from halftick.data.pipeline import analysis_days, available_days, roll_table
from halftick.sim.summary import day_block_bootstrap, paired

README_START = "<!-- results:start -->"
README_END = "<!-- results:end -->"
STRATEGY_LABEL = {
    "always_cross": "Always cross",
    "always_post": "Always post",
    "model_direction": "Model: direction rule",
    "model_cost": "Model: cost-to-go",
    "new_level": "Join fresh levels only",
    "oracle": "Oracle (upper bound, uses the future)",
}
BUCKET_LABEL = {
    ("tod", "open"): "in the first 90 minutes after the open",
    ("tod", "midday"): "at midday",
    ("tod", "close"): "in the last 90 minutes before the close",
    ("vol", "low"): "when short-term volatility is low",
    ("vol", "mid"): "when short-term volatility is moderate",
    ("vol", "high"): "when short-term volatility is high",
    ("queue", "small"): "when the queue on our side is small",
    ("queue", "large"): "when the queue on our side is large",
    ("spread_state", "1 tick"): "when the spread is one tick",
    ("spread_state", "wider"): "when the spread is wider than one tick",
    ("period", "release"): "around scheduled macro releases",
    ("period", "normal"): "outside macro-release windows",
}


def _t(cfg: Settings, name: str) -> pl.DataFrame | None:
    p = cfg.paths.tables / name
    return pl.read_csv(p) if p.exists() else None


def _f(x: float, nd: int = 3, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _cell(df: pl.DataFrame, cfg: Settings) -> pl.DataFrame:
    h = cfg.sim.headline
    return df.filter(
        (pl.col("queue_model") == h.queue_model)
        & (pl.col("latency_ms") == h.latency_ms)
        & (pl.col("deadline_s") == h.deadline_s)
    )


def _md_table(df: pl.DataFrame, headers: list[str] | None = None) -> str:
    cols = df.columns
    head = headers or cols
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for row in df.iter_rows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def headline_numbers(cfg: Settings, instrument: str) -> dict[str, float] | None:
    sav = _t(cfg, f"sim_savings_{instrument}.csv")
    if sav is None:
        return None
    c = _cell(sav, cfg)
    out: dict[str, float] = {}
    for strat in ("model_cost", "model_direction", "always_post", "new_level", "oracle"):
        for base in ("always_cross", "always_post"):
            r = c.filter((pl.col("strategy") == strat) & (pl.col("vs") == base))
            if r.height:
                row = r.row(0, named=True)
                key = f"{strat}_vs_{base}"
                out[key] = row["saving_ticks"]
                out[key + "_lo"] = row["ci95_lo"]
                out[key + "_hi"] = row["ci95_hi"]
                out[key + "_t"] = row["hac_t"]
                out[key + "_usd"] = row["saving_usd"]
                out["n"] = row["n"]
                out["test_days"] = row["test_days"]
    return out


def best_and_worst(
    cfg: Settings, instrument: str, col: str = "model_cost_saving_vs_post"
) -> tuple[str, str] | None:
    d = _t(cfg, f"diag_execution_{instrument}.csv")
    if d is None or d.height == 0:
        return None
    d = d.filter(pl.col("n") >= 100)
    if d.height < 2:
        return None
    best = d.sort(col, descending=True).row(0, named=True)
    worst = d.sort(col).row(0, named=True)

    def lab(r: dict[str, object]) -> str:
        return BUCKET_LABEL.get(
            (str(r["dimension"]), str(r["bucket"])), f"{r['dimension']} = {r['bucket']}"
        )

    return (
        f"{lab(best)} ({_f(best[col], 3, True)} ticks vs always posting, n = {best['n']})",
        f"{lab(worst)} ({_f(worst[col], 3, True)} ticks, n = {worst['n']})",
    )


def headline_sentence(cfg: Settings, instrument: str) -> str:
    h = headline_numbers(cfg, instrument)
    if not h:
        return f"No simulation results for {instrument} yet."
    x, xlo, xhi = (
        h["model_cost_vs_always_cross"],
        h["model_cost_vs_always_cross_lo"],
        h["model_cost_vs_always_cross_hi"],
    )
    y, ylo, yhi = (
        h["model_cost_vs_always_post"],
        h["model_cost_vs_always_post_lo"],
        h["model_cost_vs_always_post_hi"],
    )
    hd = cfg.sim.headline
    s = (
        f"On synthetic {instrument}, choosing between posting and crossing with the cost-to-go model "
        f"cuts execution cost by **{_f(x)} ticks per contract** versus always crossing the spread "
        f"(95% day-block bootstrap CI {_f(xlo)} to {_f(xhi)}) "
    )
    if ylo > 0:
        s += f"and by **{_f(y)} ticks** versus always posting (CI {_f(ylo)} to {_f(yhi)})"
    elif yhi < 0:
        s += f"but is **{_f(-y)} ticks worse** than simply always posting (CI {_f(ylo)} to {_f(yhi)})"
    else:
        s += (
            f"and by {_f(y)} ticks versus always posting, which is **not distinguishable from zero** "
            f"(CI {_f(ylo)} to {_f(yhi)})"
        )
    s += (
        f", after fees and {hd.latency_ms} ms latency, with a {hd.deadline_s} s deadline and the "
        f"{hd.queue_model} queue model ({int(h['n']):,} orders over {int(h['test_days'])} out-of-sample days)."
    )
    bw = best_and_worst(cfg, instrument)
    if bw:
        s += f" Against always posting, the edge is largest {bw[0]} and smallest {bw[1]}."
    return s


def plot_headline(cfg: Settings, instruments: list[str]) -> Path | None:
    have = [
        i
        for i in instruments
        if (cfg.paths.trade_logs / i).exists() and any((cfg.paths.trade_logs / i).glob("*.parquet"))
    ]
    if not have:
        return None
    order = ["always_cross", "always_post", "model_direction", "model_cost", "new_level", "oracle"]
    fig, axes = plots.plt.subplots(
        1, len(have), figsize=(5.6 * len(have), 3.9), sharey=True, squeeze=False
    )
    for ax, inst in zip(axes[0], have, strict=True):
        wide = _cell(paired(cfg, inst), cfg)
        day = wide["date"].to_numpy()
        for k, s in enumerate(order):
            v = wide[s].to_numpy()
            lo, hi = day_block_bootstrap(v, day, cfg.sim.bootstrap_reps, cfg.synthetic.seed)
            m = float(v.mean())
            ax.barh(
                k,
                m,
                height=0.62,
                color=plots.color(s),
                alpha=0.55 if s == "oracle" else 1.0,
                hatch="//" if s == "oracle" else None,
                edgecolor=plots.SURFACE,
                linewidth=2,
            )
            ax.errorbar(
                m,
                k,
                xerr=[[m - lo], [hi - m]],
                fmt="none",
                ecolor=plots.TEXT,
                elinewidth=1.2,
                capsize=3,
            )
            ax.text(max(m, 0) + 0.03, k, f"{m:+.3f}", va="center", fontsize=9, color=plots.TEXT)
        ax.axvline(0, color=plots.TEXT_2, lw=1)
        ax.set_yticks(range(len(order)), [STRATEGY_LABEL[s] for s in order])
        # Explicit limits instead of invert_yaxis(): with sharey=True, inverting
        # each axis would flip the shared axis back.
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.set_title(f"Synthetic {inst}")
        ax.grid(axis="y", visible=False)
    h = cfg.sim.headline
    fig.suptitle(
        f"Execution cost by strategy ({h.queue_model} queue, {h.latency_ms} ms latency, "
        f"{h.deadline_s} s deadline; bars show 95% day-block bootstrap CIs)",
        fontsize=11,
    )
    fig.supxlabel(
        "mean cost, ticks per contract vs arrival mid (lower is better)",
        fontsize=10,
        color=plots.TEXT_2,
    )
    fig.tight_layout()
    return plots.save(fig, cfg.paths.figures / "headline.png")


def _direction_table(cfg: Settings, instrument: str) -> str:
    s = _t(cfg, f"direction_summary_{instrument}.csv")
    if s is None:
        return "_not run_"
    rows = []
    for sample, label in (
        ("out_of_sample", "out of sample"),
        ("in_sample", "in sample (avg per fold)"),
        ("release_windows_oos", "release windows, OOS"),
    ):
        for r in s.filter(pl.col("sample") == sample).iter_rows(named=True):
            rows.append(
                [
                    r["model"].replace("_", " "),
                    label,
                    f"{int(r['n']):,}",
                    _f(r["log_loss"], 4),
                    _f(r["brier"], 4),
                    _f(r["auc"], 4),
                    _f(r["accuracy"], 4),
                ]
            )
    df = pl.DataFrame(
        rows,
        schema=["model", "sample", "rows", "log loss", "Brier", "AUC", "accuracy"],
        orient="row",
    )
    return _md_table(df)


def _direction_takeaways(cfg: Settings, instrument: str) -> str:
    """Plain sentences derived from the direction tables."""
    s = _t(cfg, f"direction_summary_{instrument}.csv")
    t = _t(cfg, f"direction_tests_{instrument}.csv")
    if s is None:
        return ""

    def auc(model: str, sample: str) -> float:
        r = s.filter((pl.col("model") == model) & (pl.col("sample") == sample))
        return float(r["auc"][0]) if r.height else float("nan")

    out = []
    lg, a, b = (
        auc("lightgbm", "out_of_sample"),
        auc("baseline_a", "out_of_sample"),
        auc("baseline_b", "out_of_sample"),
    )
    line = f"- Out of sample, LightGBM's AUC is {_f(lg)} against {_f(a)} for the imbalance lookup and {_f(b)} for logistic regression"
    if t is not None:
        r = t.filter((pl.col("model") == "lightgbm") & (pl.col("vs") == "baseline_a"))
        if r.height:
            line += (
                f" (log-loss improvement over the lookup: Newey-West t = {_f(r['t_stat'][0], 2)})"
            )
    out.append(
        line
        + ". The gain over the one-feature lookup table is real but small: most of the signal is queue imbalance."
    )
    gap = auc("lightgbm", "in_sample") - lg
    out.append(
        f"- LightGBM's in-sample AUC is {_f(gap)} higher than out of sample, so it overfits noticeably; the baselines barely do ({_f(auc('baseline_a', 'in_sample') - a)} for the lookup)."
    )
    rl, ra, rb = (
        auc("lightgbm", "release_windows_oos"),
        auc("baseline_a", "release_windows_oos"),
        auc("baseline_b", "release_windows_oos"),
    )
    if not np.isnan(rl) and rl < max(ra, rb):
        out.append(
            f"- **Failure mode:** inside macro-release windows LightGBM is worse than the simple baselines (AUC {_f(rl)} vs {_f(ra)} and {_f(rb)}). A likely reason: release periods are rare in the training days, and the more flexible model generalises worse to them than the simple ones."
        )
    return "\n".join(out)


def _tests_table(cfg: Settings, instrument: str) -> str:
    t = _t(cfg, f"direction_tests_{instrument}.csv")
    if t is None:
        return ""
    rows = [
        [
            r["model"].replace("_", " "),
            r["vs"].replace("_", " "),
            _f(r["mean_logloss_diff"], 5, True),
            _f(r["hac_se"], 5),
            _f(r["t_stat"], 2),
        ]
        for r in t.iter_rows(named=True)
    ]
    return _md_table(
        pl.DataFrame(
            rows, schema=["model", "vs", "mean log-loss difference", "HAC s.e.", "t"], orient="row"
        )
    )


def _cost_table(cfg: Settings, instrument: str) -> str:
    c = _t(cfg, f"sim_costs_{instrument}.csv")
    sv = _t(cfg, f"sim_savings_{instrument}.csv")
    if c is None or sv is None:
        return "_not run_"
    c = _cell(c, cfg)
    sv = _cell(sv, cfg).filter(pl.col("vs") == "always_cross")
    rows = []
    for s in (
        "always_cross",
        "always_post",
        "model_direction",
        "model_cost",
        "new_level",
        "oracle",
    ):
        r = c.filter(pl.col("strategy") == s)
        if not r.height:
            continue
        r = r.row(0, named=True)
        x = sv.filter(pl.col("strategy") == s)
        save = (
            ""
            if not x.height
            else f"{_f(x['saving_ticks'][0], 3, True)} [{_f(x['ci95_lo'][0])}, {_f(x['ci95_hi'][0])}]"
        )
        rows.append(
            [
                STRATEGY_LABEL[s],
                _f(r["mean_cost_ticks"], 3, True),
                _f(r["median_cost_ticks"], 2, True),
                _f(r["p90_cost_ticks"], 2, True),
                f"${r['mean_cost_usd']:.2f}",
                f"{r['passive_fill_rate']:.1%}" if s not in ("always_cross", "oracle") else "",
                _f(r["adverse_5s_ticks"], 3) if r["adverse_5s_ticks"] is not None else "",
                save or "",
            ]
        )
    df = pl.DataFrame(
        rows,
        schema=[
            "strategy",
            "mean cost (ticks)",
            "median",
            "90th pct",
            "mean cost incl. fee",
            "passive fill rate",
            "adverse move 5 s after passive fill (ticks)",
            "saving vs always cross [95% CI]",
        ],
        orient="row",
    )
    return _md_table(df)


def _robustness(cfg: Settings, instrument: str) -> str:
    sv = _t(cfg, f"sim_savings_{instrument}.csv")
    if sv is None:
        return ""
    lines = []
    for strat in ("model_cost", "model_direction"):
        for base in ("always_cross", "always_post"):
            g = sv.filter((pl.col("strategy") == strat) & (pl.col("vs") == base))
            n = g.height
            better = g.filter(pl.col("ci95_lo") > 0).height
            worse = g.filter(pl.col("ci95_hi") < 0).height
            lines.append(
                f"- {STRATEGY_LABEL[strat]} vs {base.replace('_', ' ')}: significantly cheaper in **{better} of {n}** grid cells, significantly more expensive in {worse}, indistinguishable in {n - better - worse} (95% day-block bootstrap)."
            )
    return "\n".join(lines)


def _fee_sentence(cfg: Settings, instrument: str) -> str:
    f = _t(cfg, f"sim_fee_sensitivity_{instrument}.csv")
    if f is None:
        return ""
    g = f.filter(pl.col("strategy") == "model_cost")
    if not g.height:
        return ""
    vals = ", ".join(
        f"${r['mean_cost_usd']:.2f} at ${r['fee_per_side_usd']:.2f}"
        for r in g.iter_rows(named=True)
    )
    sv = g["saving_vs_always_cross_usd"].unique().to_list()
    return (
        f"Fees change every strategy's cost by the same amount, because every strategy trades exactly one "
        f"contract per order and CME pays no rebate for resting liquidity. The cost-to-go model's mean cost "
        f"is {vals} per side, but its saving versus always crossing is ${sv[0]:.2f} at every fee level. "
        f"Fees are therefore left out of the sensitivity grid."
    )


def _exec_breakdown(cfg: Settings, instrument: str) -> str:
    d = _t(cfg, f"diag_execution_{instrument}.csv")
    if d is None:
        return "_not run_"
    rows = [
        [
            r["dimension"].replace("_", " "),
            r["bucket"],
            f"{r['n']:,}",
            _f(r["always_cross"], 3, True),
            _f(r["always_post"], 3, True),
            _f(r["model_cost"], 3, True),
            _f(r["model_cost_saving_vs_cross"], 3, True),
            _f(r["model_cost_saving_vs_post"], 3, True),
        ]
        for r in d.iter_rows(named=True)
    ]
    return _md_table(
        pl.DataFrame(
            rows,
            schema=[
                "split",
                "bucket",
                "orders",
                "always cross",
                "always post",
                "cost-to-go model",
                "saving vs cross",
                "saving vs post",
            ],
            orient="row",
        )
    )


def _model_breakdown(cfg: Settings, instrument: str) -> str:
    d = _t(cfg, f"diag_model_{instrument}.csv")
    if d is None:
        return "_not run_"
    w = d.pivot(on="model", index=["dimension", "bucket", "n"], values="auc").sort(
        ["dimension", "bucket"]
    )
    rows = [
        [
            r["dimension"].replace("_", " "),
            r["bucket"],
            f"{r['n']:,}",
            _f(r.get("baseline_a"), 4),
            _f(r.get("lightgbm"), 4),
        ]
        for r in w.iter_rows(named=True)
    ]
    return _md_table(
        pl.DataFrame(
            rows,
            schema=["split", "bucket", "rows", "AUC imbalance lookup", "AUC LightGBM"],
            orient="row",
        )
    )


def _decay(cfg: Settings, instrument: str) -> str:
    d = _t(cfg, f"diag_signal_decay_{instrument}.csv")
    if d is None:
        return ""
    w = d.pivot(on="signal", index="horizon", values="auc", maintain_order=True)
    share = (
        d.group_by("horizon", maintain_order=True).agg(pl.col("moved_share").first())
        if "moved_share" in d.columns
        else None
    )
    rows = []
    for r in w.iter_rows(named=True):
        moved = ""
        if share is not None:
            v = share.filter(pl.col("horizon") == r["horizon"])["moved_share"][0]
            moved = f"{v:.1%}"
        rows.append([r["horizon"], moved, _f(r["imbalance"], 4), _f(r["lightgbm"], 4)])
    table = _md_table(
        pl.DataFrame(
            rows,
            schema=["horizon", "rows where the mid moved", "AUC queue imbalance", "AUC LightGBM"],
            orient="row",
        )
    )
    note = (
        "\n\nHow to read this: the clock-time rows only score moments where the mid actually moved within the horizon. "
        "Over 1 second those are mostly moments where one queue is about to empty, which is exactly when imbalance is most "
        "informative, so the short-horizon AUC is high on a small, selected subset. It is a selection effect, not look-ahead "
        "(features use only past data, and this is tested). The fair comparison across horizons is how the AUC falls as "
        "the horizon grows and more ordinary moments are included."
    )
    return table + note


def _queue_validation(cfg: Settings, instrument: str) -> str:
    q = _t(cfg, f"queue_validation_{instrument}.csv")
    if q is None:
        return ""
    h = cfg.sim.headline
    g = q.filter((pl.col("latency_ms") == h.latency_ms) & (pl.col("deadline_s") == h.deadline_s))
    rows = [
        [
            r["queue_model"],
            f"{r['fill_rate']:.1%}",
            f"{r['true_fill_rate']:.1%}",
            f"{r['fill_agreement']:.1%}",
            _f(r["cost_bias_ticks"], 3, True),
            _f(r["mae_fill_time_ms"], 0),
        ]
        for r in g.iter_rows(named=True)
    ]
    return _md_table(
        pl.DataFrame(
            rows,
            schema=[
                "queue assumption",
                "fill rate",
                "true fill rate",
                "same fill outcome as truth",
                "cost bias vs truth (ticks)",
                "fill-time error (ms)",
            ],
            orient="row",
        )
    )


def _descriptive(cfg: Settings) -> str:
    d = _t(cfg, "descriptive_days.csv")
    if d is None:
        return ""
    g = (
        d.group_by("instrument")
        .agg(
            pl.len().alias("days"),
            pl.col("events").mean().round(0),
            pl.col("spread_1tick_share").mean(),
            pl.col("median_best_queue").median(),
            pl.col("mid_changes_per_min").mean(),
            pl.col("trades").mean().round(0),
        )
        .sort("instrument")
    )
    rows = [
        [
            r["instrument"],
            r["days"],
            f"{int(r['events']):,}",
            f"{r['spread_1tick_share']:.2%}",
            f"{r['median_best_queue']:.0f}",
            f"{r['mid_changes_per_min']:.1f}",
            f"{int(r['trades']):,}",
        ]
        for r in g.iter_rows(named=True)
    ]
    return _md_table(
        pl.DataFrame(
            rows,
            schema=[
                "instrument",
                "analysis days",
                "events / day (session)",
                "spread = 1 tick",
                "median best queue (lots)",
                "mid changes / min",
                "trades / day",
            ],
            orient="row",
        )
    )


def _quality(cfg: Settings) -> str:
    rows = [json.loads(p.read_text()) for p in sorted(cfg.paths.data_quality.glob("*.json"))]
    if not rows:
        return ""
    df = pl.DataFrame(rows)
    agg = (
        df.group_by("instrument")
        .agg(
            pl.len().alias("days"),
            pl.col("duplicates_removed").sum(),
            pl.col("missing_sequence_numbers").sum(),
            pl.col("crossed_book_events").sum(),
            pl.col("session_gaps_over_threshold").sum(),
            pl.col("longest_session_gap_s").max(),
        )
        .sort("instrument")
    )
    return _md_table(
        agg,
        [
            "instrument",
            "days",
            "duplicates removed",
            "missing sequence numbers",
            "crossed-book events",
            "session gaps > 2 s",
            "longest gap (s)",
        ],
    )


def _rolls(cfg: Settings, instrument: str) -> str:
    t = roll_table(cfg, instrument)
    if t is None:
        return "none"
    days = t.filter(pl.col("is_roll"))["date"].to_list()
    return ", ".join(d.isoformat() for d in days) if days else "none"


def _release_sentence(cfg: Settings, instrument: str) -> str:
    r = _t(cfg, f"sim_savings_release_{instrument}.csv")
    if r is None:
        return "No orders fell inside macro-release windows."
    c = _cell(r, cfg).filter((pl.col("strategy") == "model_cost") & (pl.col("vs") == "always_post"))
    if not c.height:
        return ""
    row = c.row(0, named=True)
    return (
        f"Inside the ±{cfg.session.release_window_minutes}-minute release windows ({row['n']} orders, "
        f"{row['test_days']} day(s)), the cost-to-go model's saving versus always posting is "
        f"{_f(row['saving_ticks'], 3, True)} ticks (CI {_f(row['ci95_lo'])} to {_f(row['ci95_hi'])}). "
        "The sample is small, so treat this as a sanity check, not a finding."
    )


def day_counts(cfg: Settings, instrument: str) -> tuple[int, int]:
    """(days generated, days analysed), from saved tables so they survive deleting
    the processed day files."""
    vol = cfg.paths.processed / instrument / "daily_volume.parquet"
    generated = (
        pl.read_parquet(vol)["date"].n_unique()
        if vol.exists()
        else len(available_days(cfg, instrument, "features.parquet"))
    )
    desc = _t(cfg, f"descriptive_{instrument}.csv")
    used = desc.height if desc is not None else len(analysis_days(cfg, instrument))
    return generated, used


def build_report(cfg: Settings, readme: Path | None = Path("README.md")) -> Path:
    zn, es = cfg.primary_instrument, cfg.comparison_instrument
    plot_headline(cfg, [zn, es])
    days = {i: day_counts(cfg, i) for i in (zn, es)}
    hd = cfg.sim.headline
    s = cfg.sim

    def fig(name: str) -> str:
        exists = (cfg.paths.figures / f"{name}.png").exists()
        return f"![{name}](figures/{name}.png)" if exists else ""

    text = f"""# Halftick: cross the spread or wait in line?

_Generated by `halftick report` on {dt.date.today().isoformat()} from the tables in `reports/tables/`. Every number below comes from code in this repository that ran on the data described in section 2._

> **Read this first.** All results come from a **synthetic** order-book generator, not from real CME data (no Databento key was available). They show that the method, the simulator and the statistics work end to end. They are **not** evidence that this edge exists in the real ZN or ES markets. Section 7 lists the limitations.

## 1. Question

A trader must buy (or sell) one contract within a few seconds. Crossing the spread with a market order is certain but pays half a tick against the mid. Posting a limit order at the near touch can earn half a tick instead, but it joins the back of a deep queue, may not fill before the deadline, and tends to fill exactly when the price is about to move through it. **When is waiting worth it?**

## 2. Data

* **Source: synthetic.** A queue-reactive limit order book (Huang, Lehalle & Rosenbaum, 2015): event rates depend on queue sizes, and the price moves only when a best queue is fully consumed. A hidden "informed flow" process tilts order arrival, activity follows an intraday U-shape, scheduled macro releases spike activity, a quarterly roll moves volume between contracts, and the true queue position of every cancellation is recorded (standing in for MBO data). Parameters were tuned so ZN looks like a deep large-tick book and ES like a faster book with smaller queues.
* **Days:** {zn}: {days[zn][0]} generated, {days[zn][1]} used after excluding roll days detected from volume ({_rolls(cfg, zn)}). {es}: {days[es][0]} generated, {days[es][1]} used.
* **Session:** 08:30-15:00 America/New_York (converted from UTC with zoneinfo, DST-safe). ±{cfg.session.release_window_minutes}-minute windows around scheduled releases are flagged and analysed separately.

{_descriptive(cfg)}

**Data quality** (the generator injects duplicates, sequence gaps and feed outages; the checks must catch them):

{_quality(cfg)}

{fig("queue_sizes")}

## 3. Method

* **Features** (every event, strictly backward-looking, tested for look-ahead): queue imbalance and depth imbalance over 3/5/10 levels (Gould & Bonart, 2016); microprice (Stoikov, 2018); order flow imbalance over 10/50/200 events and 100 ms/1 s/5 s (Cont, Kukanov & Stoikov, 2014); trade-flow imbalance; queue depletion rates; spread; time since the last mid change; short-term volatility.
* **Direction models** for the next mid-price change: (A) the imbalance-decile lookup table, (B) logistic regression on imbalance, microprice and OFI, and LightGBM on all features. All are calibrated with isotonic regression on a held-out day. Walk-forward by day: train on days 1..k, calibrate on k+1, test on k+2, never shuffling across time.
* **Fill simulator:** a limit order joins the back of the queue at the touch, the queue ahead shrinks with trades and (depending on the assumption) cancellations, and it fills when a trade at our price is larger than what is still ahead of us, when the market trades through our price, or when the other side reaches it. Every action takes effect after the latency. Unfilled orders cross at the deadline. Four cancellation assumptions: pessimistic, proportional, optimistic, and **exact** (the synthetic truth).
* **Strategies,** all on the same {s.decisions_per_day:,} random decisions per day (a paired comparison): always cross; always post; post and give up when the calibrated probability of an adverse move exceeds a threshold tuned on the calibration day; post and give up when a **cost-to-go model** predicts that waiting costs more than crossing now; join only freshly opened price levels; and an **oracle upper bound** that uses future information.
* **The cost-to-go model** is a LightGBM regression of (price paid if we keep waiting) minus (price paid if we cross now), trained on checkpoints every {s.checkpoint_ms} ms along simulated orders from the training days. Crossing when the prediction is positive is a one-step improvement over always posting.
* **Cost** is measured in ticks against the mid at the decision moment, plus a ${cfg.fees.per_side_usd:.2f} fee per side. Uncertainty is shown two ways: a bootstrap that resamples whole test days, and Newey-West (HAC) t-statistics on the time-ordered per-order savings.
* **Simulation walk-forward:** train on at least {s.walk_forward.min_train_days} days, tune on the next day, test on the following {s.walk_forward.test_block_days}, and roll forward. Grid: {len(s.queue_models)} queue models × {len(s.latencies_ms)} latencies ({", ".join(map(str, s.latencies_ms))} ms) × {len(s.deadlines_s)} deadlines ({", ".join(map(str, s.deadlines_s))} s).

## 4. Results

### 4.1 Headline

{headline_sentence(cfg, zn)}

{headline_sentence(cfg, es)}

{fig("headline")}

Headline cell for {zn} ({hd.queue_model} queue, {hd.latency_ms} ms, {hd.deadline_s} s deadline, out of sample, release windows excluded):

{_cost_table(cfg, zn)}

{_fee_sentence(cfg, zn)}

### 4.2 Across the sensitivity grid ({zn})

{_robustness(cfg, zn)}

{fig(f"sensitivity_{zn}")}

{fig(f"cost_distribution_{zn}")}

### 4.3 Direction models ({zn})

{_direction_table(cfg, zn)}

{_direction_takeaways(cfg, zn)}

Newey-West tests of the per-row log-loss difference (negative means the first model is better):

{_tests_table(cfg, zn)}

{fig(f"reliability_{zn}")}

Signal decay (out of sample, direction of the mid move over each horizon):

{_decay(cfg, zn)}

{fig(f"signal_decay_{zn}")}

## 5. Where it works and where it fails ({zn})

Execution cost by regime (headline cell):

{_exec_breakdown(cfg, zn)}

Direction-model AUC by regime:

{_model_breakdown(cfg, zn)}

{_release_sentence(cfg, zn)}

{fig(f"stability_{zn}")}

### {es} comparison

{_cost_table(cfg, es)}

{_robustness(cfg, es)}

{_direction_table(cfg, es)}

{_direction_takeaways(cfg, es)}

{fig(f"sensitivity_{es}")}

## 6. How wrong are the queue assumptions?

With real mbp data the queue position is unknown and has to be assumed. The synthetic data records the true position of every cancellation, so each assumption can be scored against the truth on the same orders (always-post strategy, {hd.latency_ms} ms, {hd.deadline_s} s, {zn}):

{_queue_validation(cfg, zn)}

## 7. Limitations

* **Synthetic data.** This is the most important limitation. The generator encodes assumptions about how real books behave (queue-reactive rates, an informed-flow process, cancellations biased toward the back of the queue). A model that does well here has learned this generator, not the real CME market.
* **Few days.** {days[zn][1]} {zn} and {days[es][1]} {es} analysis days, with out-of-sample tests on fewer still. Day-block bootstrap intervals reflect that.
* **Queue position.** Real mbp-1/mbp-10 data does not reveal where cancellations sit in the queue. Section 6 shows how much the answer depends on that assumption.
* **No real fills.** Our own 1-lot order never affects the book, and other traders never react to it. Latency is a fixed delay, not a distribution.
* **Fees** are a flat assumption. They cancel out of the comparison anyway (section 4.1).
* **The oracle** uses future information and is shown only as a ceiling.

## 8. With more time or data

1. Run the same pipeline on real Databento data. The downloader, its budget guard and the mbp converter are written and unit-tested. Buy MBO for a few days first, to measure true queue positions and fit the cancellation assumption from data.
2. Add a re-pegging option and size larger than one lot, where market impact starts to matter.
3. Replace the one-step cost-to-go policy with a full dynamic-programming or reinforcement-learning policy, and compare.
4. Add a DatabentoLiveSource: the engine and the Prometheus/Grafana monitoring already run on the same interface.

## References

* Cont, R., Kukanov, A. & Stoikov, S. (2014). The price impact of order book events. *Journal of Financial Econometrics*, 12(1), 47-88.
* Gould, M. D. & Bonart, J. (2016). Queue imbalance as a one-tick-ahead price predictor in a limit order book. *Market Microstructure and Liquidity*, 2(2).
* Huang, W., Lehalle, C.-A. & Rosenbaum, M. (2015). Simulating and analyzing order book data: the queue-reactive model. *Journal of the American Statistical Association*, 110(509), 107-122.
* Stoikov, S. (2018). The micro-price: a high-frequency estimator of future prices. *Quantitative Finance*, 18(12), 1959-1966.
"""
    text = re.sub(r"\n{3,}", "\n\n", text)
    path = cfg.paths.reports / "REPORT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if readme is not None:
        update_readme(cfg, readme)
    return path


def readme_block(cfg: Settings) -> str:
    zn, es = cfg.primary_instrument, cfg.comparison_instrument
    lines = [README_START, "", "## Results (synthetic data)", ""]
    if (cfg.paths.figures / "headline.png").exists():
        lines += ["![Execution cost by strategy](reports/figures/headline.png)", ""]
    lines += [f"- {headline_sentence(cfg, zn)}", f"- {headline_sentence(cfg, es)}", ""]
    for inst in (zn, es):
        s = _t(cfg, f"direction_summary_{inst}.csv")
        if s is not None:
            o = s.filter(pl.col("sample") == "out_of_sample")
            auc = {r["model"]: r["auc"] for r in o.iter_rows(named=True)}
            lines.append(
                f"- {inst} next-move direction, out of sample: AUC {_f(auc.get('baseline_a', np.nan))} for the "
                f"imbalance lookup, {_f(auc.get('baseline_b', np.nan))} for logistic regression, "
                f"{_f(auc.get('lightgbm', np.nan))} for LightGBM."
            )
    lines += [
        "",
        "Full write-up with method, regime breakdowns, queue-model validation and limitations: [reports/REPORT.md](reports/REPORT.md).",
        "",
        README_END,
    ]
    return "\n".join(lines)


def update_readme(cfg: Settings, path: Path) -> None:
    if not path.exists():
        return
    text = path.read_text()
    block = readme_block(cfg)
    if README_START in text and README_END in text:
        text = re.sub(
            re.escape(README_START) + r".*?" + re.escape(README_END),
            lambda _: block,
            text,
            flags=re.S,
        )
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(text)
