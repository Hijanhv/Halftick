"""Whole research pipeline on tiny synthetic data: train -> simulate -> diagnose.

Checks plumbing and invariants that must hold for any data, not result values.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl
import pytest

from halftick.config import load_config
from halftick.data.pipeline import available_days, build_day
from halftick.data.synthetic import generate_dataset
from halftick.diagnostics.breakdowns import descriptive, run_all
from halftick.models.evaluate import run_direction_walk_forward
from halftick.report import README_END, README_START, build_report
from halftick.sim.policy import STRATEGIES
from halftick.sim.runner import run_simulation
from halftick.sim.summary import write_tables
from tests.conftest import ROOT


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    t = tmp_path_factory.mktemp("tiny")
    over = [
        f"paths.processed={t}/processed",
        f"paths.calendar={t}/calendar",
        f"paths.reports={t}/reports",
        f"paths.figures={t}/reports/figures",
        f"paths.tables={t}/reports/tables",
        f"paths.trade_logs={t}/reports/trade_logs",
        f"paths.data_quality={t}/reports/data_quality",
        "synthetic.generate_from='08:20'",
        "synthetic.generate_to='09:00'",
        "session.end='08:58'",
        "synthetic.instruments.ZN.roll_day_index=50",
        "synthetic.releases=[{day_index: 4, time: '08:40', event: Test release}]",
        "models.min_train_days=2",
        "models.sample_rows_per_day=4000",
        "models.lightgbm.n_estimators=20",
        "sim.cost_model.n_estimators=20",
        "sim.decisions_per_day=60",
        "sim.latencies_ms=[0,10]",
        "sim.deadlines_s=[5,15]",
        "sim.queue_models=[proportional,exact]",
        "sim.headline={queue_model: proportional, latency_ms: 10, deadline_s: 15}",
        "sim.walk_forward.min_train_days=2",
        "sim.walk_forward.test_block_days=2",
        "sim.bootstrap_reps=50",
        "sim.cost_model_rows_per_day=2000",
    ]
    cfg = load_config(ROOT / "config.yaml", over)
    generate_dataset(cfg, "ZN", n_days=6)
    for d in available_days(cfg, "ZN"):
        build_day(cfg, "ZN", d)
    run_direction_walk_forward(cfg, "ZN")
    run_simulation(cfg, "ZN")
    tables = write_tables(cfg, "ZN")
    run_all(cfg, "ZN")
    descriptive(cfg, ["ZN"])
    return cfg, tables


def test_direction_outputs(tiny) -> None:  # type: ignore[no-untyped-def]
    cfg, _ = tiny
    summary = pl.read_csv(cfg.paths.tables / "direction_summary_ZN.csv")
    assert set(summary["sample"]) >= {"out_of_sample", "in_sample"}
    assert set(summary["model"]) == {"baseline_a", "baseline_b", "lightgbm"}
    oos = summary.filter(pl.col("sample") == "out_of_sample")
    assert ((oos["auc"] > 0) & (oos["auc"] < 1)).all()
    assert (cfg.paths.figures / "reliability_ZN.png").exists()


def test_trade_log_invariants(tiny) -> None:  # type: ignore[no-untyped-def]
    cfg, _ = tiny
    logs = pl.read_parquet(cfg.paths.trade_logs / "ZN" / "*.parquet")
    assert set(logs["strategy"]) == set(STRATEGIES)
    # every cell and strategy faced the same decisions (a paired comparison)
    counts = logs.group_by(["queue_model", "latency_ms", "deadline_s", "strategy"]).len()["len"]
    assert counts.n_unique() == 1
    wide = duckdb.sql(
        f"PIVOT (SELECT * FROM read_parquet('{cfg.paths.trade_logs}/ZN/*.parquet')) "
        "ON strategy USING first(cost_ticks) "
        "GROUP BY date, decision_id, queue_model, latency_ms, deadline_s"
    ).pl()
    for s in ("always_cross", "always_post", "model_cost", "model_direction"):
        assert (wide["oracle"] <= wide[s] + 1e-9).all(), f"oracle beaten by {s}"
    # a market order with no latency pays exactly half the spread
    cross0 = logs.filter((pl.col("strategy") == "always_cross") & (pl.col("latency_ms") == 0))
    spread = cross0["spread_ticks"].to_numpy()
    assert (cross0["cost_ticks"].to_numpy() == spread / 2).all()
    # passive fills happen at the posted price, never better
    fills = logs.filter(pl.col("filled_passive") & (pl.col("strategy") == "always_post"))
    assert (fills["fill_px_ticks"] == fills["entry_px_ticks"]).all()
    # the release window is flagged, not dropped
    assert logs["in_release_window"].any()


def test_summary_tables_and_figures(tiny) -> None:  # type: ignore[no-untyped-def]
    cfg, tables = tiny
    sav = tables["savings"]
    assert {"saving_ticks", "ci95_lo", "ci95_hi", "hac_t"} <= set(sav.columns)
    assert (
        (sav["ci95_lo"] <= sav["saving_ticks"] + 1e-9)
        & (sav["saving_ticks"] <= sav["ci95_hi"] + 1e-9)
    ).all()
    fees = tables["fees"]
    per_fee = fees.filter(pl.col("strategy") == "model_cost")["saving_vs_always_cross_usd"]
    assert per_fee.n_unique() == 1  # fees cancel out of the savings
    for name in (
        "cost_distribution_ZN",
        "sensitivity_ZN",
        "signal_decay_ZN",
        "stability_ZN",
        "queue_sizes",
    ):
        assert (cfg.paths.figures / f"{name}.png").exists(), name


def test_queue_validation_table(tiny) -> None:  # type: ignore[no-untyped-def]
    cfg, _ = tiny
    q = pl.read_csv(cfg.paths.tables / "queue_validation_ZN.csv")
    exact = q.filter(pl.col("queue_model") == "exact")
    assert (exact["fill_agreement"] == 1.0).all() and (exact["cost_bias_ticks"].abs() < 1e-12).all()


def test_outputs_are_written_under_the_configured_paths(tiny) -> None:  # type: ignore[no-untyped-def]
    cfg, _ = tiny
    assert Path(cfg.paths.tables).is_dir() and any(Path(cfg.paths.tables).glob("diag_*_ZN.csv"))


def test_report_is_generated_from_tables(tiny, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    cfg, _ = tiny
    readme = tmp_path / "README.md"
    readme.write_text(f"# x\n\n{README_START}\nold\n{README_END}\n\ntail\n")
    path = build_report(cfg, readme)
    text = path.read_text()
    for heading in ("## 1. Question", "## 4. Results", "## 6. How wrong", "## 7. Limitations"):
        assert heading in text
    assert "synthetic" in text.lower()
    assert "upper bound" in text.lower()
    assert "0.0 ticks" not in text  # numbers are formatted, never left empty
    new = readme.read_text()
    assert "old" not in new and "tail" in new and "ticks per contract" in new
    assert (cfg.paths.figures / "headline.png").exists()
