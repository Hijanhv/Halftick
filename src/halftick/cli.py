"""Command-line interface: halftick <command>.

synth           generate synthetic order-book days
download        cost-checked Databento download (dry run unless --yes)
build-features  quality checks, book, features and targets for every stored day
train           walk-forward direction models, metrics, reliability plots
simulate        queue-aware execution simulation across the sensitivity grid
diagnose        breakdowns, signal decay, stability, queue-model validation
report          REPORT.md, figures and the README results block
replay          event-driven replay with Prometheus metrics (optionally real time)
all             synth -> build-features -> train -> simulate -> diagnose -> report
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Annotated

import typer

from halftick.config import Settings, load_config
from halftick.log import configure_logging, get_logger

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
log = get_logger("halftick.cli")
_state: dict[str, Settings] = {}

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="Path to config.yaml")]
SetOpt = Annotated[
    list[str] | None,
    typer.Option("--set", help="Override a config key, e.g. --set sim.decisions_per_day=500"),
]
InstOpt = Annotated[str, typer.Option("--instrument", "-i", help="Instrument key from config.yaml")]


@app.callback()
def main(
    config: ConfigOpt = Path("config.yaml"),
    set_: SetOpt = None,
    log_level: Annotated[str, typer.Option("--log-level")] = "INFO",
    pretty: Annotated[
        bool, typer.Option("--pretty", help="Human-readable logs instead of JSON")
    ] = False,
) -> None:
    configure_logging(log_level, json=not pretty)
    _state["cfg"] = load_config(config, set_)


def cfg() -> Settings:
    return _state["cfg"]


@app.command()
def synth(
    instrument: InstOpt = "ZN",
    days: Annotated[int | None, typer.Option(help="Override the number of days")] = None,
) -> None:
    """Generate synthetic order-book days for an instrument."""
    from halftick.data.synthetic import generate_dataset

    paths = generate_dataset(cfg(), instrument, days)
    log.info("synth_done", instrument=instrument, days=len(paths))


@app.command()
def download(
    instrument: InstOpt = "ZN",
    start: Annotated[str, typer.Option(help="First trading day, YYYY-MM-DD")] = "",
    end: Annotated[str, typer.Option(help="Last trading day, YYYY-MM-DD (inclusive)")] = "",
    schema: Annotated[str, typer.Option(help="mbp-1, mbp-10, definition or mbo")] = "mbp-1",
    yes: Annotated[
        bool, typer.Option("--yes", help="Actually buy; without it this only quotes the cost")
    ] = False,
) -> None:
    """Quote, budget-check and (with --yes) download Databento data, one cached file per day."""
    from halftick.data.databento_io import download as dl
    from halftick.data.databento_io import make_request, quote

    d0 = dt.date.fromisoformat(start)
    d1 = dt.date.fromisoformat(end or start)
    total = 0.0
    d = d0
    while d <= d1:
        if d.weekday() < 5:
            req = make_request(cfg(), instrument, schema, d)
            if not req.path(cfg().paths.raw).exists():
                total += quote(cfg(), req)
            dl(cfg(), req, confirm=yes)
        d += dt.timedelta(days=1)
    log.info(
        "download_summary",
        instrument=instrument,
        schema=schema,
        quoted_usd=round(total, 4),
        bought=yes,
    )


@app.command("build-features")
def build_features(
    instrument: InstOpt = "ZN",
    date: Annotated[str | None, typer.Option(help="Only this day")] = None,
    drop_events: Annotated[
        bool,
        typer.Option(
            "--drop-events",
            help="Delete synthetic events.parquet after a successful build (regenerate with synth)",
        ),
    ] = False,
) -> None:
    """Validate, rebuild the book, and compute features and targets for stored days."""
    import json

    from halftick.data.pipeline import available_days, build_day, day_dir
    from halftick.data.quality import write_markdown_summary

    days = [dt.date.fromisoformat(date)] if date else available_days(cfg(), instrument)
    for d in days:
        build_day(cfg(), instrument, d)
        meta = day_dir(cfg(), instrument, d) / "meta.json"
        if (
            drop_events
            and meta.exists()
            and json.loads(meta.read_text()).get("source") == "synthetic"
        ):
            # Synthetic events are a pure function of the seed, so this only saves disk.
            (day_dir(cfg(), instrument, d) / "events.parquet").unlink(missing_ok=True)
    write_markdown_summary(cfg().paths.data_quality)


@app.command()
def train(instrument: InstOpt = "ZN") -> None:
    """Walk-forward evaluation of baseline A, baseline B and LightGBM."""
    from halftick.models.evaluate import run_direction_walk_forward

    summary = run_direction_walk_forward(cfg(), instrument)
    log.info("train_done", instrument=instrument, rows=summary.height)


@app.command()
def simulate(instrument: InstOpt = "ZN") -> None:
    """Run the execution simulator over the full grid and write cost tables."""
    from halftick.sim.runner import run_simulation
    from halftick.sim.summary import write_tables

    run_simulation(cfg(), instrument)
    write_tables(cfg(), instrument)


@app.command()
def diagnose(instrument: InstOpt = "ZN") -> None:
    """Breakdowns by regime, signal decay, stability and queue-model validation."""
    from halftick.diagnostics.breakdowns import run_all

    run_all(cfg(), instrument)


@app.command()
def report() -> None:
    """Descriptive statistics, reports/REPORT.md and the README results block."""
    from halftick.diagnostics.breakdowns import descriptive
    from halftick.report import build_report

    c = cfg()
    descriptive(c, [c.primary_instrument, c.comparison_instrument])
    path = build_report(c)
    log.info("report_written", path=str(path))


@app.command()
def replay(
    instrument: InstOpt = "ZN",
    date: Annotated[
        str | None, typer.Option(help="Processed day to replay; default: first available")
    ] = None,
    synthetic: Annotated[
        bool, typer.Option("--synthetic", help="Generate the day in memory instead of reading disk")
    ] = False,
    realtime: Annotated[
        bool, typer.Option("--realtime", help="Pace the replay at market speed")
    ] = False,
    speed: Annotated[
        float, typer.Option(help="Market seconds per wall second in --realtime mode")
    ] = 1.0,
    start: Annotated[str | None, typer.Option(help="Start at this ET time, e.g. 09:30")] = None,
    port: Annotated[
        int | None, typer.Option("--metrics-port", help="Serve Prometheus metrics on this port")
    ] = None,
    loop: Annotated[
        bool, typer.Option("--loop", help="Restart from the beginning when the day ends")
    ] = False,
) -> None:
    """Replay a day through the event-driven engine with live metrics."""
    from halftick.data.pipeline import available_days, day_dir
    from halftick.data.sessions import local_to_utc_ns, session_bounds_ns
    from halftick.monitoring.metrics import FeatureMonitor, ReplayMetrics, TradeLogPlayback
    from halftick.replay.engine import ReplayEngine
    from halftick.replay.sources import MarketDataSource, ParquetSource, SyntheticSource

    c = cfg()
    if date:
        day = dt.date.fromisoformat(date)
    elif synthetic:
        day = dt.date.fromisoformat(c.synthetic.start_date)
    else:
        # Prefer a day with a simulated trade log so the dashboard shows the
        # research orders; then any processed day; then generate one in memory.
        processed = available_days(c, instrument, "features.parquet")
        logged = [
            d
            for d in processed
            if (c.paths.trade_logs / instrument / f"{d.isoformat()}.parquet").exists()
        ]
        if logged or processed:
            day = (logged or processed)[0]
        else:
            synthetic = True
            day = dt.date.fromisoformat(c.synthetic.start_date)
            log.info("replay_fallback_synthetic", reason="no processed days found")
    metrics = ReplayMetrics(
        instrument,
        session_bounds_ns=session_bounds_ns(c, day),
        tick_value_usd=c.instrument(instrument).tick_value_usd,
    )
    if port:
        metrics.serve(port)
    start_ns = local_to_utc_ns(day, start, c.session.timezone) if start else None
    while True:
        source: MarketDataSource
        if synthetic:
            source = SyntheticSource(c, instrument, day)
        else:
            source = ParquetSource(day_dir(c, instrument, day) / "features.parquet", instrument)
        subs: list = [FeatureMonitor(metrics)]
        tl = c.paths.trade_logs / instrument / f"{day.isoformat()}.parquet"
        if tl.exists():
            h = c.sim.headline
            subs.append(TradeLogPlayback(metrics, tl, h.queue_model, h.latency_ms, h.deadline_s))
        engine = ReplayEngine(
            source, subs, realtime=realtime, speed=speed, observer=metrics, start_ns=start_ns
        )
        stats = engine.run()
        log.info(
            "replay_done",
            instrument=instrument,
            date=day.isoformat(),
            events=stats.events,
            trades=stats.trades,
            crossed=stats.crossed,
            wall_seconds=round(stats.wall_seconds, 2),
        )
        if not loop:
            break


@app.command(name="all")
def run_all_cmd() -> None:
    """Generate synthetic data and run the whole research pipeline for both instruments."""
    c = cfg()
    for inst in (c.primary_instrument, c.comparison_instrument):
        if c.data_source == "synthetic":
            synth(inst)
        build_features(inst, drop_events=c.data_source == "synthetic")
        train(inst)
        simulate(inst)
        diagnose(inst)
    report()


if __name__ == "__main__":
    app()
