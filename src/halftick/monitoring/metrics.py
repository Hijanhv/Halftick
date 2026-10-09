"""Prometheus metrics for the replay engine.

Metrics are registered on a private CollectorRegistry so tests can create as
many exporters as they like without clashing with the global registry.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import polars as pl
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, start_http_server

from halftick.features.streaming import StreamingFeatures
from halftick.replay.engine import BookUpdate, Trade

NS = 1_000_000_000


class ReplayMetrics:
    """EngineObserver implementation that exports Prometheus metrics."""

    def __init__(
        self,
        instrument: str,
        registry: CollectorRegistry | None = None,
        session_bounds_ns: tuple[int, int] | None = None,
        tick_value_usd: float = 1.0,
    ) -> None:
        self.registry = registry or CollectorRegistry()
        self.instrument = instrument
        self.session = session_bounds_ns
        self.tick_value = tick_value_usd
        r = self.registry
        lbl = ["instrument"]
        self.events = Counter("halftick_events", "Market data events processed", lbl, registry=r)
        self.events_per_s = Gauge(
            "halftick_events_per_second", "Events processed per wall-clock second", lbl, registry=r
        )
        self.lag = Gauge(
            "halftick_replay_lag_seconds",
            "How far the replay runs behind its schedule",
            lbl,
            registry=r,
        )
        self.last_event_wall = Gauge(
            "halftick_last_event_wall_timestamp_seconds",
            "Wall time of the last processed event",
            lbl,
            registry=r,
        )
        self.market_time = Gauge(
            "halftick_market_timestamp_seconds",
            "Exchange timestamp of the last event",
            lbl,
            registry=r,
        )
        self.in_session = Gauge(
            "halftick_in_session",
            "1 while the replay clock is inside the day session",
            lbl,
            registry=r,
        )
        self.dq_errors = Counter(
            "halftick_data_quality_errors",
            "Data-quality problems seen during replay",
            [*lbl, "check"],
            registry=r,
        )
        self.mid = Gauge("halftick_mid_ticks", "Mid price in ticks", lbl, registry=r)
        self.spread = Gauge("halftick_spread_ticks", "Bid-ask spread in ticks", lbl, registry=r)
        self.imbalance = Gauge(
            "halftick_queue_imbalance", "Best-level queue imbalance", lbl, registry=r
        )
        self.microprice = Gauge(
            "halftick_microprice_ticks", "Microprice minus mid, ticks", lbl, registry=r
        )
        self.ofi = Gauge(
            "halftick_ofi", "Order flow imbalance over the last N events", lbl, registry=r
        )
        self.feature_latency = Histogram(
            "halftick_feature_latency_seconds",
            "Time to update streaming features per event",
            lbl,
            registry=r,
            buckets=(1e-6, 5e-6, 1e-5, 2.5e-5, 5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3, 5e-3, 1e-2),
        )
        self.orders = Counter(
            "halftick_sim_orders", "Simulated parent orders started", [*lbl, "strategy"], registry=r
        )
        self.fills = Counter(
            "halftick_sim_passive_fills", "Simulated passive fills", [*lbl, "strategy"], registry=r
        )
        self.completed = Counter(
            "halftick_sim_completed", "Simulated orders completed", [*lbl, "strategy"], registry=r
        )
        self.cost_ticks = Gauge(
            "halftick_sim_cumulative_cost_ticks",
            "Cumulative execution cost, ticks",
            [*lbl, "strategy"],
            registry=r,
        )
        self.pnl_usd = Gauge(
            "halftick_sim_cumulative_cost_usd",
            "Cumulative execution cost incl. fees, USD",
            [*lbl, "strategy"],
            registry=r,
        )
        self.avg_cost = Gauge(
            "halftick_sim_avg_cost_ticks",
            "Average cost per completed order, ticks",
            [*lbl, "strategy"],
            registry=r,
        )
        self._count = 0
        self._window_start = time.perf_counter()
        self._window_count = 0

    def serve(self, port: int) -> None:
        start_http_server(port, registry=self.registry)

    # EngineObserver ---------------------------------------------------------
    def event_processed(self, update: BookUpdate, wall_lag_s: float) -> None:
        i = self.instrument
        self.events.labels(i).inc()
        self._window_count += 1
        now = time.perf_counter()
        if now - self._window_start >= 1.0:
            self.events_per_s.labels(i).set(self._window_count / (now - self._window_start))
            self._window_start = now
            self._window_count = 0
        self.lag.labels(i).set(wall_lag_s)
        self.last_event_wall.labels(i).set(time.time())
        self.market_time.labels(i).set(update.ts_event / NS)
        if self.session:
            self.in_session.labels(i).set(
                1.0 if self.session[0] <= update.ts_event < self.session[1] else 0.0
            )
        if update.valid:
            self.mid.labels(i).set(update.mid)
            self.spread.labels(i).set(update.ask_px - update.bid_px)

    def data_quality_error(self, check: str) -> None:
        self.dq_errors.labels(self.instrument, check).inc()


class FeatureMonitor:
    """Subscriber that runs the streaming features and exports them with their latency."""

    def __init__(self, metrics: ReplayMetrics, features: StreamingFeatures | None = None) -> None:
        self.m = metrics
        self.f = features or StreamingFeatures()

    def on_trade(self, trade: Trade) -> None:
        self.f.on_trade(trade)

    def on_book_update(self, update: BookUpdate) -> None:
        self.f.on_book_update(update)
        i = self.m.instrument
        self.m.feature_latency.labels(i).observe(self.f.last_compute_seconds)
        self.m.imbalance.labels(i).set(self.f.imbalance)
        self.m.microprice.labels(i).set(self.f.microprice)
        self.m.ofi.labels(i).set(self.f.ofi)


class TradeLogPlayback:
    """Plays the simulator's trade log in step with the replay clock.

    Orders appear when the replay passes their decision time and complete at
    their recorded execution time, so the dashboard shows the same orders the
    research numbers are computed from, not a second simulation.
    """

    def __init__(
        self,
        metrics: ReplayMetrics,
        log_path: Path,
        queue_model: str,
        latency_ms: int,
        deadline_s: int,
    ) -> None:
        self.m = metrics
        df = (
            pl.read_parquet(log_path)
            .filter(
                (pl.col("queue_model") == queue_model)
                & (pl.col("latency_ms") == latency_ms)
                & (pl.col("deadline_s") == deadline_s)
                & (pl.col("strategy") != "oracle")
            )
            .select(
                "strategy", "decision_ts", "exec_ts", "filled_passive", "cost_ticks", "cost_usd"
            )
        )
        self.starts = df.sort("decision_ts")
        self.ends = df.sort("exec_ts")
        self._si = 0
        self._ei = 0
        self._started = False
        self._skipped_ts = 0
        self._totals: dict[str, list[float]] = {}
        self._start_ts = self.starts["decision_ts"].to_numpy()
        self._start_strat = self.starts["strategy"].to_list()
        self._end_ts = self.ends["exec_ts"].to_numpy()
        self._end_rows = self.ends.rows(named=True)

    def on_trade(self, trade: Trade) -> None:
        return None

    def on_book_update(self, update: BookUpdate) -> None:
        i = self.m.instrument
        t = update.ts_event
        if not self._started:
            # The replay may start mid-day (--start 09:30): orders decided before
            # the first replayed event belong to the part of the day we skipped.
            self._si = int(np.searchsorted(self._start_ts, t, side="left"))
            self._ei = int(np.searchsorted(self._end_ts, t, side="left"))
            self._skipped_ts = t
            self._started = True
        n = int(np.searchsorted(self._start_ts, t, side="right"))
        for k in range(self._si, n):
            self.m.orders.labels(i, self._start_strat[k]).inc()
        self._si = max(self._si, n)
        n = int(np.searchsorted(self._end_ts, t, side="right"))
        for k in range(self._ei, n):
            row = self._end_rows[k]
            if row["decision_ts"] < self._skipped_ts:
                continue  # started before the replay window; never shown as started
            s = row["strategy"]
            tot = self._totals.setdefault(s, [0.0, 0.0, 0.0])
            tot[0] += row["cost_ticks"]
            tot[1] += row["cost_usd"]
            tot[2] += 1
            self.m.completed.labels(i, s).inc()
            if row["filled_passive"]:
                self.m.fills.labels(i, s).inc()
            self.m.cost_ticks.labels(i, s).set(tot[0])
            self.m.pnl_usd.labels(i, s).set(tot[1])
            self.m.avg_cost.labels(i, s).set(tot[0] / tot[2])
        self._ei = max(self._ei, n)
