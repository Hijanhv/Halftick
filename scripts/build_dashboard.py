"""Generate monitoring/grafana/dashboards/halftick.json.

Run with: uv run python scripts/build_dashboard.py
Strategy colors match the report figures (halftick.plots.ENTITY_COLORS).
"""

from __future__ import annotations

import json
from pathlib import Path

from halftick.plots import ENTITY_COLORS

DS = {"type": "prometheus", "uid": "prometheus"}
LATENCY_RATE = "sum by (le) (rate(halftick_feature_latency_seconds_bucket[30s]))"
STRATS = ["always_cross", "always_post", "model_direction", "model_cost", "new_level"]
_id = 0


def panel(
    title,
    kind,
    x,
    y,
    w,
    h,
    targets,
    unit="short",
    desc="",
    decimals=None,
    overrides=None,
    opts=None,
):
    global _id
    _id += 1
    p = {
        "id": _id,
        "title": title,
        "type": kind,
        "description": desc,
        "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [
            {"refId": chr(65 + i), "datasource": DS, "expr": e, "legendFormat": lf}
            for i, (e, lf) in enumerate(targets)
        ],
        "fieldConfig": {
            "defaults": {"unit": unit, "custom": {"lineWidth": 2, "fillOpacity": 0}},
            "overrides": overrides or [],
        },
        "options": opts or {},
    }
    if decimals is not None:
        p["fieldConfig"]["defaults"]["decimals"] = decimals
    return p


def strategy_colors():
    return [
        {
            "matcher": {"id": "byName", "options": s},
            "properties": [
                {"id": "color", "value": {"mode": "fixed", "fixedColor": ENTITY_COLORS[s]}}
            ],
        }
        for s in STRATS
    ]


def stat(title, x, y, expr, unit="short", desc="", decimals=None, thresholds=None):
    p = panel(
        title,
        "stat",
        x,
        y,
        4,
        4,
        [(expr, "")],
        unit,
        desc,
        decimals,
        opts={
            "reduceOptions": {"calcs": ["lastNotNull"]},
            "colorMode": "value",
            "graphMode": "area",
        },
    )
    if thresholds:
        p["fieldConfig"]["defaults"]["thresholds"] = {"mode": "absolute", "steps": thresholds}
        p["fieldConfig"]["defaults"]["color"] = {"mode": "thresholds"}
    return p


def row(title, y):
    global _id
    _id += 1
    return {
        "id": _id,
        "type": "row",
        "title": title,
        "collapsed": False,
        "gridPos": {"x": 0, "y": y, "w": 24, "h": 1},
    }


ok_bad = [{"color": "green", "value": None}, {"color": "red", "value": 1}]
panels = [
    row("Replay engine", 0),
    stat(
        "Events / second",
        0,
        1,
        "sum(halftick_events_per_second)",
        desc="Market data events processed per wall-clock second",
    ),
    stat(
        "Replay lag",
        4,
        1,
        "max(halftick_replay_lag_seconds)",
        "s",
        "How far the replay runs behind its schedule",
        3,
        [
            {"color": "green", "value": None},
            {"color": "orange", "value": 0.5},
            {"color": "red", "value": 1},
        ],
    ),
    stat(
        "Seconds since last event",
        8,
        1,
        "time() - max(halftick_last_event_wall_timestamp_seconds)",
        "s",
        "Alert fires above 5 s during the session",
        1,
        [
            {"color": "green", "value": None},
            {"color": "orange", "value": 2},
            {"color": "red", "value": 5},
        ],
    ),
    stat(
        "In session",
        12,
        1,
        "max(halftick_in_session)",
        desc="1 while the replay clock is inside 08:30-15:00 ET",
    ),
    stat(
        "Data-quality errors",
        16,
        1,
        "sum(halftick_data_quality_errors_total) or vector(0)",
        desc="Crossed or locked books seen during replay",
        thresholds=ok_bad,
    ),
    stat(
        "Firing alerts",
        20,
        1,
        'count(ALERTS{alertstate="firing"}) or vector(0)',
        desc="Alerts defined in monitoring/alerts.yml",
        thresholds=ok_bad,
    ),
    row("Market", 5),
    panel(
        "Mid price (ticks)",
        "timeseries",
        0,
        6,
        12,
        8,
        [("halftick_mid_ticks", "{{instrument}} mid")],
        decimals=1,
    ),
    panel(
        "Queue imbalance and microprice",
        "timeseries",
        12,
        6,
        12,
        8,
        [
            ("halftick_queue_imbalance", "queue imbalance"),
            ("halftick_microprice_ticks", "microprice - mid (ticks)"),
        ],
        desc="Imbalance near +1 means the bid queue is much larger than the ask queue",
        decimals=2,
    ),
    panel(
        "Order flow imbalance (last 50 events)",
        "timeseries",
        0,
        14,
        12,
        7,
        [("halftick_ofi", "OFI")],
        desc="Cont, Kukanov & Stoikov (2014)",
    ),
    panel(
        "Spread (ticks)",
        "timeseries",
        12,
        14,
        12,
        7,
        [("halftick_spread_ticks", "spread")],
        decimals=0,
    ),
    row("Simulated execution (headline cell, from the research trade log)", 21),
    panel(
        "Average cost per order (ticks vs arrival mid, lower is better)",
        "timeseries",
        0,
        22,
        12,
        9,
        [("halftick_sim_avg_cost_ticks", "{{strategy}}")],
        decimals=3,
        overrides=strategy_colors(),
    ),
    panel(
        "Cumulative cost incl. fees (USD)",
        "timeseries",
        12,
        22,
        12,
        9,
        [("halftick_sim_cumulative_cost_usd", "{{strategy}}")],
        "currencyUSD",
        overrides=strategy_colors(),
    ),
    panel(
        "Orders started and passive fills",
        "timeseries",
        0,
        31,
        12,
        8,
        [
            ("halftick_sim_orders_total", "orders {{strategy}}"),
            ("halftick_sim_passive_fills_total", "passive fills {{strategy}}"),
        ],
    ),
    panel(
        "Feature computation latency",
        "timeseries",
        12,
        31,
        12,
        8,
        [
            (
                f"histogram_quantile(0.5, {LATENCY_RATE})",
                "p50",
            ),
            (
                f"histogram_quantile(0.99, {LATENCY_RATE})",
                "p99",
            ),
        ],
        "s",
        "Alert threshold: p99 above 5 ms",
    ),
]
for p in panels:
    if p["type"] == "timeseries":
        p["options"] = {
            "legend": {"displayMode": "list", "placement": "bottom"},
            "tooltip": {"mode": "multi"},
        }

dashboard = {
    "uid": "halftick-replay",
    "title": "Halftick replay",
    "tags": ["halftick"],
    "timezone": "browser",
    "schemaVersion": 39,
    "version": 1,
    "refresh": "2s",
    "time": {"from": "now-15m", "to": "now"},
    "panels": panels,
}
out = Path(__file__).resolve().parents[1] / "monitoring/grafana/dashboards/halftick.json"
out.write_text(json.dumps(dashboard, indent=2) + "\n")
print(f"wrote {out}")
