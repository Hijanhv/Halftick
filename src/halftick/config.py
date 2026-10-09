"""Typed configuration loaded from config.yaml.

Pydantic validates the whole file at load time, so a typo or a wrong type fails
immediately with a clear message instead of surfacing deep inside a run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

QueueModelName = Literal["pessimistic", "proportional", "optimistic", "exact"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Paths(_Strict):
    raw: Path
    processed: Path
    calendar: Path
    reports: Path
    figures: Path
    tables: Path
    trade_logs: Path
    data_quality: Path
    readme: Path


class InstrumentSpec(_Strict):
    databento_symbol: str
    tick_size: float = Field(gt=0)
    tick_value_usd: float = Field(gt=0)
    contract_multiplier: float = Field(gt=0)


class Fees(_Strict):
    per_side_usd: float = Field(ge=0)
    sensitivity_usd: list[float]


class Session(_Strict):
    timezone: str
    start: str
    end: str
    release_window_minutes: int = Field(ge=0)
    time_of_day_buckets: dict[str, tuple[int, int]]


class Databento(_Strict):
    dataset: str
    stype_in: str
    budget_usd: float = Field(gt=0)
    spend_ledger: Path
    api_key_env: str


class Quality(_Strict):
    max_gap_seconds: float = Field(gt=0)


class Roll(_Strict):
    min_front_share: float = Field(gt=0, le=1)
    roll_buffer_days: int = Field(ge=0)


class BookCfg(_Strict):
    depth_levels: int = Field(ge=1)
    price_window_ticks: int = Field(ge=64)


class FeaturesCfg(_Strict):
    windows_ms: list[int]
    windows_events: list[int]
    depth_levels: list[int]
    vol_windows_s: list[int]


class TargetsCfg(_Strict):
    horizons_s: list[int]


class ModelsCfg(_Strict):
    sample_rows_per_day: int = Field(gt=0)
    imbalance_bins: int = Field(ge=2)
    laplace_alpha: float = Field(ge=0)
    min_train_days: int = Field(ge=1)
    calibration: Literal["isotonic", "platt"]
    logistic_features: list[str]
    lightgbm: dict[str, Any]
    reliability_bins: int = Field(ge=2)
    hac_lags: int = Field(ge=0)


class Headline(_Strict):
    queue_model: QueueModelName
    latency_ms: int
    deadline_s: int


class SimWalkForward(_Strict):
    min_train_days: int = Field(ge=1)
    test_block_days: int = Field(ge=1)


class SimCfg(_Strict):
    decisions_per_day: int = Field(gt=0)
    deadlines_s: list[int]
    latencies_ms: list[int]
    queue_models: list[QueueModelName]
    checkpoint_ms: int = Field(gt=0)
    headline: Headline
    new_level_max_queue_frac: float = Field(gt=0)
    direction_thresholds: list[float]
    cost_model_margin_ticks: float
    cost_model_rows_per_day: int = Field(gt=0)
    cost_model: dict[str, Any]
    walk_forward: SimWalkForward
    bootstrap_reps: int = Field(gt=0)
    adverse_horizons_s: list[int]


class DiagnosticsCfg(_Strict):
    vol_feature: str
    queue_feature: str


class MonitoringCfg(_Strict):
    metrics_port: int
    feature_latency_alert_ms: float
    no_data_alert_seconds: float


class QualityFaults(_Strict):
    duplicate_rate: float = Field(ge=0)
    sequence_gap_rate: float = Field(ge=0)
    outage_probability_per_day: float = Field(ge=0, le=1)
    outage_seconds: tuple[float, float]


class SyntheticRelease(_Strict):
    day_index: int = Field(ge=0)
    time: str
    event: str


class ReleaseShock(_Strict):
    activity_multiplier: float = Field(ge=1)
    decay_seconds: float = Field(gt=0)
    signal_jump: float


class SyntheticRoll(_Strict):
    roll_days: int = Field(ge=1)


class SyntheticInstrument(_Strict):
    n_days: int = Field(ge=1)
    roll_day_index: int = Field(ge=0)
    instrument_ids: tuple[int, int]
    raw_symbols: tuple[str, str]
    start_price: float = Field(gt=0)
    levels: int = Field(ge=2)
    queue_target: float = Field(gt=0)
    deep_queue_target: float = Field(gt=0)
    add_rate: float = Field(gt=0)
    deep_add_rate: float = Field(gt=0)
    cancel_rate: float = Field(gt=0)
    cancel_queue_power: float = Field(ge=0)
    market_rate: float = Field(gt=0)
    inside_rate: float = Field(gt=0)
    add_size_mean: float = Field(ge=1)
    cancel_size_mean: float = Field(ge=1)
    market_size_mean: float = Field(ge=1)
    market_big_prob: float = Field(ge=0, le=1)
    market_big_mean: float = Field(ge=1)
    new_level_size_mean: float = Field(ge=1)
    refill_boost: float = Field(ge=0)
    signal_tau_s: float = Field(gt=0)
    signal_vol: float = Field(ge=0)
    signal_market_kappa: float
    signal_limit_kappa: float
    cancel_back_skew: float = Field(gt=0)
    continue_prob: float = Field(ge=0, le=1)


class SyntheticCfg(_Strict):
    seed: int
    start_date: str
    generate_from: str
    generate_to: str
    holidays: list[str]
    quality_faults: QualityFaults
    releases: list[SyntheticRelease]
    release_shock: ReleaseShock
    roll: SyntheticRoll
    instruments: dict[str, SyntheticInstrument]


class Settings(_Strict):
    data_source: Literal["synthetic", "databento"]
    paths: Paths
    instruments: dict[str, InstrumentSpec]
    primary_instrument: str
    comparison_instrument: str
    fees: Fees
    session: Session
    databento: Databento
    quality: Quality
    roll: Roll
    book: BookCfg
    features: FeaturesCfg
    targets: TargetsCfg
    models: ModelsCfg
    sim: SimCfg
    diagnostics: DiagnosticsCfg
    monitoring: MonitoringCfg
    synthetic: SyntheticCfg

    @field_validator("instruments")
    @classmethod
    def _non_empty(cls, v: dict[str, InstrumentSpec]) -> dict[str, InstrumentSpec]:
        if not v:
            raise ValueError("at least one instrument is required")
        return v

    def instrument(self, name: str) -> InstrumentSpec:
        if name not in self.instruments:
            raise KeyError(f"unknown instrument {name!r}; configured: {sorted(self.instruments)}")
        return self.instruments[name]


def _set_dotted(data: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = data
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def load_config(path: str | Path = "config.yaml", overrides: list[str] | None = None) -> Settings:
    """Load and validate config.yaml.

    overrides are "dotted.key=value" strings; the value is parsed as YAML so
    numbers and lists work, e.g. "sim.decisions_per_day=500".
    """
    raw = yaml.safe_load(Path(path).read_text())
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got {item!r}")
        key, value = item.split("=", 1)
        _set_dotted(raw, key.strip(), yaml.safe_load(value))
    return Settings.model_validate(raw)
