"""Data-quality checks.

Every check counts problems and logs them; nothing is silently dropped. The
only automatic repairs are removing exact duplicate records and restoring
sequence order, and both are counted in the report. Everything else is
flagged so the analysis can exclude or study it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from halftick.book.l2book import (
    FLAG_BAD_SIZE,
    FLAG_CROSSED,
    FLAG_LOCKED,
    FLAG_NEGATIVE_SIZE,
    FLAG_OUT_OF_RANGE,
)
from halftick.log import get_logger
from halftick.schema import CLEAR

log = get_logger(__name__)
NS = 1_000_000_000


class DataQualityError(RuntimeError):
    """Raised when a day is too broken to analyse."""


@dataclass
class QualityReport:
    instrument: str
    date: str
    source: str
    n_raw: int = 0
    n_clean: int = 0
    duplicates_removed: int = 0
    out_of_order_timestamps: int = 0
    sequence_gaps: int = 0
    missing_sequence_numbers: int = 0
    zero_or_negative_sizes: int = 0
    crossed_book_events: int = 0
    locked_book_events: int = 0
    negative_level_after_update: int = 0
    out_of_price_window: int = 0
    session_gaps_over_threshold: int = 0
    longest_session_gap_s: float = 0.0
    gap_threshold_s: float = 2.0
    notes: list[str] = field(default_factory=list)

    def write(self, root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{self.instrument}_{self.date}.json"
        path.write_text(json.dumps(asdict(self), indent=2))
        return path


def clean_events(events: pl.DataFrame, report: QualityReport) -> pl.DataFrame:
    """Remove exact duplicates, restore sequence order, and count sequence problems."""
    report.n_raw = events.height
    deduped = events.unique(maintain_order=True)
    report.duplicates_removed = events.height - deduped.height

    ts = deduped["ts_event"].to_numpy()
    report.out_of_order_timestamps = int(np.sum(np.diff(ts) < 0))
    ordered = deduped.sort("sequence", maintain_order=True)

    seq = ordered["sequence"].to_numpy()
    d = np.diff(seq)
    report.sequence_gaps = int(np.sum(d > 1))
    report.missing_sequence_numbers = int(np.sum(d[d > 1] - 1))

    sizes = ordered["size"].to_numpy()
    actions = ordered["action"].to_numpy()
    report.zero_or_negative_sizes = int(np.sum((sizes <= 0) & (actions != CLEAR)))
    report.n_clean = ordered.height
    return ordered


def check_book_flags(flags: np.ndarray, report: QualityReport) -> None:
    report.crossed_book_events = int(np.sum((flags & FLAG_CROSSED) != 0))
    report.locked_book_events = int(np.sum((flags & FLAG_LOCKED) != 0))
    report.negative_level_after_update = int(np.sum((flags & FLAG_NEGATIVE_SIZE) != 0))
    report.out_of_price_window = int(np.sum((flags & FLAG_OUT_OF_RANGE) != 0))
    report.zero_or_negative_sizes += int(np.sum((flags & FLAG_BAD_SIZE) != 0))


def check_session_gaps(
    ts: np.ndarray, in_session: np.ndarray, report: QualityReport, max_gap_s: float
) -> None:
    report.gap_threshold_s = max_gap_s
    s = ts[in_session]
    if s.size < 2:
        report.notes.append("fewer than two in-session events")
        return
    gaps = np.diff(s) / NS
    report.session_gaps_over_threshold = int(np.sum(gaps > max_gap_s))
    report.longest_session_gap_s = round(float(gaps.max()), 3)


def enforce(report: QualityReport, max_crossed_share: float = 0.001) -> None:
    """Log every non-zero counter and raise if the day cannot be trusted."""
    counters = {
        k: v
        for k, v in asdict(report).items()
        if isinstance(v, int) and v and k not in ("n_raw", "n_clean")
    }
    if counters:
        log.warning(
            "data_quality_issues", instrument=report.instrument, date=report.date, **counters
        )
    if report.n_clean == 0:
        raise DataQualityError(f"{report.instrument} {report.date}: no events after cleaning")
    if report.crossed_book_events > max_crossed_share * report.n_clean:
        raise DataQualityError(
            f"{report.instrument} {report.date}: {report.crossed_book_events} crossed-book events "
            f"exceed {max_crossed_share:.3%} of {report.n_clean}"
        )
    if report.out_of_price_window:
        raise DataQualityError(
            f"{report.instrument} {report.date}: {report.out_of_price_window} updates fell outside "
            "the book's price window; increase book.price_window_ticks"
        )


def write_markdown_summary(root: Path) -> Path:
    rows = [json.loads(p.read_text()) for p in sorted(root.glob("*.json"))]
    lines = [
        "# Data quality summary",
        "",
        "| instrument | date | events | duplicates | seq gaps | out-of-order ts | bad sizes "
        "| crossed | locked | gaps > threshold | longest gap (s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['instrument']} | {r['date']} | {r['n_clean']:,} | {r['duplicates_removed']} | "
            f"{r['sequence_gaps']} | {r['out_of_order_timestamps']} | "
            f"{r['zero_or_negative_sizes']} | "
            f"{r['crossed_book_events']} | {r['locked_book_events']} | "
            f"{r['session_gaps_over_threshold']} | {r['longest_session_gap_s']} |"
        )
    path = root / "SUMMARY.md"
    path.write_text("\n".join(lines) + "\n")
    return path
