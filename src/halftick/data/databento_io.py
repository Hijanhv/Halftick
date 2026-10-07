"""Cost-checked Databento downloads and DBN-to-book-frame conversion.

Credits are scarce, so every request goes through the same three guards:

1. the file is cached in data/raw and never requested twice;
2. metadata.get_cost is called first and the price is logged;
3. a local spend ledger refuses any request that would push cumulative spend
   past databento.budget_usd, and nothing is bought without confirm=True.

This path needs DATABENTO_API_KEY. The project currently runs on synthetic
data, so this module is covered by unit tests on its pure functions (ledger,
paths, price conversion) but has not been run against the live API.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
from dotenv import load_dotenv

from halftick.config import Settings
from halftick.data.sessions import local_to_utc_ns
from halftick.log import get_logger
from halftick.schema import ADD, ASK, BID, CANCEL, CLEAR, MODIFY, NO_ASK, NO_BID, NONE, TRADE

log = get_logger(__name__)

FIXED_PRICE_SCALE = 1e-9  # Databento prices are int64 in units of 1e-9
UNDEF_PRICE = np.iinfo(np.int64).max
ACTION_MAP = {"A": ADD, "C": CANCEL, "M": MODIFY, "T": TRADE, "F": TRADE, "R": CLEAR, "N": MODIFY}
SIDE_MAP = {"B": BID, "A": ASK, "N": NONE}


class BudgetExceededError(RuntimeError):
    pass


class MissingApiKeyError(RuntimeError):
    pass


@dataclass(frozen=True)
class Request:
    instrument: str
    schema: str
    date: dt.date
    symbol: str
    start: str
    end: str

    def path(self, raw_root: Path) -> Path:
        return raw_root / self.instrument / self.schema / f"{self.date.isoformat()}.dbn.zst"


def make_request(cfg: Settings, instrument: str, schema: str, date: dt.date) -> Request:
    """One request per instrument, schema and day, covering a margin around the session."""
    spec = cfg.instrument(instrument)
    start_ns = local_to_utc_ns(date, cfg.synthetic.generate_from, cfg.session.timezone)
    end_ns = local_to_utc_ns(date, cfg.synthetic.generate_to, cfg.session.timezone)
    iso = lambda ns: dt.datetime.fromtimestamp(ns / 1e9, dt.UTC).isoformat()  # noqa: E731
    return Request(instrument, schema, date, spec.databento_symbol, iso(start_ns), iso(end_ns))


class SpendLedger:
    """Append-only record of what we have paid for, stored next to the raw data."""

    def __init__(self, path: Path, budget_usd: float) -> None:
        self.path = path
        self.budget = budget_usd

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        data: list[dict[str, Any]] = json.loads(self.path.read_text())
        return data

    @property
    def spent(self) -> float:
        return float(sum(e["cost_usd"] for e in self.entries()))

    def check(self, cost: float) -> None:
        if self.spent + cost > self.budget:
            raise BudgetExceededError(
                f"request costs ${cost:.4f}; spent ${self.spent:.4f} of ${self.budget:.2f} budget"
            )

    def record(self, request: Request, cost: float) -> None:
        entries = self.entries()
        entries.append(
            {
                "when": dt.datetime.now(dt.UTC).isoformat(),
                "instrument": request.instrument,
                "schema": request.schema,
                "date": request.date.isoformat(),
                "cost_usd": cost,
            }
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(entries, indent=2))


def _client(cfg: Settings) -> Any:
    load_dotenv()
    key = os.environ.get(cfg.databento.api_key_env)
    if not key:
        raise MissingApiKeyError(
            f"{cfg.databento.api_key_env} is not set. Put it in .env (see .env.example)."
        )
    import databento as db

    return db.Historical(key)


def quote(cfg: Settings, request: Request, client: Any | None = None) -> float:
    client = client or _client(cfg)
    cost = float(
        client.metadata.get_cost(
            dataset=cfg.databento.dataset,
            symbols=[request.symbol],
            schema=request.schema,
            start=request.start,
            end=request.end,
            stype_in=cfg.databento.stype_in,
        )
    )
    log.info(
        "databento_quote",
        instrument=request.instrument,
        schema=request.schema,
        date=request.date.isoformat(),
        cost_usd=round(cost, 4),
    )
    return cost


def download(
    cfg: Settings, request: Request, confirm: bool, client: Any | None = None
) -> Path | None:
    """Download one day if it is not cached, the budget allows it and confirm is True."""
    path = request.path(cfg.paths.raw)
    if path.exists():
        log.info("databento_cache_hit", path=str(path))
        return path
    client = client or _client(cfg)
    cost = quote(cfg, request, client)
    ledger = SpendLedger(cfg.databento.spend_ledger, cfg.databento.budget_usd)
    ledger.check(cost)
    if not confirm:
        log.info(
            "databento_dry_run",
            cost_usd=round(cost, 4),
            spent_usd=round(ledger.spent, 4),
            budget_usd=ledger.budget,
            hint="rerun with --yes to buy",
        )
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    client.timeseries.get_range(
        dataset=cfg.databento.dataset,
        symbols=[request.symbol],
        schema=request.schema,
        start=request.start,
        end=request.end,
        stype_in=cfg.databento.stype_in,
        path=str(path),
    )
    ledger.record(request, cost)
    log.info("databento_downloaded", path=str(path), cost_usd=round(cost, 4))
    return path


def to_ticks(fixed: npt.NDArray[np.int64], tick_size: float, empty: int) -> npt.NDArray[np.int64]:
    """Convert Databento fixed-point prices to integer ticks; undefined prices map to `empty`."""
    out = np.full(fixed.shape[0], empty, dtype=np.int64)
    ok = fixed != UNDEF_PRICE
    out[ok] = np.rint(fixed[ok] * FIXED_PRICE_SCALE / tick_size).astype(np.int64)
    return out


def mbp_to_book_frame(records: npt.NDArray[Any], tick_size: float, levels: int) -> pl.DataFrame:
    """Turn an mbp-1 or mbp-10 structured array into the canonical book frame.

    MBP records already carry the post-event book, so no book replay is
    needed. Databento lists non-empty levels; we place them on the dense price
    grid used everywhere else (column i = i ticks from the touch).
    """
    n = records.shape[0]
    names = records.dtype.names or ()
    depth = sum(1 for nm in names if nm.startswith("bid_px_"))
    bid_px = to_ticks(records["bid_px_00"].astype(np.int64), tick_size, NO_BID)
    ask_px = to_ticks(records["ask_px_00"].astype(np.int64), tick_size, NO_ASK)
    bid_grid = np.zeros((n, levels), dtype=np.int64)
    ask_grid = np.zeros((n, levels), dtype=np.int64)
    for lvl in range(depth):
        bp = to_ticks(records[f"bid_px_{lvl:02d}"].astype(np.int64), tick_size, NO_BID)
        ap = to_ticks(records[f"ask_px_{lvl:02d}"].astype(np.int64), tick_size, NO_ASK)
        bs = records[f"bid_sz_{lvl:02d}"].astype(np.int64)
        asz = records[f"ask_sz_{lvl:02d}"].astype(np.int64)
        bo = bid_px - bp
        ao = ap - ask_px
        rows = np.arange(n)
        okb = (bp != NO_BID) & (bo >= 0) & (bo < levels)
        oka = (ap != NO_ASK) & (ao >= 0) & (ao < levels)
        bid_grid[rows[okb], bo[okb]] = bs[okb]
        ask_grid[rows[oka], ao[oka]] = asz[oka]
    actions = np.array(
        [
            ACTION_MAP.get(chr(a) if isinstance(a, (int, np.integer)) else str(a), MODIFY)
            for a in records["action"]
        ],
        dtype=np.int8,
    )
    sides = np.array(
        [
            SIDE_MAP.get(chr(s) if isinstance(s, (int, np.integer)) else str(s), NONE)
            for s in records["side"]
        ],
        dtype=np.int8,
    )
    frame: dict[str, Any] = {
        "ts_event": records["ts_event"].astype(np.int64),
        "sequence": records["sequence"].astype(np.int64),
        "instrument_id": records["instrument_id"].astype(np.int32),
        "action": actions,
        "side": sides,
        "price": to_ticks(records["price"].astype(np.int64), tick_size, 0),
        "size": records["size"].astype(np.int64),
        "queue_pos": np.full(n, -1, dtype=np.int64),  # unknown without MBO
        "bid_px": bid_px,
        "ask_px": ask_px,
    }
    for i in range(levels):
        frame[f"bid_sz_{i}"] = bid_grid[:, i]
        frame[f"ask_sz_{i}"] = ask_grid[:, i]
    return pl.DataFrame(frame)


def read_dbn(path: Path) -> npt.NDArray[Any]:
    import databento as db

    store = db.DBNStore.from_file(str(path))
    arr: npt.NDArray[Any] = store.to_ndarray()
    return arr


def read_definitions(path: Path) -> pl.DataFrame:
    """Extract tick size and symbol per instrument from a cached definition file."""
    rec = read_dbn(path)
    return pl.DataFrame(
        {
            "instrument_id": rec["instrument_id"].astype(np.int64),
            "raw_symbol": [
                s.decode() if isinstance(s, bytes) else str(s) for s in rec["raw_symbol"]
            ],
            "min_price_increment": rec["min_price_increment"].astype(np.int64) * FIXED_PRICE_SCALE,
            "expiration": rec["expiration"].astype(np.int64),
        }
    ).unique("instrument_id")


def verify_tick_size(cfg: Settings, instrument: str, definitions: pl.DataFrame) -> None:
    expected = cfg.instrument(instrument).tick_size
    seen = set(np.round(definitions["min_price_increment"].to_numpy(), 9))
    if seen != {round(expected, 9)}:
        raise ValueError(
            f"{instrument}: config tick_size {expected} but definition says {sorted(seen)}"
        )
