"""Synthetic large-tick order book generator.

This is a queue-reactive model in the spirit of Huang, Lehalle & Rosenbaum
(2015), "Simulating and analyzing order book data: the queue-reactive model".
Event intensities depend on the current queue sizes, and the price only moves
when a best queue is fully consumed. That is the mechanism that makes queue
imbalance predictive in real large-tick markets, so the signal the models find
is produced by market structure rather than written into the data directly.

On top of that we add:

* a hidden "informed flow" signal x(t), an Ornstein-Uhlenbeck process that tilts
  market-order and limit-order arrival. It makes order-flow features (OFI,
  trade-flow imbalance) carry information beyond the static queue sizes;
* an intraday U-shaped activity profile (busier at the open and close);
* scheduled "macro release" shocks: activity spikes and x jumps;
* the true queue position of every cancellation, standing in for MBO data,
  so the fill simulator's queue assumptions can be checked against the truth;
* a few injected data faults (duplicates, sequence gaps, a feed outage) so the
  data-quality checks have something real to catch.

Everything here is synthetic. Results computed from it show that the method
and the system work; they are not evidence about the real ZN market.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import njit

from halftick.config import Settings, SyntheticInstrument
from halftick.log import get_logger
from halftick.schema import ADD, ASK, BID, CANCEL, TRADE

log = get_logger(__name__)

# Parameter vector layout (numba wants a flat float array, not a pydantic model).
P_QT, P_DQT, P_ADD, P_DADD, P_CXL, P_MKT, P_IN = 0, 1, 2, 3, 4, 5, 6
P_ADDSZ, P_CXLSZ, P_MKTSZ, P_BIGP, P_BIGSZ, P_NEWSZ, P_REFILL = 7, 8, 9, 10, 11, 12, 13
P_TAU, P_SVOL, P_KM, P_KL, P_SKEW, P_CONT = 14, 15, 16, 17, 18, 19
P_RMULT, P_RDECAY, P_RJUMP = 20, 21, 22
P_CPOW = 23
N_PARAMS = 24

# Buffer columns
B_TS, B_ACT, B_SIDE, B_PX, B_SZ, B_QPOS = 0, 1, 2, 3, 4, 5

NS = 1_000_000_000


def params_vector(
    p: SyntheticInstrument, shock_mult: float, shock_decay: float, shock_jump: float
) -> npt.NDArray[np.float64]:
    v = np.zeros(N_PARAMS)
    v[P_QT] = p.queue_target
    v[P_DQT] = p.deep_queue_target
    v[P_ADD] = p.add_rate
    v[P_DADD] = p.deep_add_rate
    v[P_CXL] = p.cancel_rate
    v[P_MKT] = p.market_rate
    v[P_IN] = p.inside_rate
    v[P_ADDSZ] = p.add_size_mean
    v[P_CXLSZ] = p.cancel_size_mean
    v[P_MKTSZ] = p.market_size_mean
    v[P_BIGP] = p.market_big_prob
    v[P_BIGSZ] = p.market_big_mean
    v[P_NEWSZ] = p.new_level_size_mean
    v[P_REFILL] = p.refill_boost
    v[P_TAU] = p.signal_tau_s
    v[P_SVOL] = p.signal_vol
    v[P_KM] = p.signal_market_kappa
    v[P_KL] = p.signal_limit_kappa
    v[P_SKEW] = p.cancel_back_skew
    v[P_CONT] = p.continue_prob
    v[P_RMULT] = shock_mult
    v[P_RDECAY] = shock_decay
    v[P_RJUMP] = shock_jump
    v[P_CPOW] = p.cancel_queue_power
    return v


@njit(cache=True)
def _geom(mean: float) -> int:
    return int(np.random.geometric(1.0 / mean))


@njit(cache=True)
def _emit(
    buf: npt.NDArray[np.int64], n: int, ts: int, act: int, side: int, px: int, sz: int, qpos: int
) -> int:
    if n < buf.shape[0]:
        buf[n, B_TS] = ts
        buf[n, B_ACT] = act
        buf[n, B_SIDE] = side
        buf[n, B_PX] = px
        buf[n, B_SZ] = sz
        buf[n, B_QPOS] = qpos
    return n + 1


@njit(cache=True)
def _cancel_pos(q: int, c: int, skew: float) -> int:
    # Position (lots from the front) of the cancelled volume. u**skew with
    # skew < 1 piles mass near 1, so recent orders near the back cancel more
    # often, which matches the empirical pattern in limit order books.
    room = q - c
    if room <= 0:
        return 0
    return int(room * np.random.random() ** skew)


@njit(cache=True)
def _shift_bid_down(
    qb: npt.NDArray[np.int64],
    b: int,
    p: npt.NDArray[np.float64],
    buf: npt.NDArray[np.int64],
    n: int,
    ts: int,
) -> tuple[int, int]:
    """Best bid emptied: level 1 becomes the touch and a new far level enters the window."""
    L = qb.shape[0]
    for i in range(L - 1):
        qb[i] = qb[i + 1]
    new_sz = int(p[P_DQT] * (0.5 + np.random.random()))
    qb[L - 1] = new_sz
    b -= 1
    n = _emit(buf, n, ts, ADD, BID, b - (L - 1), new_sz, -1)
    return b, n


@njit(cache=True)
def _shift_ask_up(
    qa: npt.NDArray[np.int64],
    a: int,
    p: npt.NDArray[np.float64],
    buf: npt.NDArray[np.int64],
    n: int,
    ts: int,
) -> tuple[int, int]:
    L = qa.shape[0]
    for i in range(L - 1):
        qa[i] = qa[i + 1]
    new_sz = int(p[P_DQT] * (0.5 + np.random.random()))
    qa[L - 1] = new_sz
    a += 1
    n = _emit(buf, n, ts, ADD, ASK, a + (L - 1), new_sz, -1)
    return a, n


@njit(cache=True)
def _activity(
    t_ns: int, open_ns: int, close_ns: int, rel_ns: npt.NDArray[np.int64], mult: float, decay: float
) -> float:
    if t_ns < open_ns:
        base = 0.5
    else:
        since_open = (t_ns - open_ns) / NS
        to_close = max((close_ns - t_ns) / NS, 0.0)
        base = 1.0 + 0.9 * np.exp(-since_open / 1800.0) + 0.6 * np.exp(-to_close / 1800.0)
    boost = 1.0
    for j in range(rel_ns.shape[0]):
        if t_ns >= rel_ns[j]:
            age = (t_ns - rel_ns[j]) / NS
            if age < 10.0 * decay:
                boost += (mult - 1.0) * np.exp(-age / decay)
    return base * boost


@njit(cache=True)
def simulate_day(
    p: npt.NDArray[np.float64],
    levels: int,
    t0_ns: int,
    t1_ns: int,
    open_ns: int,
    close_ns: int,
    rel_ns: npt.NDArray[np.int64],
    rel_sign: npt.NDArray[np.float64],
    b0: int,
    seed: int,
    capacity: int,
) -> tuple[npt.NDArray[np.int64], int, int, npt.NDArray[np.float64]]:
    """Simulate one day. Returns (buffer, n_events, final_best_bid, signal_at_events).

    If n_events > capacity the buffer was too small; the caller retries bigger.
    """
    np.random.seed(seed)
    L = levels
    buf = np.zeros((capacity, 6), dtype=np.int64)
    sig = np.zeros(capacity, dtype=np.float64)
    qb = np.zeros(L, dtype=np.int64)
    qa = np.zeros(L, dtype=np.int64)
    b = b0
    a = b0 + 1
    n = 0
    # Initial book snapshot as a burst of adds at t0.
    for i in range(L):
        qb[i] = int(p[P_QT]) if i == 0 else int(p[P_DQT])
        qa[i] = int(p[P_QT]) if i == 0 else int(p[P_DQT])
        n = _emit(buf, n, t0_ns, ADD, BID, b - i, qb[i], -1)
        n = _emit(buf, n, t0_ns, ADD, ASK, a + i, qa[i], -1)

    sd = p[P_SVOL]
    x = sd * np.random.standard_normal()
    t_ns = t0_ns
    next_rel = 0
    last_dir = 0  # -1 after the bid emptied, +1 after the ask emptied
    rates = np.zeros(12)
    while True:
        A = _activity(t_ns, open_ns, close_ns, rel_ns, p[P_RMULT], p[P_RDECAY])
        qt = p[P_QT]
        dqt = p[P_DQT]
        em = np.exp(p[P_KM] * x)
        el = np.exp(p[P_KL] * x)
        spread = a - b
        rates[0] = p[P_MKT] * A * em  # market buy
        rates[1] = p[P_MKT] * A / em  # market sell
        rates[2] = p[P_ADD] * A * el * (1.0 + p[P_REFILL] * np.exp(-3.0 * qb[0] / qt))
        rates[3] = p[P_ADD] * A / el * (1.0 + p[P_REFILL] * np.exp(-3.0 * qa[0] / qt))
        rates[4] = p[P_CXL] * A * (qb[0] / qt) ** p[P_CPOW] / el
        rates[5] = p[P_CXL] * A * (qa[0] / qt) ** p[P_CPOW] * el
        rates[6] = p[P_DADD] * A * (L - 1)
        rates[7] = p[P_DADD] * A * (L - 1)
        sb = 0.0
        sa = 0.0
        for i in range(1, L):
            sb += qb[i]
            sa += qa[i]
        rates[8] = p[P_DADD] * A * sb / dqt
        rates[9] = p[P_DADD] * A * sa / dqt
        if spread >= 2:
            cont = p[P_CONT]
            # After the bid emptied (price falling) the vacated level is more
            # likely to be taken by a seller, continuing the move.
            w_ask = cont if last_dir == -1 else (1.0 - cont if last_dir == 1 else 0.5)
            rates[10] = p[P_IN] * A * 2.0 * (1.0 - w_ask) * el
            rates[11] = p[P_IN] * A * 2.0 * w_ask / el
        else:
            rates[10] = 0.0
            rates[11] = 0.0
        R = 0.0
        for k in range(12):
            R += rates[k]
        dt_s = np.random.exponential(1.0 / R)
        t_ns = t_ns + int(dt_s * NS) + 1
        if t_ns >= t1_ns:
            break
        # Signal: exact OU transition over dt, plus jumps at scheduled releases.
        decay = np.exp(-dt_s / p[P_TAU])
        x = x * decay + sd * np.sqrt(1.0 - decay * decay) * np.random.standard_normal()
        while next_rel < rel_ns.shape[0] and t_ns >= rel_ns[next_rel]:
            x += rel_sign[next_rel] * p[P_RJUMP] * sd
            next_rel += 1

        u = np.random.random() * R
        ev = 0
        acc = rates[0]
        while u > acc and ev < 11:
            ev += 1
            acc += rates[ev]

        start_n = n
        if ev == 0 or ev == 1:
            big = np.random.random() < p[P_BIGP]
            v = _geom(p[P_BIGSZ]) if big else _geom(p[P_MKTSZ])
            if ev == 0:  # buyer sweeps asks
                while v > 0:
                    take = min(v, qa[0])
                    n = _emit(buf, n, t_ns, TRADE, BID, a, take, 0)
                    qa[0] -= take
                    v -= take
                    if qa[0] == 0:
                        a, n = _shift_ask_up(qa, a, p, buf, n, t_ns)
                        last_dir = 1
                        while qa[0] == 0:
                            a, n = _shift_ask_up(qa, a, p, buf, n, t_ns)
            else:
                while v > 0:
                    take = min(v, qb[0])
                    n = _emit(buf, n, t_ns, TRADE, ASK, b, take, 0)
                    qb[0] -= take
                    v -= take
                    if qb[0] == 0:
                        b, n = _shift_bid_down(qb, b, p, buf, n, t_ns)
                        last_dir = -1
                        while qb[0] == 0:
                            b, n = _shift_bid_down(qb, b, p, buf, n, t_ns)
        elif ev == 2:
            s = _geom(p[P_ADDSZ])
            qb[0] += s
            n = _emit(buf, n, t_ns, ADD, BID, b, s, -1)
        elif ev == 3:
            s = _geom(p[P_ADDSZ])
            qa[0] += s
            n = _emit(buf, n, t_ns, ADD, ASK, a, s, -1)
        elif ev == 4:
            c = min(_geom(p[P_CXLSZ]), qb[0])
            n = _emit(buf, n, t_ns, CANCEL, BID, b, c, _cancel_pos(qb[0], c, p[P_SKEW]))
            qb[0] -= c
            if qb[0] == 0:
                b, n = _shift_bid_down(qb, b, p, buf, n, t_ns)
                last_dir = -1
                while qb[0] == 0:
                    b, n = _shift_bid_down(qb, b, p, buf, n, t_ns)
        elif ev == 5:
            c = min(_geom(p[P_CXLSZ]), qa[0])
            n = _emit(buf, n, t_ns, CANCEL, ASK, a, c, _cancel_pos(qa[0], c, p[P_SKEW]))
            qa[0] -= c
            if qa[0] == 0:
                a, n = _shift_ask_up(qa, a, p, buf, n, t_ns)
                last_dir = 1
                while qa[0] == 0:
                    a, n = _shift_ask_up(qa, a, p, buf, n, t_ns)
        elif ev == 6 or ev == 7:
            i = 1 + int(np.random.random() * (L - 1))
            s = _geom(p[P_ADDSZ])
            if ev == 6:
                qb[i] += s
                n = _emit(buf, n, t_ns, ADD, BID, b - i, s, -1)
            else:
                qa[i] += s
                n = _emit(buf, n, t_ns, ADD, ASK, a + i, s, -1)
        elif ev == 8 or ev == 9:
            q = qb if ev == 8 else qa
            tot = 0.0
            for i in range(1, L):
                tot += q[i]
            if tot > 0:
                w = np.random.random() * tot
                i = 1
                accq = float(q[1])
                while w > accq and i < L - 1:
                    i += 1
                    accq += q[i]
                if q[i] > 0:
                    c = min(_geom(p[P_CXLSZ]), q[i])
                    qp = _cancel_pos(q[i], c, p[P_SKEW])
                    if ev == 8:
                        n = _emit(buf, n, t_ns, CANCEL, BID, b - i, c, qp)
                    else:
                        n = _emit(buf, n, t_ns, CANCEL, ASK, a + i, c, qp)
                    q[i] -= c
        elif ev == 10:
            # Improve the bid into a 2+ tick spread: a fresh, small level.
            s = _geom(p[P_NEWSZ])
            out = qb[L - 1]
            if out > 0:
                n = _emit(buf, n, t_ns, CANCEL, BID, b - (L - 1), out, 0)
            for i in range(L - 1, 0, -1):
                qb[i] = qb[i - 1]
            b += 1
            qb[0] = s
            n = _emit(buf, n, t_ns, ADD, BID, b, s, -1)
        elif ev == 11:
            s = _geom(p[P_NEWSZ])
            out = qa[L - 1]
            if out > 0:
                n = _emit(buf, n, t_ns, CANCEL, ASK, a + (L - 1), out, 0)
            for i in range(L - 1, 0, -1):
                qa[i] = qa[i - 1]
            a -= 1
            qa[0] = s
            n = _emit(buf, n, t_ns, ADD, ASK, a, s, -1)
        for k in range(start_n, min(n, capacity)):
            sig[k] = x
    return buf, n, b, sig


@dataclass(frozen=True)
class SyntheticDay:
    instrument: str
    date: dt.date
    events: pl.DataFrame
    instrument_id: int
    raw_symbol: str
    releases: list[tuple[dt.datetime, str]]
    final_mid_ticks: float
    fault_counts: dict[str, int]


def trading_days(start: str, n: int, holidays: list[str]) -> list[dt.date]:
    out: list[dt.date] = []
    d = dt.date.fromisoformat(start)
    skip = {dt.date.fromisoformat(h) for h in holidays}
    while len(out) < n:
        if d.weekday() < 5 and d not in skip:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def _ns(date: dt.date, hhmm: str, tz: ZoneInfo) -> int:
    h, m = (int(x) for x in hhmm.split(":"))
    local = dt.datetime(date.year, date.month, date.day, h, m, tzinfo=tz)
    return int(local.timestamp()) * NS


def _seed(base: int, instrument: str, date: dt.date) -> int:
    digest = hashlib.sha256(f"{base}:{instrument}:{date.isoformat()}".encode()).hexdigest()
    return int(digest[:8], 16)


def front_contract_index(cfg: Settings, day_index: int) -> int:
    """Which of the two synthetic contracts is front by volume on this day."""
    r = cfg.synthetic.roll
    return 0 if day_index < r.first_roll_day_index + (r.roll_days + 1) // 2 else 1


def volume_shares(cfg: Settings, day_index: int) -> tuple[float, float]:
    """Synthetic daily volume split between the expiring and the next contract."""
    r = cfg.synthetic.roll
    k = day_index - r.first_roll_day_index
    if k < 0:
        old = 0.97
    elif k < r.roll_days:
        old = 0.75 - 0.5 * k / max(r.roll_days - 1, 1)
    else:
        old = 0.03
    return old, 1.0 - old


def generate_day(
    cfg: Settings, instrument: str, day_index: int, date: dt.date, start_bid_ticks: int
) -> SyntheticDay:
    sp = cfg.synthetic.instruments[instrument]
    tz = ZoneInfo(cfg.session.timezone)
    shock = cfg.synthetic.release_shock
    p = params_vector(sp, shock.activity_multiplier, shock.decay_seconds, shock.signal_jump)
    t0 = _ns(date, cfg.synthetic.generate_from, tz)
    t1 = _ns(date, cfg.synthetic.generate_to, tz)
    open_ns = _ns(date, cfg.session.start, tz)
    close_ns = _ns(date, cfg.session.end, tz)
    seed = _seed(cfg.synthetic.seed, instrument, date)
    rng = np.random.default_rng(seed)

    releases = [r for r in cfg.synthetic.releases if r.day_index == day_index]
    rel_ns = np.array([_ns(date, r.time, tz) for r in releases], dtype=np.int64)
    rel_sign = rng.choice([-1.0, 1.0], size=len(releases)).astype(np.float64)
    release_list = [
        (dt.datetime.fromtimestamp(int(ns) // NS, tz), r.event)
        for ns, r in zip(rel_ns, releases, strict=True)
    ]

    capacity = 3_000_000
    while True:
        buf, n, final_bid, sig = simulate_day(
            p,
            sp.levels,
            t0,
            t1,
            open_ns,
            close_ns,
            rel_ns,
            rel_sign,
            start_bid_ticks,
            seed,
            capacity,
        )
        if n <= capacity:
            break
        capacity = int(n * 1.2)
    buf = buf[:n]
    sig = sig[:n]

    ts = buf[:, B_TS].copy()
    # Feed outage: shift every later timestamp so the stream goes silent for a
    # few seconds without losing any deltas (the book stays consistent).
    faults = {"duplicates": 0, "sequence_gaps": 0, "outage_seconds": 0}
    qf = cfg.synthetic.quality_faults
    if rng.random() < qf.outage_probability_per_day:
        in_session = np.flatnonzero((ts > open_ns) & (ts < close_ns - 600 * NS))
        if in_session.size:
            k = int(rng.choice(in_session))
            gap = float(rng.uniform(*qf.outage_seconds))
            ts[k:] += int(gap * NS)
            faults["outage_seconds"] = round(gap, 3)

    seq = np.arange(n, dtype=np.int64) + 1
    gaps = rng.random(n) < qf.sequence_gap_rate
    seq = seq + np.cumsum(gaps)
    faults["sequence_gaps"] = int(gaps.sum())

    contracts = sp.instrument_ids
    front = front_contract_index(cfg, day_index)
    events = pl.DataFrame(
        {
            "ts_event": ts,
            "sequence": seq,
            "instrument_id": np.full(n, contracts[front], dtype=np.int32),
            "action": buf[:, B_ACT].astype(np.int8),
            "side": buf[:, B_SIDE].astype(np.int8),
            "price": buf[:, B_PX],
            "size": buf[:, B_SZ],
            "queue_pos": buf[:, B_QPOS],
            "signal_x": sig,  # hidden state, kept only for diagnostics; never a model input
        }
    )
    dup_mask = rng.random(n) < qf.duplicate_rate
    faults["duplicates"] = int(dup_mask.sum())
    if faults["duplicates"]:
        events = pl.concat([events, events.filter(pl.Series(dup_mask))]).sort(
            ["ts_event", "sequence"], maintain_order=True
        )
    final_mid = final_bid + 0.5
    return SyntheticDay(
        instrument=instrument,
        date=date,
        events=events,
        instrument_id=int(contracts[front]),
        raw_symbol=sp.raw_symbols[front],
        releases=release_list,
        final_mid_ticks=final_mid,
        fault_counts=faults,
    )


def generate_dataset(cfg: Settings, instrument: str, n_days: int | None = None) -> list[Path]:
    """Generate and store synthetic days, definitions, daily volume and releases."""
    sp = cfg.synthetic.instruments[instrument]
    spec = cfg.instrument(instrument)
    days = trading_days(cfg.synthetic.start_date, n_days or sp.n_days, cfg.synthetic.holidays)
    root = cfg.paths.processed / instrument
    root.mkdir(parents=True, exist_ok=True)
    bid = round(sp.start_price / spec.tick_size)
    written: list[Path] = []
    volume_rows = []
    release_rows = []
    for i, date in enumerate(days):
        day = generate_day(cfg, instrument, i, date, bid)
        bid = int(day.final_mid_ticks - 0.5)
        out = root / date.isoformat()
        out.mkdir(parents=True, exist_ok=True)
        day.events.write_parquet(out / "events.parquet")
        meta = {
            "instrument": instrument,
            "date": date.isoformat(),
            "source": "synthetic",
            "instrument_id": day.instrument_id,
            "raw_symbol": day.raw_symbol,
            "n_events": day.events.height,
            "faults_injected": day.fault_counts,
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        traded = int(day.events.filter(pl.col("action") == TRADE)["size"].sum())
        old, new = volume_shares(cfg, i)
        for cid, sym, share in zip(sp.instrument_ids, sp.raw_symbols, (old, new), strict=True):
            volume_rows.append(
                {
                    "date": date,
                    "instrument_id": cid,
                    "raw_symbol": sym,
                    "volume": int(traded * share),
                }
            )
        for when, name in day.releases:
            release_rows.append(
                {"date": date.isoformat(), "time_et": when.strftime("%H:%M"), "event": name}
            )
        written.append(out / "events.parquet")
        log.info(
            "synthetic_day_written",
            instrument=instrument,
            date=date.isoformat(),
            events=day.events.height,
            faults=day.fault_counts,
        )

    pl.DataFrame(volume_rows).write_parquet(root / "daily_volume.parquet")
    definitions = [
        {
            "instrument_id": cid,
            "raw_symbol": sym,
            "min_price_increment": spec.tick_size,
            "contract_multiplier": spec.contract_multiplier,
            "source": "synthetic",
        }
        for cid, sym in zip(sp.instrument_ids, sp.raw_symbols, strict=True)
    ]
    (root / "definitions.json").write_text(json.dumps(definitions, indent=2))
    if release_rows:
        cal = cfg.paths.calendar
        cal.mkdir(parents=True, exist_ok=True)
        path = cal / "releases_synthetic.csv"
        existing = (
            pl.read_csv(path)
            if path.exists()
            else pl.DataFrame(schema={"date": pl.String, "time_et": pl.String, "event": pl.String})
        )
        pl.concat([existing, pl.DataFrame(release_rows)]).unique().sort(
            ["date", "time_et"]
        ).write_csv(path)
    return written
