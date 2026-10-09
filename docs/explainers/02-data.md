# Data layer (`data/`)

## What it does

Turns raw market data into clean, validated days, one Parquet file per instrument per day.

| File | Job |
|---|---|
| `databento_io.py` | Cost-checked downloads and conversion of Databento mbp records to the internal book format |
| `sessions.py` | U.S. day session in New York time, converted to UTC correctly across daylight-saving changes; macro-release windows |
| `roll.py` | Finds roll days from daily volume by contract |
| `quality.py` | Counts and reports data problems; removes exact duplicates; restores sequence order |
| `pipeline.py` | Runs everything for one day and writes `features.parquet` |

## Protecting the Databento budget

Three guards stand between the code and the credit card:

1. **Cache first.** Each day/schema is one `.dbn.zst` file under `data/raw/`. If it exists, nothing is requested.
2. **Quote before buying.** `metadata.get_cost` is called and logged before every download.
3. **A spend ledger.** Every purchase is appended to `data/raw/spend_ledger.json`. A request that would push cumulative spend past `databento.budget_usd` raises `BudgetExceededError`. And nothing is bought at all unless `--yes` is passed; without it the command is a dry run that only prints the price.

These are unit-tested with a fake client, so the logic is checked without a key or network.

## Daylight saving

The session is 08:30-15:00 New York time. New York is UTC-4 in summer and UTC-5 in winter, so a hard-coded offset would shift the session by an hour for part of the year. `zoneinfo` converts each date's local open and close to UTC, and a test checks dates on both sides of the November change.

## Roll weeks

Futures roll quarterly. For a few days, volume moves from the expiring contract to the next one, and "front by volume" can flip between them. A day counts as a roll day when the front contract holds less than 85% of volume, or the front contract changed within one day. This works from the data alone, as the spec requires, rather than from a hard-coded calendar.

## Data quality

The checks: duplicate records, sequence gaps, timestamps out of order, zero or negative sizes, crossed or locked books, and gaps longer than 2 seconds inside the session. Every count is logged and written to `reports/data_quality/`. Only two repairs are automatic, and both are counted: removing exact duplicates and restoring sequence order. A day with too many crossed books, or prices outside the book's array, raises an error instead of being analysed quietly.

## Storage

Features are stored as float32 and sizes as int32 to halve disk use. Best bid/ask prices stay int64 because the "empty side" sentinel does not fit in 32 bits. Everything is cast back to 64-bit before the numba kernels run.

## What could go wrong

* Databento's real mbp records have details the synthetic path never exercises (for example implied prices from spreads, or `F` fill records). The converter maps them, but it has only been tested on hand-built records.
* A release calendar must be maintained by hand for real data (`data/calendar/releases.csv`).

## Alternatives considered

* **Buying MBO first**: better (true queue positions), but more expensive per day; the plan was to quote both and decide.
* **Hard-coded roll dates**: simpler, but the spec asked for detection from data, and calendars drift.
