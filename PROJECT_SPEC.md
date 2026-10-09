# Project Spec: Queue-Imbalance Execution in U.S. Treasury Futures

## Amendments (agreed 2026-10-07)

The original spec below is kept as written. These decisions change how parts of it are carried out:

1. **Synthetic data instead of Databento.** No Databento key was available (signup requires a card). The project runs on a synthetic queue-reactive order-book generator (`src/halftick/data/synthetic.py`). The Databento downloader, budget guard and converter are still implemented and unit-tested against a fake client, so real data is a drop-in change. Every result is labelled as synthetic.
2. **MBO-style truth first.** The generator records the true queue position of every cancellation, which plays the role of MBO data. The fill simulator therefore has a fourth queue model, `exact`, and the three assumed models (pessimistic, proportional, optimistic) are scored against it.
3. **A cost-to-go model.** Besides the direction model, a LightGBM regression predicts "cost if I keep waiting minus cost if I cross now" at checkpoints along each resting order. This is the main model-based strategy (`model_cost`). The direction-threshold strategy from §7 is kept as `model_direction`.
4. **Fees out of the sensitivity grid.** Every strategy trades exactly one contract per order and pays exactly one fee, so fees cancel out of all savings. The fee table shows this explicitly instead of a fee dimension in the grid.
5. **An early rough headline.** The pipeline produces a first savings number from a small run before the full grid.
6. **A "join fresh levels" strategy** (`new_level`): wait for a new price level to open on our side and join it near the front of the queue, otherwise cross at the deadline.
7. **Day counts.** ZN uses 22 synthetic days (roll days are detected from volume and excluded). ES uses 11 days, limited by free disk space on the development machine. Each instrument rolls on its own synthetic date (ZN late August, ES mid September), as in the real calendar.

---

For Claude Code: This file is the source of truth for the project. Read it fully before writing code. Work one phase at a time, stop at the end of each phase, summarise what you built, and wait for my review before starting the next phase. Never invent results, numbers, or charts. Every number in the report must come from code that ran on real data in this repo. Use only the technologies listed in §14 (Tech Stack); if you think something else is needed, ask me first and explain why.

## 1. Goal

Build a research and simulation system that answers one practical execution question:

"I need to buy (or sell) ZN futures in the next few seconds. Should I cross the spread with a market order now, or post a limit order and wait?"

The answer comes from a calibrated, short-horizon price-direction signal built from the order book, evaluated inside a realistic, queue-aware execution simulator.

Primary instrument: ZN (CBOT 10-Year U.S. Treasury Note futures), front contract. Comparison instrument: ES (CME E-mini S&P 500 futures), to show whether results generalise.

### Why this matters (context for design decisions)

- ZN and ES are large-tick markets: the bid-ask spread is one tick almost all the time and the queues at the best bid/ask are very deep. In such markets, the relative size of the bid and ask queues (queue imbalance) is known to carry information about the next price change.
- A limit order saves the spread but risks (a) not getting filled and (b) adverse selection: getting filled precisely when the price is about to move against you. A good model must account for both.
- The deliverable should be something a trading desk could actually use: honest costs, honest failure modes.

### Headline result the project should produce

"On ZN, using the model to choose between posting and crossing reduces execution cost by X ticks per contract versus always crossing the spread (and by Y vs. always posting), after fees and latency. The edge is strongest when [condition] and disappears when [condition]."

X and Y are unknown until measured. If the edge is zero or negative, report that honestly and explain why.

## 2. Background literature (implement ideas from these; cite them in the report)

- Gould & Bonart (2016), Queue Imbalance as a One-Tick-Ahead Price Predictor in a Limit Order Book.
- Cont, Kukanov & Stoikov (2014), The Price Impact of Order Book Events (Order Flow Imbalance, OFI).
- Stoikov (2018), The Micro-Price: A High-Frequency Estimator of Future Prices.

## 3. Data

### Source

Databento, dataset GLBX.MDP3 (CME Globex, includes CBOT). Python client: databento. New accounts receive $125 in free historical-data credits (expire after 6 months). Budget is tight; protect it.

### Rules for spending credits

- Always call `client.metadata.get_cost(...)` before any download and print the cost. Never download if the cumulative spend would exceed the budget in config.yaml (default: $100).
- Cache every download to `data/raw/` as `.dbn.zst` and never re-download a file that exists locally.
- Store the API key in `.env` as `DATABENTO_API_KEY`. `.env` must be in `.gitignore`. Never print or commit the key.

### Schemas

- mbp-1 (top of book + trades): main dataset. Cheap; pull ~15-20 trading days for ZN and ~5 days for ES.
- mbp-10 (10 levels of depth): pull only ~3-5 days for ZN, for deeper-book features. Check cost first.
- definition: instrument metadata (tick size, contract symbol, expiry).
- Optional, only if budget allows: mbo (market-by-order) for 1-2 days to validate the queue-position assumptions in the simulator.

### Symbology

- Use `stype_in="continuous"` with `ZN.v.0` / `ES.v.0` (front contract by volume), but record the raw contract symbol and instrument_id for every row.
- Exclude roll weeks (when volume is shifting between quarterly contracts) from the main analysis. Detect them from the data, not hard-coded dates.

### Session filter

- Main analysis: U.S. day session, 08:30-15:00 America/New_York (convert from UTC ts_event; handle DST correctly).
- Flag (do not delete) windows of ±5 minutes around scheduled U.S. macro releases (CPI, NFP, FOMC, etc.). Keep a small hand-maintained CSV `data/calendar/releases.csv` (date, time ET, event). Analyse these windows separately.

### Contract facts (put in config.yaml, verify against the definition schema)

- ZN tick size = 1/2 of 1/32 of a point (0.015625), $15.625 per tick.
- ES tick size = 0.25 index points, $12.50 per tick.
- Fees: configurable, default $2.50 per contract per side all-in (exchange + clearing + broker). Run sensitivity at $1.00 and $4.00.

### Data quality checks (fail loudly, log counts)

Out-of-order timestamps, sequence gaps, crossed or locked books, zero/negative sizes, gaps > 2 s in the day session, duplicate records. Write a per-day data quality report to `reports/data_quality/`.

## 4. Features (computed on every book update, event time)

All features must use only information available at or before the event timestamp. No look-ahead.

| Feature | Definition |
|---|---|
| Queue imbalance I | (bid_sz - ask_sz) / (bid_sz + ask_sz) at best level |
| Depth imbalance (k levels) | Same, summed over levels 1..k (k = 3, 5, 10; mbp-10 days only) |
| Microprice | (bid_px·ask_sz + ask_px·bid_sz) / (bid_sz + ask_sz) - mid, in ticks |
| OFI | Cont-Kukanov-Stoikov order flow imbalance over last N events and last T ms |
| Trade-flow imbalance | (buy volume - sell volume) / total over last T ms (use aggressor side) |
| Queue depletion rate | Rate of decrease of best bid / best ask size over last T ms |
| Spread | In ticks (mostly 1; keep the rare wide-spread states as a flag) |
| Time since last mid change | ms |
| Short-term volatility | Mid changes per second over last 30 s / 5 min |
| Time of day | Minutes since 08:30 ET (for diagnostics, not necessarily as a model input) |

Windows T: 100 ms, 1 s, 5 s. N: 10, 50, 200 events.

## 5. Targets

- Primary (event time): direction of the next mid-price change: up (1) or down (0).
- Secondary (clock time): sign and size (in ticks) of mid-price change over the next 1 s, 5 s, 30 s.

## 6. Models

- Baseline A, empirical lookup: bin queue imbalance into deciles; predict P(up) = historical frequency in each bin. This is the Gould-Bonart benchmark.
- Baseline B: logistic regression on queue imbalance + microprice + OFI.
- Main model: LightGBM on the full feature set.
- Calibrate model outputs (isotonic or Platt) on a held-out calibration fold.

### Validation (strict)

- Walk-forward by day: train on days 1..k, calibrate on day k+1, test on day k+2; roll forward. Never shuffle across time.
- Metrics: log loss, Brier score, AUC, accuracy, and a reliability diagram per model.
- Report whether LightGBM actually beats the baselines out of sample. If it does not, say so. A simple model that is nearly as good is a valid and valuable finding.
- Include a no-look-ahead unit test: features at time t must not change if all data after t is deleted.

## 7. Queue-Aware Execution Simulator (the core of the project)

### The task being simulated

"Buy (or sell) 1 ZN contract within a deadline of D seconds (D = 5, 15, 30)." Decision moments are sampled at random times during the day session (e.g., several thousand per day), with buy/sell chosen at random.

### Strategies to compare

- Always cross: market order immediately (pay the spread).
- Always post: limit order at the near touch; if not filled by the deadline, cross.
- Model-based: at each book update, compute expected cost of crossing now vs. continuing to wait, using the calibrated P(adverse move) and the estimated fill probability. Cross when crossing is cheaper; otherwise keep the limit order. Cross at the deadline if unfilled.
- Oracle (upper bound, clearly labelled): uses future information. Only for showing the theoretical ceiling.

### Fill model (must be realistic; this is what separates the project from naive backtests)

- A limit order joins the back of the queue at its price. Queue ahead = displayed size at that level at the moment of entry.
- Queue ahead decreases with trades at that price.
- Cancellations at that level are handled under a configurable assumption, and results must be reported for all three:
  - pessimistic: cancellations come from behind us (queue ahead only shrinks via trades),
  - proportional: cancellations reduce queue ahead pro-rata,
  - optimistic: cancellations come from ahead of us first.
- The order fills when queue ahead reaches zero and a trade occurs at our price. If the price moves away (level disappears), the order is no longer at the touch; handle re-pegging explicitly as an option (default: no re-peg; cross at deadline).
- Latency: every action takes effect after a configurable delay (0 ms, 1 ms, 10 ms, 100 ms). The book used for the fill is the book after the delay.
- If mbo data was downloaded, validate the queue model against true queue positions on those days and report the error.

### Cost measurement

- Cost per contract, in ticks and dollars, measured against the mid-price at the decision moment (arrival price). Include fees.
- Also report: fill rate, average time to fill, adverse selection (mid move 1 s / 5 s after a passive fill), and the distribution of costs (not just the mean).
- Every simulated order and fill is logged to a trade log (Parquet) with: decision time, side, strategy, prices, queue position at entry, fill time, fill price, cost, fees.

### Sensitivity analysis (required)

Results must be shown across: cancellation assumption × latency × fees × deadline. Present the main result as a table plus one or two clear charts.

## 8. Diagnostics (where does it work, where does it fail?)

Break down model accuracy and execution savings by:

- time of day (open, midday, close),
- volatility regime (terciles),
- queue size (small vs. large queues),
- spread state (1 tick vs. wider),
- macro release windows vs. normal periods,
- ZN vs. ES.

Also show signal decay: predictive power vs. prediction horizon, and stability of results across days.

## 9. Infrastructure

### Event-driven replay engine

- A single ReplayEngine that streams historical records through the same interfaces a live feed would use: on_book_update, on_trade. Strategies and the simulator subscribe to these.
- Design a MarketDataSource interface with a DatabentoHistoricalSource implementation and a SyntheticSource (for tests). The design should make a future DatabentoLiveSource a drop-in addition (do not implement live).

### Monitoring

- Expose Prometheus metrics from the replay engine: events processed/sec, replay lag vs. wall clock (in real-time replay mode), data-quality error counters, simulated orders/fills, running P&L and cost per strategy, latency of feature computation.
- docker-compose.yml with three services: app, prometheus, grafana. Provision a Grafana dashboard (JSON in repo) automatically.
- Prometheus alert rules (in repo): no data for > 5 s during session, crossed book detected, feature computation latency above threshold.
- A `--realtime` replay mode (1× speed) so the dashboard can be demonstrated live.

### Engineering standards

- Python 3.11+, type hints throughout, pyproject.toml, ruff for lint/format, pytest for tests, mypy on core modules.
- GitHub Actions CI: lint + tests on every push. Tests use SyntheticSource only; CI must never need an API key or network.
- Structured logging (JSON) with log levels; no bare print in library code.
- Config via config.yaml (+ CLI overrides). No magic numbers in code.
- Small, focused commits with clear messages.

### Required unit tests (at minimum)

- Order book state updates correctly for each action type (add, modify, cancel, trade, clear).
- Feature correctness on hand-built examples (imbalance, microprice, OFI).
- No-look-ahead test for all features and targets.
- Queue simulator: hand-crafted scenarios where the correct fill time is known for each cancellation assumption.
- Cost accounting: crossing and posting costs computed correctly, including fees and the deadline cross.
- Walk-forward splitter never leaks future days into training.

## 10. Repository structure

```
zn-execution/
├── README.md                  # problem, results summary, how to run, key charts
├── PROJECT_SPEC.md            # this file
├── config.yaml
├── pyproject.toml
├── uv.lock
├── Dockerfile
├── docker-compose.yml
├── .pre-commit-config.yaml
├── .env.example               # DATABENTO_API_KEY=
├── .github/workflows/ci.yml
├── src/znexec/
│   ├── data/                  # download (cost-checked), caching, quality checks, sessions, calendar
│   ├── book/                  # order book state
│   ├── replay/                # ReplayEngine, MarketDataSource, Historical + Synthetic sources
│   ├── features/
│   ├── targets/
│   ├── models/                # baselines, LightGBM, calibration, walk-forward
│   ├── sim/                   # queue-aware fill model, strategies, cost accounting, trade log
│   ├── diagnostics/
│   ├── monitoring/            # Prometheus metrics
│   └── cli.py                 # download / build-features / train / simulate / report / replay
├── tests/
├── monitoring/                # prometheus.yml, alert rules, grafana dashboards
├── notebooks/                 # exploration only; final results come from CLI scripts
├── reports/                   # generated figures, tables, REPORT.md
└── docs/
    └── explainers/            # one plain-English explainer per module (see §12)
```

(Built as `Halftick` with the package at `src/halftick/`.)

## 11. Phases (stop after each one for my review)

- **Phase 0, Skeleton.** Repo structure, pyproject.toml, config, CI, logging, .env handling, SyntheticSource that generates a realistic-looking large-tick order book with trades. CI green. Done when: pytest passes and CI runs without network access.
- **Phase 1, Data.** Cost-checked Databento downloader with caching and budget guard, definition lookup, session filter, roll-week detection, data-quality report. First pull: 2 days of ZN mbp-1 only (show me the cost before downloading). Done when: I can run one CLI command to download, validate and store a day, and read the quality report.
- **Phase 2, Book, replay, features, targets.** Order book state, ReplayEngine, all features in §4, targets in §5, no-look-ahead tests. Basic descriptive stats: how often the spread is 1 tick, queue size distributions, mid-change frequency. Done when: features for a full day compute in reasonable time and all tests pass.
- **Phase 3, Models.** Baselines A and B, LightGBM, calibration, walk-forward evaluation, reliability diagrams. Then download the remaining ZN days (cost-checked). Done when: a results table and reliability plots exist for all three models.
- **Phase 4, Execution simulator.** Queue-aware fill model, four strategies, cost accounting, trade log, full sensitivity grid. Done when: the headline cost-savings table exists across all sensitivity settings.
- **Phase 5, Diagnostics + ES comparison.** §8 breakdowns, signal decay, macro-release windows, ES results.
- **Phase 6, Monitoring.** Prometheus metrics, docker-compose, provisioned Grafana dashboard, alert rules, `--realtime` replay mode. Done when: `docker compose up` shows a live dashboard during a replay.
- **Phase 7, Report.** `reports/REPORT.md` (2-4 pages + figures): question, data, method, results, where it fails, limitations, what I would do with more data (e.g., MBO, more days, live deployment). Polished README.md with the headline result and one key chart at the top.

## 12. Learning requirement (important)

I need to explain every part of this project in an interview. For each module, write a short plain-English explainer in `docs/explainers/` covering: what it does, why it is designed this way, the key assumptions, and what could go wrong. Add brief comments in code where the reasoning is not obvious. When you make a design choice, tell me the alternatives you considered.

## 13. Honesty rules

- No fabricated numbers, charts, or citations. If something could not be computed, say so.
- Clearly separate in-sample from out-of-sample results.
- The oracle strategy must always be labelled as an upper bound using future information.
- State the main limitations prominently: small number of days, queue-position assumptions without MBO data, no real fills, simulated latency, historical fee assumptions.

## 14. Tech Stack (use exactly these; ask before adding anything)

Use the latest stable version of each library, pinned in pyproject.toml / uv.lock.

### 14.1 Language and environment

| Technology | Used for | Why |
|---|---|---|
| Python 3.12 (3.11 minimum) | All code | Industry standard for quant research; the job post asks for strong Python |
| uv | Virtual environment, dependency install, lockfile (uv.lock), running commands (uv run pytest) | Fast and reproducible; one tool instead of pip + venv + pip-tools |
| pyproject.toml | Project metadata, dependencies, tool config (ruff, mypy, pytest) | Single standard config file |
| Git + GitHub | Version control, public portfolio repo | The job post requires Git and collaborative workflow |

OS note: develop on Linux or macOS. On Windows, use WSL2 (Ubuntu) for everything, including Docker.

### 14.2 Market data

| Technology | Used for | Why |
|---|---|---|
| Databento Python client (databento) | Historical CME data (GLBX.MDP3): mbp-1, mbp-10, definition, optional mbo. Use Historical.metadata.get_cost before every request and Historical.timeseries.get_range to download | Institutional-grade CME order-book data with free starter credits |
| DBN files (.dbn.zst) | Raw cache of every download in data/raw/ | Databento's native compressed binary format; never pay twice for the same data |
| python-dotenv | Load DATABENTO_API_KEY from .env | Keeps secrets out of code and Git |

### 14.3 Data processing and storage

| Technology | Used for | Why |
|---|---|---|
| Polars | Main dataframe library: cleaning, session filtering, feature tables, joins, group-by diagnostics | Much faster than pandas on millions of rows; lazy queries; low memory |
| NumPy | Arrays for the order-book and simulator hot paths | Fast numeric core |
| Numba | JIT-compile the per-event loops: order-book updates, OFI, queue depletion, queue-aware fill simulation | Event-by-event loops cannot be fully vectorised; Numba makes them run at near-C speed in Python |
| PyArrow + Parquet | Storage of cleaned data, features, targets, trade logs (data/processed/, partitioned by instrument and date) | Columnar, compressed, fast, works with Polars and DuckDB |
| DuckDB | SQL queries across many Parquet files for analysis and the report | Query a month of data without loading it all into memory |
| pandas | Only where a library requires it (e.g. statsmodels, some plotting) | Convert from Polars at the boundary; not the main tool |
| zoneinfo (standard library) | UTC to America/New_York conversion with correct DST | No extra dependency; correct DST handling |

Performance rules: never iterate over dataframe rows in Python. Use Polars expressions for vectorisable work, Numba @njit functions over NumPy arrays for sequential event logic. Target: one full ZN day of mbp-1 processed into features in under a minute on a laptop.

### 14.4 Statistics and machine learning

| Technology | Used for | Why |
|---|---|---|
| scikit-learn | Logistic regression (Baseline B), CalibratedClassifierCV / isotonic regression, metrics (log loss, Brier, AUC), reliability curves (calibration_curve) | Standard, well-tested ML toolkit |
| LightGBM | Main model | Fast gradient boosting, strong on tabular features, handles large row counts |
| statsmodels | Statistical tests and regressions with robust (HAC / Newey-West) standard errors for diagnostics; confidence intervals on cost savings | Time-series data has autocorrelation; naive standard errors overstate significance |
| SciPy | Bootstrap confidence intervals, distribution tests | Quantify uncertainty in the headline result |
| Custom walk-forward splitter (written in src/znexec/models/) | Day-based train / calibrate / test splits | sklearn's splitters don't do this exact scheme; must be tested for leakage |

No deep learning (PyTorch/TensorFlow): not justified for this data size and adds complexity without clear benefit.

### 14.5 Visualisation and reporting

| Technology | Used for | Why |
|---|---|---|
| Matplotlib | All report figures saved to reports/figures/ as PNG: reliability diagrams, cost distributions, sensitivity heatmaps, signal decay, diagnostics | Full control, static figures that render on GitHub |
| JupyterLab | Exploration only (notebooks/) | Final results must come from CLI commands, not notebooks |
| Markdown | README.md, reports/REPORT.md, docs/explainers/ | Renders on GitHub; easy for reviewers |

### 14.6 Application structure

| Technology | Used for | Why |
|---|---|---|
| Typer | CLI: znexec download, build-features, train, simulate, report, replay | Clean, typed command-line interface |
| Pydantic v2 + PyYAML | Load and validate config.yaml into typed settings objects | Bad config fails immediately with a clear error |
| structlog | Structured JSON logging | Logs that are searchable and machine-readable, like a production system |
| Python dataclasses / typing.Protocol | MarketDataSource interface, book state, order and fill records | Clear interfaces; easy to add a live source later |

### 14.7 Monitoring and deployment

| Technology | Used for | Why |
|---|---|---|
| prometheus-client (Python) | Expose metrics from the replay engine on an HTTP /metrics endpoint: counters, gauges, histograms (events/sec, lag, data-quality errors, orders, fills, P&L, feature latency) | Standard way to instrument Python services |
| Prometheus | Scrape and store metrics; evaluate alert rules (monitoring/alerts.yml) | The job post names Prometheus explicitly |
| Grafana | Dashboard, auto-provisioned from JSON in monitoring/grafana/ (data source + dashboard provisioning files) | The job post names Grafana explicitly |
| Docker | Container image for the app (multi-stage Dockerfile, slim Python base, non-root user) | Reproducible deployment |
| Docker Compose | docker-compose.yml running app, prometheus, grafana together with one command | One-command demo for reviewers |

Alertmanager is not required; alerts are defined in Prometheus and shown in Grafana.

### 14.8 Code quality and testing

| Technology | Used for | Why |
|---|---|---|
| pytest | Unit and integration tests | Standard Python testing |
| Hypothesis | Property-based tests for order-book invariants (e.g. best bid < best ask after every valid update; sizes never negative; queue-ahead never increases from trades) | Finds edge cases hand-written tests miss |
| pytest-cov | Coverage report (aim for >80% on book/, features/, sim/) | Shows the critical code is tested |
| ruff | Linting and formatting | Fast, replaces flake8 + black + isort |
| mypy | Static type checking on book/, features/, sim/, replay/ | Catches bugs in the high-consequence code |
| pre-commit | Run ruff and mypy before each commit | Keeps the repo clean |
| GitHub Actions | CI: install with uv, run ruff, mypy, pytest (synthetic data only) on every push | Shows professional workflow; never needs an API key |

### 14.9 Explicitly out of scope

Do not add: Kafka/Redis/message queues, a database server (Postgres etc.), Kubernetes, cloud deployment, web frameworks, deep learning, live trading or live broker connections, or MLflow. Keep the system simple, fast, and explainable. These can be mentioned in the report as future work.

### 14.10 Hardware assumptions

A laptop with 8-16 GB RAM and ~20 GB free disk. If memory becomes a problem, process one day at a time and use Polars lazy scans and DuckDB over Parquet instead of loading everything.
