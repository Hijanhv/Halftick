# Halftick

Should you cross the spread or wait in line? A queue-aware execution simulator that answers that, tick by tick, in large-tick futures markets.

> **Status: work in progress.** The research pipeline runs end to end, but tests, CI, the Docker/Grafana stack, the written report and the module explainers are still to come. No results are published yet.
>
> **Data: synthetic.** The project currently runs on a synthetic order-book generator (a queue-reactive model calibrated to look like ZN and ES), not on real CME data. Numbers it produces show that the method and the system work; they are not evidence about the real Treasury futures market. The Databento path is written but has not been run against the live API.

## The question

You need to buy (or sell) one ZN contract in the next few seconds. Cross the spread now and pay half a tick, or post a limit order, join the back of the queue, and hope to fill before the price runs away?

## What is built

| Area | Module | What it does |
|---|---|---|
| Synthetic market | `data/synthetic.py` | Queue-reactive order book (Huang, Lehalle & Rosenbaum 2015) with a hidden informed-flow signal, intraday activity, macro-release shocks, a contract roll, true cancellation queue positions, and injected data faults |
| Data | `data/` | Cost-checked Databento downloader with budget ledger and caching, DST-correct session filter, macro-release windows, data-from-volume roll detection, data-quality reports |
| Order book | `book/l2book.py` | Numba price-level book used by both the batch pipeline and the event-driven engine |
| Features | `features/` | Queue imbalance, depth imbalance, microprice, OFI (Cont, Kukanov & Stoikov 2014), trade-flow imbalance, queue depletion, volatility, all strictly backward-looking; a streaming version for live use that matches the batch values exactly |
| Targets | `targets/` | Direction of the next mid change (event time), mid change over 1 s / 5 s / 30 s |
| Models | `models/` | Gould & Bonart imbalance lookup, logistic regression, LightGBM; isotonic calibration; day-based walk-forward with leakage checks; HAC tests |
| Simulator | `sim/` | Queue-aware fills under pessimistic / proportional / optimistic / exact cancellation models, latency, deadlines; six strategies including a cost-to-go model and an oracle upper bound; trade logs; bootstrap and HAC uncertainty |
| Diagnostics | `diagnostics/` | Breakdowns by time of day, volatility, queue size, spread and release windows; signal decay; day-to-day stability; queue-model validation against the truth |
| Replay and monitoring | `replay/`, `monitoring/` | Event-driven `ReplayEngine` with a `MarketDataSource` interface, Prometheus metrics, real-time replay mode |

## Run it

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/). On macOS, LightGBM also needs OpenMP: `brew install libomp`.

```bash
uv sync
uv run halftick synth -i ZN            # generate synthetic days
uv run halftick build-features -i ZN   # quality checks, book, features, targets
uv run halftick train -i ZN            # walk-forward direction models
uv run halftick simulate -i ZN         # execution simulation across the sensitivity grid
uv run halftick diagnose -i ZN         # where it works and where it fails
uv run halftick replay -i ZN --realtime --metrics-port 8000
```

Every tunable number lives in `config.yaml`; override any key with `--set key=value`.
