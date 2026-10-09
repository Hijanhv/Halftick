# Replay engine and monitoring (`replay/`, `monitoring/`, `docker-compose.yml`)

## The replay engine

`ReplayEngine` walks a `MarketDataSource` event by event and calls subscribers through two hooks, `on_trade` and `on_book_update`, the same shape a live feed handler has. Trades are dispatched before the book update they caused.

Sources implement one method, `batches()`, which yields book-frame batches:

* `SyntheticSource` generates a day in memory (tests and the demo);
* `ParquetSource` replays a processed day from disk;
* `DatabentoHistoricalSource` replays a cached DBN file.

A future `DatabentoLiveSource` would subscribe to Databento's live gateway, convert each record with the same converter, and yield. Nothing else changes.

In `--realtime` mode the engine sleeps so one second of market time takes 1/speed seconds of wall time, and reports how far it is behind schedule.

**Why the research pipeline does not use this loop:** Python callbacks per event are about a thousand times slower than the numba kernels. The research runs on whole-day arrays. To keep the two paths from drifting apart, a test checks that the streaming features computed in the engine equal the batch features exactly.

## Metrics (Prometheus)

Served on `/metrics` (port 8000): events processed and events per second; replay lag; wall time of the last event; whether the replay clock is inside the session; data-quality errors by type; mid, spread, imbalance, microprice and OFI; a feature-latency histogram; and, for the simulated strategies, orders started, passive fills, cumulative cost in ticks and dollars, and average cost.

The simulated-order metrics come from **playing back the research trade log** in step with the replay clock: an order appears when the replay passes its decision time and completes at its recorded execution time. The dashboard therefore shows the exact orders the report is computed from, not a second, possibly different simulation.

## Alerts (`monitoring/alerts.yml`)

| Alert | Fires when |
|---|---|
| HalftickNoMarketData | No event for over 5 s while inside the session. The synthetic days contain injected 3-8 s feed outages, so this fires for real during a replay. |
| HalftickCrossedBook | A crossed book was seen in the last minute |
| HalftickFeatureLatencyHigh | p99 feature computation time above 5 ms for 15 s |
| HalftickReplayDown / HalftickReplayLagging | The metrics endpoint is down, or the replay is more than 1 s behind schedule for 30 s |

## Docker Compose

`docker compose up --build` starts three services: the app (replaying a day at market speed), Prometheus (scraping every second and evaluating the alerts) and Grafana (with the data source and the "Halftick replay" dashboard provisioned from files, and anonymous read-only access). If `./data` has processed days, the replay uses a day with a trade log; otherwise it generates a synthetic day in memory, so a fresh clone works with no data.

The image is a two-stage build: dependencies are installed from the lockfile with uv in a build stage, and the runtime stage copies only the virtual environment and runs as a non-root user.

## Alternatives considered

* **Alertmanager**: not needed for a demo; alerts show in Prometheus and Grafana.
* **Pushgateway**: for batch jobs. A long-running replay is the standard pull-scrape case.
