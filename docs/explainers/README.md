# Explainers

One short, plain-English note per module, written to be explained out loud in an interview. Each covers what the module does, why it is built this way, the assumptions it makes, what could go wrong, and the alternatives that were considered.

Read them in pipeline order:

1. [Synthetic market](01-synthetic-market.md): where the data comes from
2. [Data layer](02-data.md): Databento downloads, sessions, roll weeks, quality checks
3. [Order book](03-order-book.md): rebuilding the book from events
4. [Features](04-features.md): imbalance, microprice, OFI and the rest
5. [Targets](05-targets.md): what we predict
6. [Direction models](06-direction-models.md): baselines, LightGBM, calibration, walk-forward
7. [Fill simulator](07-fill-simulator.md): the queue-aware core
8. [Strategies and the cost-to-go model](08-strategies.md)
9. [Statistics and diagnostics](09-statistics-and-diagnostics.md)
10. [Replay engine and monitoring](10-replay-and-monitoring.md)
11. [Engineering](11-engineering.md): config, CLI, tests, CI, Docker

A one-paragraph version of the whole project is at the end of [08-strategies.md](08-strategies.md#the-whole-project-in-one-paragraph).
