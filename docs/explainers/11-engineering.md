# Engineering (`config.py`, `cli.py`, tests, CI)

## Configuration

Every tunable number is in `config.yaml`, loaded into typed Pydantic models. Unknown keys, wrong types and invalid values (a negative fee, an unknown queue model) fail at startup with a clear message. Any key can be overridden from the command line: `--set sim.decisions_per_day=500`.

## Command line

```
halftick synth           # generate synthetic days
halftick download        # Databento: quote, budget-check, buy only with --yes
halftick build-features  # quality checks, book, features, targets
halftick train           # walk-forward direction models
halftick simulate        # execution simulation over the full grid
halftick diagnose        # regime breakdowns, signal decay, stability, queue validation
halftick report          # REPORT.md and the README results block
halftick replay          # event-driven replay with Prometheus metrics
halftick all             # everything, for both instruments
```

Results only ever come from these commands, never from notebooks, and `report` builds the write-up from the tables, so no number in it is typed by hand.

## Logging

structlog writes JSON lines with a level and a timestamp. Library code never prints.

## Tests

68 tests, run on every push. The highlights:

* **Order book:** each action type by hand, plus a property-based test (Hypothesis) that checks the book against a dictionary reference model over hundreds of random order flows.
* **Features:** hand-computed imbalance, microprice and OFI; no-look-ahead tests at several cut points; streaming equals batch.
* **Simulator:** known fill times for each queue assumption; deadline crosses; latency; a property test that queue-ahead never increases and never exceeds the displayed level.
* **Costs:** crossing and posting costs, abandonment, the oracle, and fees (one fee per order, savings independent of the fee).
* **Walk-forward:** a property test that no fold ever leaks a future day.
* **Databento:** budget guard, cache hits, dry run, missing key, and record conversion, all with a fake client.
* **Integration:** the whole pipeline on tiny data, with invariants such as "the oracle is never beaten" and "every strategy faced the same orders".

Coverage is measured with numba's JIT turned off (otherwise compiled functions are invisible to the coverage tool): about 97% overall.

## CI

GitHub Actions installs with uv from the lockfile, then runs ruff (lint and format), mypy on the core modules, the tests with numba compiled (catches numba typing errors), and the tests again with JIT off for coverage, failing below 90%. CI never needs an API key or network access to market data.
