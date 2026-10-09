# Features (`features/compute.py`, `features/streaming.py`)

## What they are

Every feature is computed at every event using only that event and earlier ones.

| Feature | Plain meaning | Source |
|---|---|---|
| Queue imbalance | (bid size - ask size) / (bid size + ask size). Near +1: lots of buyers waiting, few sellers, so the ask is more likely to run out first and the price to tick up. | Gould & Bonart (2016) |
| Depth imbalance (3/5/10 levels) | Same idea over several levels | |
| Microprice | A size-weighted mid: it leans toward the side with the smaller queue. Shown as distance from the mid in ticks. | Stoikov (2018), first-order version |
| OFI (order flow imbalance) | Net pressure from changes at the touch: bid queue growing or bid price rising counts as buying pressure, the ask side the opposite. Summed over the last 10/50/200 events and 100 ms/1 s/5 s. | Cont, Kukanov & Stoikov (2014) |
| Trade-flow imbalance | (buy volume - sell volume) / total over the window, by aggressor side | |
| Queue depletion rate | How fast the best bid or ask queue is shrinking, in lots per second | |
| Spread, wide-spread flag | Usually one tick | |
| Time since last mid change | Stale books behave differently | |
| Volatility | Mid changes per second over 30 s and 5 min | |
| Minutes since 08:30 | For diagnostics only, not a model input | |

## How they are computed fast

One numba pass per day. Rolling sums use prefix sums; time windows use a two-pointer sweep (for each event, the start of its window only ever moves forward), so each window costs O(n). A full ZN day takes under a second, against the spec's one-minute target.

## No look-ahead

Two tests guard this: features computed on a truncated day must equal the first rows of the full day exactly, at several cut points; and targets computed after deleting earlier rows must be unchanged.

## Streaming version

`StreamingFeatures` computes imbalance, microprice, OFI over 50 events and trade-flow imbalance over 1 s incrementally, with O(1) work per event, the way a live system would. A test replays a synthetic day through it and requires the values to equal the batch kernel's exactly. That keeps the live path and the research path from drifting apart.

## Assumptions and what could go wrong

* OFI here uses only the touch. Multi-level OFI could add information.
* The depletion rate resets when the best price changes, because a new level is a different queue.
* With real mbp-1 data, depth beyond the touch is not available, so depth imbalance would need mbp-10 days.

## Alternatives considered

* **Clock-time sampling** (features every 100 ms): simpler, but loses the event-time structure that large-tick books have. Event time matches the spec and the literature.
* **Polars rolling expressions** instead of numba: fine for fixed event windows, awkward for "since the current best price formed" logic.
