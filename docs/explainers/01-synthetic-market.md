# Synthetic market (`data/synthetic.py`)

## What it does

Generates realistic-looking order-book days for ZN and ES: every add, cancel and trade at every price level, with exchange timestamps, sequence numbers and the true queue position of every cancellation.

## Why it exists

Real CME data needs a Databento account, which needs a card. Instead of stopping, the project runs on a generator whose structure is close enough to a real large-tick book that every downstream piece (features, models, simulator, statistics, monitoring) can be built and tested for real. Swapping in Databento later changes only where the events come from.

## How it works

It is a **queue-reactive model** (Huang, Lehalle & Rosenbaum, 2015). At every moment, each possible event has a rate that depends on the current book:

* limit orders arrive at the best bid and ask, faster when the queue is small (people refill thin queues);
* cancellations arrive at a rate that grows with queue size;
* market orders arrive and eat the front of the opposite queue;
* deeper levels get their own adds and cancels.

The next event is drawn with the Gillespie method: draw the waiting time from an exponential distribution with the total rate, then pick which event happened in proportion to its rate.

The key point: **the price only moves when a best queue is fully used up.** The smaller queue is more likely to empty first, so queue imbalance predicts the next move *because of the market's mechanics*, not because I wrote "imbalance predicts price" into the code. That is the same reason it works in real large-tick markets.

On top of that:

| Ingredient | Why |
|---|---|
| Hidden "informed flow" signal x(t), an Ornstein-Uhlenbeck process that tilts buy vs sell market orders and limit-order placement | Gives order-flow features (OFI, trade-flow imbalance) information beyond static queue sizes, so LightGBM has something to find beyond the baseline |
| Intraday U-shape in activity | Real markets are busier at the open and close; gives the time-of-day diagnostics something to show |
| Scheduled "macro release" shocks: activity jumps 4x and x(t) jumps | So release windows behave differently and can be analysed separately |
| A contract roll: daily volume migrates from one contract to the next over three days | So roll-week detection has to work from volume, as with real data |
| True queue position of each cancellation, skewed toward the back of the queue | Plays the role of MBO data; lets the simulator's queue assumptions be scored against the truth |
| Injected faults: duplicate records, sequence gaps, a few-second feed outage | So the data-quality checks have something real to catch |

## Calibration

Parameters were tuned by simulating one hour at a time and checking: median best queue (ZN about 770 lots, ES about 40), spread at one tick about 99% of the time, a full ZN tick move roughly every 20 seconds, ES much faster, and P(next move up) rising steadily across imbalance deciles. These are rough, plausible targets, **not** fitted to real ZN data.

## Assumptions and what could go wrong

* The generator encodes my beliefs about real books. A model that wins here has learned the generator. This is the main limitation of every result in the project, and the report says so up front.
* Market orders do not react to our own orders, and nobody reacts to anyone strategically beyond the rate rules.
* Cancellations are placed in the queue with a fixed skew toward the back. Real cancellation behaviour varies by participant type.

## Alternatives considered

* **Replay a public sample dataset**: no free full-depth CME data with queue positions exists.
* **Zero-intelligence model** (constant rates, Smith et al. 2003): simpler, but queue sizes would not feed back into rates, so imbalance would carry less realistic information.
* **Hawkes processes** (self-exciting arrivals): more realistic clustering, but harder to explain and to tune without real data. A good next step once real data is available.
