# Strategies and the cost-to-go model (`sim/policy.py`, `sim/runner.py`)

## Cost

Cost is measured in ticks against the **arrival mid** (the mid at the decision moment), signed so that positive means we paid. A market buy with a one-tick spread costs +0.5. A passive buy filled at the bid costs -0.5 (we earned half the spread). The fee ($2.50 per side by default) is added in dollars.

**Fees cancel out.** Every strategy trades exactly one contract per order and pays exactly one fee, and CME pays no rebate for posting. Fees change every strategy's cost by the same amount and never change which one wins. The report shows this with a fee table instead of a fee dimension in the grid.

## The strategies

All of them face **exactly the same decisions** (same times, same sides), so the comparison is paired and much less noisy.

| Strategy | Rule |
|---|---|
| Always cross | Market order now |
| Always post | Join the touch, cross at the deadline if unfilled |
| Direction rule | Post. At each checkpoint, cross if the calibrated P(next move is against us) > θ. θ is chosen on the calibration day from a grid. |
| Cost-to-go model | Post. At each checkpoint, cross if a model predicts waiting will cost more than crossing now |
| Join fresh levels | Wait for a new small level on our side and join it near the front; cross at the deadline otherwise |
| Oracle | **Upper bound, uses future information.** The cheapest of crossing at any checkpoint or waiting to the end |

## The cost-to-go model

At every checkpoint of every simulated order on the training days, we know two things after the fact:

* what we would pay if we **keep waiting** (the eventual fill price, or the deadline cross);
* what we would pay if we **cross right now**.

The label is the difference, in ticks. A LightGBM regression learns it from the market features (turned around to the order's point of view: "imbalance against me" instead of "imbalance up") plus the order's own state: queue ahead, queue ahead as a share of the level, time left, whether we are still at the touch, and latency. The policy crosses the first time the prediction is above zero.

**Why this rule is sound.** It compares "cross now" with "never give up" at each step. That is a one-step policy improvement over always posting: if the model were exact, it could only lower the expected cost compared with always posting. It does not search over all future decisions, which a full dynamic program or reinforcement learning would.

**Why it was added.** The spec's direction rule only uses P(adverse move). But what matters is cost, and cost also depends on how deep we are in the queue and how much time is left. Training on the decision we actually make targets the right thing.

## Walk-forward for the simulator

Training on at least 8 days, tuning on the next day, testing on the following 3, then rolling forward. One cost model per queue assumption is fitted per fold on rows pooled over the latency and deadline settings (time left and latency are features), which keeps memory small.

## What could go wrong

* The cost model learns this generator's dynamics. It is the most likely component to overfit synthetic quirks.
* If the cost model is biased high, it crosses too often and gives back the spread. If biased low, it waits too long.

## The whole project in one paragraph

Halftick asks when a trader should pay half a tick to cross the spread instead of waiting in a deep queue. It rebuilds the order book event by event, computes the classic microstructure signals (queue imbalance, microprice, order flow imbalance), and predicts the next price move with calibrated models tested strictly forward in time. Then it simulates limit orders properly: joining the back of the queue, moving up only as trades and cancellations clear the volume ahead. It compares six strategies on identical orders under four queue assumptions, four latencies and three deadlines, and reports the savings with day-level bootstrap confidence intervals. Everything runs on a synthetic market for now, so the numbers demonstrate the method, not a real-market edge.
