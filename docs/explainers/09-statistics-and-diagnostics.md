# Statistics and diagnostics (`sim/summary.py`, `diagnostics/breakdowns.py`)

## Uncertainty on the savings

Saying "the model saves 0.04 ticks" is meaningless without a confidence interval. Two complementary methods:

* **Day-block bootstrap.** Orders on the same day share market conditions, so they are not independent. We resample whole test days with replacement (2,000 times), recompute the mean saving each time, and take the 2.5% and 97.5% points. With few test days the interval is honestly wide.
* **Newey-West (HAC) t-statistic** on the time-ordered per-order savings, which corrects the standard error for autocorrelation.

When the two disagree, trust the bootstrap more: it captures day-to-day variation, which is the bigger risk with few days.

## Where it works and where it fails

The headline cell is broken down by:

* time of day (first 90 minutes, midday, last 90 minutes);
* volatility tercile (mid changes per second over the last 5 minutes);
* queue size on our side (below or above the median);
* spread state (one tick or wider);
* macro-release windows versus normal periods (release windows are flagged, never deleted, and analysed separately);
* ZN versus ES.

The report's "works best when / fails when" sentence is picked automatically from the bucket with the largest and smallest saving against always posting, among buckets with at least 100 orders.

## Signal decay

The AUC of queue imbalance alone and of LightGBM is measured against the direction of the mid move over the next event, 1 s, 5 s and 30 s. Order-book signals are short-lived; the curve shows how fast the information fades.

## Stability

Per-day AUC and per-day mean cost by strategy. One great day can make an average look good; this shows whether the edge is steady.

## Queue-model validation

On the same orders, each queue assumption is compared with the exact truth: fill rate, how often the fill outcome agrees, the bias in mean cost, and the error in fill time. With real data this is what a few days of MBO would be bought for.
