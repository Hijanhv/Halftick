# Fill simulator (`sim/kernel.py`)

This is the core of the project and the part most backtests get wrong.

## The task

At a random moment, we must buy (or sell) one contract within D seconds (5, 15 or 30). Several thousand such decisions are drawn per day, each with a random side.

## The naive backtest, and why it lies

A naive backtest says "I posted at the bid, and the price later traded at the bid, so I got filled." In ZN there may be 800 lots ahead of you at the bid. A trade of 40 lots at your price fills the people in front of you, not you. Naive fills make passive strategies look far better than they are.

## What this simulator does

1. **Entry.** The order is sent at t0 and arrives after the latency (0, 1, 10 or 100 ms). It joins the **back** of the queue at the near touch, so `queue_ahead` = the displayed size at that moment, using the book after the delay.
2. **Trades at our price** shrink the queue ahead. We fill when a trade at our price is **larger than what is still ahead of us**.
3. **Cancellations at our price** may or may not be ahead of us. The data does not say, so it is an assumption:
   * **pessimistic:** cancellations always come from behind us;
   * **proportional:** a cancellation removes volume ahead and behind in proportion;
   * **optimistic:** cancellations come from ahead of us first;
   * **exact:** use the true position (available from the synthetic generator, or from MBO data in real life).
4. **Other fill rules.** We also fill if the market trades *through* our price (our level must have emptied), or if the other side's best price reaches ours (an incoming order would have matched us).
5. **A sanity clamp.** The queue ahead can never exceed the size displayed at our level, and it never increases (later orders join behind us).
6. **Deadline.** If we are not filled by t0 + D, we cross the spread then, paying the ask (or bid) in force after the latency.

## Checkpoints

While the order rests, the kernel records a checkpoint every 250 ms: where we are in the queue, how much time is left, and the price we would pay if we gave up and crossed right then. Strategies that can abandon the order are evaluated on these checkpoints afterwards in numpy, so the expensive walk through the events happens only once per decision.

## Join fresh levels

A separate mode waits for a **new** price level to open on our side (someone improves a 2-tick spread with a small order) and joins it near the front of the queue. If no fresh level appears before the deadline, it crosses.

## Testing

A hand-built scenario has a known fill time under each assumption (optimistic 3 s, proportional and exact 4 s, pessimistic 5 s), and the tests check them. Others cover deadline crosses, trade-throughs, the sell side, latency picking up a moved ask, and the fresh-level mode. A property-based test feeds random order flow at one level and checks that queue-ahead never increases and never exceeds the displayed size, for every queue model.

## Assumptions and what could go wrong

* Our own 1-lot order is invisible to everyone else. Fine for one lot; wrong for size.
* No re-pegging: if the price moves away, the order stays where it is until the deadline. Re-pegging is listed as future work.
* Latency is a fixed delay, not a distribution, and we assume the abandon-and-cross decision does not race with a fill during the latency.
* Real mbp data hides cancellation positions. The report scores each assumption against the exact truth to show how much that matters.
