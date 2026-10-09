# Order book (`book/l2book.py`)

## What it does

Keeps the size resting at every price on both sides, and after each event reports the best bid, best ask and the sizes at the first ten price levels.

## Why it is built this way

* **Dense arrays, not a dictionary.** Large-tick futures trade inside a narrow band each day, so the book is two arrays of a few thousand ticks indexed by `price - base`. Every update is O(1). The only search happens when the best level empties, and then the next level is usually one tick away.
* **Numba.** A ZN day is about a million events. A Python loop would take minutes; the numba version rebuilds a day in under a second.
* **One implementation, two users.** The same `apply_event` function is called by the batch pipeline and by the event-driven `OrderBook` object used in tests and the replay engine, so they cannot disagree.
* **Prices as integer ticks.** Comparisons are exact; there is no floating-point equality to worry about.

## Actions

| Action | Effect |
|---|---|
| ADD | add size at a price |
| CANCEL | remove size at a price |
| MODIFY | set a level to an absolute size |
| TRADE | the aggressor on one side removes size from the other side's level |
| CLEAR | empty the book |

Each update returns bit flags for problems: crossed book, locked book, size going negative, price outside the array, bad size. The data-quality report counts them.

## Depth on a price grid

Depth column *i* is the size *i ticks* from the touch, including empty ticks as zero. Databento lists non-empty levels instead; the converter maps them onto the same grid so every downstream module sees one format.

## Tests

Each action type is tested by hand, and a property-based test (Hypothesis) runs hundreds of random valid order flows against a simple dictionary model and checks after every event that best prices and sizes match and the book never crosses.

## What could go wrong

* If prices wander outside the array window, updates are flagged and the day fails loudly. The window (16,384 ticks) is far wider than any ZN or ES day.
* The L2 book cannot tell individual orders apart. That is exactly why the fill simulator needs a queue-position assumption.
