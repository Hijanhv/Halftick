# Targets (`targets/compute.py`)

## What we predict

* **Next mid move direction (event time):** up (1) or down (0) for the next time the mid price changes, however long that takes. This is the Gould-Bonart target and the one the direction models are trained on. Rows where the day ends first have no target.
* **Mid change over 1 s, 5 s and 30 s (clock time):** the mid in force at time t + h minus the mid now, in ticks. Used for signal decay: how quickly does the information fade?

## Why event time for the main target

In a large-tick market the mid can sit still for many seconds. A clock-time target would be mostly zeros, and the model would mostly learn "nothing happens". The next-change target asks the useful question directly: when the price does move, which way?

## Implementation

A backward scan with numba: walking from the end of the day, keep track of the start of the next "run" of equal mids, and label each row by whether that next run is higher or lower. Clock-time targets use binary search on timestamps.

## What could go wrong

* A half-tick mid change (spread widening from 1 to 2 ticks) counts as a move. In this data that is right: a depleted queue moves the mid by half a tick before the spread closes again.
* Rows near the end of the day lose their target. They are excluded, not filled in.
