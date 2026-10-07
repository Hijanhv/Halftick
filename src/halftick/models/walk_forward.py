"""Day-based walk-forward splits.

Order-book data is strongly autocorrelated within a day, so random row splits
leak: a test row's neighbours, one millisecond apart, sit in the training set.
We split by whole days and only ever move forward in time:

    train on days [0, k)  ->  calibrate on day k  ->  test on day(s) after k

sklearn's TimeSeriesSplit works on rows and has no calibration fold, so this
small custom splitter is clearer and is tested directly for leakage.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    train: tuple[dt.date, ...]
    calibrate: tuple[dt.date, ...]
    test: tuple[dt.date, ...]


def walk_forward(
    days: Sequence[dt.date],
    min_train_days: int,
    calibrate_days: int = 1,
    test_days: int = 1,
    expanding: bool = True,
    max_train_days: int | None = None,
) -> Iterator[Fold]:
    """Yield folds over sorted, unique days. The last fold may test fewer days."""
    ordered = sorted(set(days))
    if list(days) != ordered:
        raise ValueError("days must be sorted and unique")
    k = min_train_days
    while k + calibrate_days < len(ordered):
        lo = 0 if expanding else k - min_train_days
        if max_train_days is not None:
            lo = max(lo, k - max_train_days)
        train = tuple(ordered[lo:k])
        cal = tuple(ordered[k : k + calibrate_days])
        test = tuple(ordered[k + calibrate_days : k + calibrate_days + test_days])
        if not test:
            break
        yield Fold(train, cal, test)
        k += test_days


def assert_no_leakage(fold: Fold) -> None:
    if not fold.train or not fold.calibrate or not fold.test:
        raise AssertionError("every fold needs train, calibrate and test days")
    if max(fold.train) >= min(fold.calibrate) or max(fold.calibrate) >= min(fold.test):
        raise AssertionError(f"time order violated in {fold}")
