"""Canonical event and book-frame layouts shared by every module.

Two tables flow through the pipeline:

* events: level-by-level book deltas (what an MBO or L2 feed carries).
  Columns: ts_event, sequence, instrument_id, action, side, price, size, queue_pos.
* book frame: one row per event with the book state *after* that event
  (what Databento mbp-1 / mbp-10 records carry). Features, targets and the
  fill simulator all read the book frame.

Prices are stored as integer ticks so that comparisons are exact.
"""

from __future__ import annotations

from typing import Final

# Actions. Values are small ints so numba kernels can switch on them.
ADD: Final = 0  # add `size` lots at `price` on `side`
CANCEL: Final = 1  # remove `size` lots at `price` on `side`
MODIFY: Final = 2  # set the level at `price` on `side` to an absolute `size`
TRADE: Final = 3  # aggressor on `side` takes `size` lots resting at `price` on the other side
CLEAR: Final = 4  # wipe the book (session reset / recovery)

ACTION_NAMES: Final = {
    ADD: "add",
    CANCEL: "cancel",
    MODIFY: "modify",
    TRADE: "trade",
    CLEAR: "clear",
}

# Sides. For book actions the side is the resting side; for trades it is the
# aggressor (BID = a buyer lifted the ask, ASK = a seller hit the bid).
BID: Final = 1
ASK: Final = -1
NONE: Final = 0

# Sentinels for an empty side of the book.
NO_BID: Final = -(2**40)
NO_ASK: Final = 2**40

EVENT_COLUMNS: Final = (
    "ts_event",
    "sequence",
    "instrument_id",
    "action",
    "side",
    "price",
    "size",
    "queue_pos",
)

TOP_COLUMNS: Final = ("bid_px", "ask_px", "bid_sz", "ask_sz")


def depth_columns(levels: int) -> tuple[list[str], list[str]]:
    """Names of the per-level size columns on a dense price grid.

    bid_sz_i is the size at price bid_px - i, ask_sz_i at ask_px + i
    (i = 0 is the touch, identical to bid_sz / ask_sz).
    """
    return [f"bid_sz_{i}" for i in range(levels)], [f"ask_sz_{i}" for i in range(levels)]
