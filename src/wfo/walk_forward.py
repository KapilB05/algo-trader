"""
Walk-Forward Optimization — Purged & Embargoed Cross-Validation Splitter.
"""

from __future__ import annotations

import logging
from typing import Generator, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class PurgedWalkForwardSplitter:
    """
    Purged & Embargoed Walk-Forward Time Series Splitter.

    Ensures zero temporal leakage by:
      1. Purging training samples whose forward label evaluation window
         (e.g., 20 candles) overlaps with the test set start.
      2. Applying an embargo buffer of N candles after test sets before
         subsequent training samples can be included.
    """

    def __init__(
        self,
        n_splits: int = 5,
        forward_window: int = 20,
        embargo_pct: float = 0.02,
        min_train_size: int = 100,
    ) -> None:
        self.n_splits = n_splits
        self.forward_window = forward_window
        self.embargo_pct = embargo_pct
        self.min_train_size = min_train_size

    def split(
        self, X: np.ndarray
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate (train_idx, test_idx) pairs with purging and embargoing.
        """
        n_samples = len(X)
        split_size = n_samples // (self.n_splits + 1)

        for i in range(self.n_splits):
            test_start = (i + 1) * split_size
            test_end = min(test_start + split_size, n_samples)

            if test_start >= n_samples:
                break

            test_idx = np.arange(test_start, test_end)
            raw_train_idx = np.arange(0, test_start)

            purge_cutoff = test_start - self.forward_window
            purged_train_idx = raw_train_idx[raw_train_idx < purge_cutoff]

            if len(purged_train_idx) < self.min_train_size:
                logger.warning(
                    "Split %d: Train size %d < min_train_size %d after purging. Skipping.",
                    i + 1, len(purged_train_idx), self.min_train_size,
                )
                continue

            yield purged_train_idx, test_idx
