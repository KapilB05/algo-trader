"""
Lot Sizes Configuration Loader — Single Source of Truth for Indian Indices.

Reads config/lot_sizes.yaml and provides helper accessors with fallback defaults.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

LOT_SIZES_FILE = Path("config/lot_sizes.yaml")

DEFAULT_LOT_SIZES: Dict[str, Any] = {
    "BANKNIFTY": 30,
    "NIFTY": 65,
    "pre_2026": {
        "BANKNIFTY": 35,
        "NIFTY": 75,
    },
}

_cached_lot_sizes: Optional[Dict[str, Any]] = None


def load_lot_sizes(config_path: Optional[Path | str] = None) -> Dict[str, Any]:
    """Load lot sizes from YAML file or return defaults if unavailable."""
    global _cached_lot_sizes
    if _cached_lot_sizes is not None and config_path is None:
        return _cached_lot_sizes

    target_path = Path(config_path) if config_path else LOT_SIZES_FILE

    # If running from tests or subdirectories, search parent paths as well
    if not target_path.exists():
        for parent in [Path.cwd(), Path(__file__).resolve().parents[2]]:
            candidate = parent / "config" / "lot_sizes.yaml"
            if candidate.exists():
                target_path = candidate
                break

    if target_path.exists():
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data and "lot_sizes" in data:
                    result = data.get("lot_sizes") or DEFAULT_LOT_SIZES
                    _cached_lot_sizes = result
                    return result
        except Exception as e:
            logger.warning("Failed to load %s: %s. Using default lot sizes.", target_path, e)

    result = DEFAULT_LOT_SIZES
    _cached_lot_sizes = result
    return result


def get_lot_size(symbol: str, pre_2026: bool = False) -> int:
    """
    Get lot size for a given index symbol.

    Args:
        symbol: Symbol string (e.g. 'BANKNIFTY', 'NIFTY 50', 'BANKNIFTY26JANFUT')
        pre_2026: Whether to return pre-2026 historical lot size (e.g. for backtesting)

    Returns:
        Lot size integer (default 30 for BankNifty, 65 for Nifty).
    """
    sizes = load_lot_sizes()
    norm = symbol.upper().replace(" ", "").replace("_", "")

    if "BANK" in norm:
        key = "BANKNIFTY"
    elif "NIFTY" in norm:
        key = "NIFTY"
    else:
        key = norm

    if pre_2026:
        pre_dict = sizes.get("pre_2026", {})
        if key in pre_dict:
            return int(pre_dict[key])

    if key in sizes:
        return int(sizes[key])

    # Fallback heuristic
    return 30 if "BANK" in norm else 65
