"""
Regression test suite for lot sizes across the codebase.
Asserts that no production source file hardcodes stale lot sizes (BANKNIFTY=15 or NIFTY=25/75)
and verifies that config/lot_sizes.yaml correctly resolves current and historical values.
"""

import re
from pathlib import Path
from src.config.lot_sizes import get_lot_size, load_lot_sizes


def test_lot_sizes_yaml_values():
    """Verify lot_sizes.yaml loads correctly with post-2026 and pre-2026 values."""
    config = load_lot_sizes()
    assert config["BANKNIFTY"] == 30
    assert config["NIFTY"] == 65
    assert config["pre_2026"]["BANKNIFTY"] == 35
    assert config["pre_2026"]["NIFTY"] == 75

    # Check getter helper
    assert get_lot_size("BANKNIFTY") == 30
    assert get_lot_size("NIFTY") == 65
    assert get_lot_size("BANKNIFTY 50") == 30
    assert get_lot_size("NIFTY 50") == 65

    # Check pre-2026
    assert get_lot_size("BANKNIFTY", pre_2026=True) == 35
    assert get_lot_size("NIFTY", pre_2026=True) == 75


def test_no_hardcoded_stale_lot_sizes_in_src():
    """Asserts no Python file in src/ hardcodes BANKNIFTY=15 or NIFTY=25/75."""
    bad_patterns = [
        re.compile(r"lot_size\s*=\s*15\b"),
        re.compile(r"lot_size\s*=\s*25\b"),
        re.compile(r"lot_size:\s*int\s*=\s*15\b"),
        re.compile(r"lot_size:\s*int\s*=\s*25\b"),
        re.compile(r"['\"]lot_size['\"]\s*:\s*15\b"),
        re.compile(r"['\"]lot_size['\"]\s*:\s*25\b"),
        re.compile(r"else\s+25\b"),
        re.compile(r"else\s+75\b"),
        re.compile(r"#\s*1 lot = 15\b"),
        re.compile(r"qty\s*=\s*15\b"),
    ]

    src_dir = Path(__file__).resolve().parents[1] / "src"
    hits = []

    for py_file in src_dir.rglob("*.py"):
        # Ignore tests or temp scratch scripts if any
        if "test_" in py_file.name:
            continue
        try:
            content = py_file.read_text(encoding="utf-8")
        except Exception:
            continue

        for pat in bad_patterns:
            matches = pat.findall(content)
            if matches:
                hits.append(f"{py_file.name} matched {pat.pattern}")

    assert hits == [], f"Hardcoded stale lot sizes found in src/: {hits}"
