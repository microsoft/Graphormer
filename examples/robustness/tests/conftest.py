"""Tests for graphormer-redteam.

Run with::

    pytest -q

The tests are organized to mirror the package layout: one test file
per module, with the highest-value invariants (Betti computation,
trigger attachment, Graphormer adapter shape, end-to-end robustness
run) covered explicitly.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
