"""Pytest configuration for fuzzylogs tests.

Ensures local modules are importable when tests are run from the repository
root (for example: ``pytest fuzzylogs``).
"""

from pathlib import Path
import sys


FUZZYLOGS_DIR = Path(__file__).resolve().parent.parent

if str(FUZZYLOGS_DIR) not in sys.path:
    sys.path.insert(0, str(FUZZYLOGS_DIR))
