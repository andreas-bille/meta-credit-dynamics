from __future__ import annotations

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--regenerate",
        action="store_true",
        default=False,
        help="Regenerate reference values for tests that support it.",
    )
