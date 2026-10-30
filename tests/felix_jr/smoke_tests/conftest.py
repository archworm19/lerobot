"""Everything in this directory is a smoke test: slow, end to end, no hardware (run with `-m smoke`)."""

import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if "smoke_tests" in item.path.parts:
            item.add_marker(pytest.mark.smoke)
