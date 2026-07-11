"""Testing-policy enforcement for the whole suite.

Every test must:

1. Be fuzzed with Hypothesis (``@given``) OR explicitly opt out with
   ``@pytest.mark.no_fuzz``.
2. Use the Method of Manufactured Solutions (``@pytest.mark.mms``) OR
   explicitly opt out with ``@pytest.mark.no_mms``.

A test that satisfies neither is reported as an error so the omission is
deliberate and visible rather than silent.
"""

import pytest


def pytest_configure(config: pytest.Config):
    config.addinivalue_line(
        "markers", "no_fuzz: test is intentionally not fuzzed with Hypothesis"
    )
    config.addinivalue_line(
        "markers", "mms: test uses the Method of Manufactured Solutions"
    )
    config.addinivalue_line("markers", "no_mms: test intentionally does not use MMS")


def _is_hypothesis_test(item: pytest.Item) -> bool:
    if item.get_closest_marker("hypothesis") is not None:
        return True
    return getattr(getattr(item, "obj", None), "is_hypothesis_test", False)


def pytest_runtest_setup(item: pytest.Item):
    fuzzed = _is_hypothesis_test(item) or item.get_closest_marker("no_fuzz")
    if not fuzzed:
        raise Exception(
            f"{item.nodeid}: test is not fuzzed. Add a Hypothesis @given "
            "strategy or mark it @pytest.mark.no_fuzz to opt out explicitly."
        )
    mms = item.get_closest_marker("mms") or item.get_closest_marker("no_mms")
    if not mms:
        raise Exception(
            f"{item.nodeid}: test does not declare MMS usage. Mark it "
            "@pytest.mark.mms (uses Method of Manufactured Solutions) or "
            "@pytest.mark.no_mms to opt out explicitly."
        )
