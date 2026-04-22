"""conftest.py for the tinygrad engine tests.

Ensures tinygrad's Device.DEFAULT is restored to the real hardware device
before each test, so that tests in other modules that set `Device.DEFAULT = "CPU"`
do not pollute these tests (which require GPU execution).
"""

import pytest
from tinygrad.device import Device

# Capture the hardware device at collection time, before any test module runs.
_REAL_DEVICE: str = Device.DEFAULT


@pytest.fixture(autouse=True)
def restore_tinygrad_device() -> None:
    """Reset Device.DEFAULT to the real hardware device before each test."""
    Device.DEFAULT = _REAL_DEVICE
