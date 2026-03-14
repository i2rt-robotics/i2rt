def pytest_configure(config: object) -> None:
    config.addinivalue_line("markers", "sim: tests that run in simulation (no hardware required)")
    config.addinivalue_line("markers", "real: tests that require real hardware (CAN bus + motors)")
