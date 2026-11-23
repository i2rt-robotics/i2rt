"""Tests for ``DeviceStream``: one device per spawned process, samples streamed back over a pipe.

These run REAL child processes (with the fakes in ``yumi_fake_devices``), because the properties that matter only
exist across a process boundary: the device is built and sampled in the child, a construction error must come
back as the same exception type, a device that stops producing must make the parent's stream go stale rather than
freeze on its last sample, a crashed child must be reported rather than read as "stale forever", and closing
must actually close the device in the child and reap the process -- including a device still stuck in its
constructor, and a child whose parent went away.
"""

import os
import signal
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from i2rt.yumi.device_process import DeviceStream, RemoteTraceback, close_all, wait_ready_all
from i2rt.yumi.tests import yumi_fake_devices as fakes

# Spawning a child re-imports numpy/cv2 and the fake module; generous for a loaded CI runner.
_READY_TIMEOUT_S = 60.0


def _wait_until(predicate: Callable[[], object], timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _stream(**kwargs: object) -> DeviceStream[fakes.Tick]:
    return DeviceStream(
        "counting",
        fakes.CountingDevice,
        kwargs,
        sample_method="read",
        stale_after_s=0.5,
        info_attrs=("label", "pid", "inheritable_fds"),
    )


@pytest.fixture
def stream() -> Iterator[DeviceStream[fakes.Tick]]:
    device_stream = _stream(period_s=0.005)
    try:
        device_stream.wait_ready(_READY_TIMEOUT_S)
        yield device_stream
    finally:
        device_stream.close()


def test_device_runs_in_its_own_process_and_reports_its_info(stream: DeviceStream[fakes.Tick]) -> None:
    assert stream.info["label"] == "counting"
    assert stream.info["pid"] == stream.pid
    assert stream.pid is not None and stream.pid != os.getpid()
    assert _wait_until(lambda: stream.sample_count > 0)
    assert stream.latest().pid == stream.pid


def test_every_new_sample_is_streamed_once_in_order(stream: DeviceStream[fakes.Tick]) -> None:
    assert _wait_until(lambda: stream.sample_count >= 20)
    first = stream.latest()
    time.sleep(0.2)
    second = stream.latest()

    # The child samples every 1 ms but forwards only NEW samples (by timestamp), so the count tracks the device's
    # own 5 ms period rather than the 1 ms polling.
    assert second.index > first.index
    produced = second.index + 1
    assert stream.sample_count <= produced
    assert stream.sample_count >= 0.5 * produced


def test_construction_error_is_reraised_with_its_type_and_the_child_traceback() -> None:
    device_stream = _stream(fail_init="no encoder answered on can_fake")

    with pytest.raises(ValueError, match="no encoder answered on can_fake") as info:
        device_stream.wait_ready(_READY_TIMEOUT_S)

    assert isinstance(info.value.__cause__, RemoteTraceback)
    assert "yumi_fake_devices.py" in str(info.value.__cause__)
    assert _wait_until(lambda: not device_stream._process.is_alive())
    with pytest.raises(RuntimeError, match="closed"):
        device_stream.latest()


def test_a_device_that_stops_producing_goes_stale_with_its_last_error() -> None:
    with _stream(period_s=0.005, stop_after=10) as device_stream:
        device_stream.wait_ready(_READY_TIMEOUT_S)
        assert _wait_until(lambda: device_stream.last_error is not None)
        time.sleep(device_stream.stale_after_s + 0.2)

        with pytest.raises(TimeoutError, match="stopped producing after sample 10"):
            device_stream.latest()


def test_a_crashed_device_process_is_reported_not_read_as_stale() -> None:
    with _stream(period_s=0.005, crash_after=20) as device_stream:
        device_stream.wait_ready(_READY_TIMEOUT_S)
        assert _wait_until(lambda: not device_stream._process.is_alive())

        with pytest.raises(RuntimeError, match="exited"):
            device_stream.latest()


def test_close_closes_the_device_in_the_child_and_reaps_it(tmp_path: Path) -> None:
    flag = tmp_path / "closed"
    device_stream = _stream(closed_flag=str(flag))
    device_stream.wait_ready(_READY_TIMEOUT_S)
    pid = device_stream.pid

    device_stream.close()
    device_stream.close()

    assert not device_stream._process.is_alive()
    assert device_stream._process.exitcode == 0
    assert flag.read_text() == f"closed by {pid}"
    with pytest.raises(RuntimeError, match="closed"):
        device_stream.latest()


def test_close_during_construction_terminates_promptly() -> None:
    device_stream = _stream(init_sleep_s=30.0)
    time.sleep(0.5)

    started = time.monotonic()
    device_stream.close()

    assert time.monotonic() - started < 10.0
    assert not device_stream._process.is_alive()


def test_the_child_ignores_sigint_so_the_parent_owns_shutdown(stream: DeviceStream[fakes.Tick]) -> None:
    assert _wait_until(lambda: stream.sample_count > 0)
    assert stream.pid is not None
    os.kill(stream.pid, signal.SIGINT)
    before = stream.sample_count
    time.sleep(0.3)

    assert stream._process.is_alive()
    assert stream.sample_count > before
    stream.latest()


def test_the_child_closes_its_device_when_the_parent_goes_away(tmp_path: Path) -> None:
    flag = tmp_path / "closed"
    device_stream = _stream(closed_flag=str(flag))
    device_stream.wait_ready(_READY_TIMEOUT_S)
    try:
        # What the child sees when the parent dies: its control pipe hits EOF.
        device_stream._control.close()

        assert _wait_until(flag.exists)
        assert _wait_until(lambda: not device_stream._process.is_alive())
    finally:
        device_stream.close()


def test_wait_ready_returns_only_once_the_first_sample_has_arrived() -> None:
    with _stream(period_s=0.005) as device_stream:
        device_stream.wait_ready(_READY_TIMEOUT_S)

        # No sleep, no retry: usable immediately.
        assert device_stream.sample_count >= 1
        device_stream.latest()


def test_a_device_that_never_produces_fails_wait_ready_with_its_error() -> None:
    device_stream = _stream(period_s=0.005, stop_after=0)
    # Let the spawn and imports finish under whatever load CI has, so the 3 s below covers only the first sample.
    assert _wait_until(device_stream._came_up.is_set, _READY_TIMEOUT_S)

    with pytest.raises(TimeoutError, match=r"produced no fresh sample.*stopped producing after sample 0"):
        device_stream.wait_ready(3.0)

    with pytest.raises(RuntimeError, match="closed"):
        device_stream.latest()


def test_samples_taken_before_wait_ready_are_not_served_as_fresh() -> None:
    # A stream waited on late (Yumi waits for its slowest device first) must not hand out the samples that piled up
    # meanwhile as if they were current: the receiver drains from construction, and age is the child's stamp.
    with _stream(period_s=0.005) as device_stream:
        time.sleep(1.5)
        device_stream.wait_ready(_READY_TIMEOUT_S)

        assert time.time() - device_stream.latest().timestamp < 0.3


def test_a_child_that_dies_after_ready_fails_wait_ready_at_once() -> None:
    device_stream = _stream(period_s=0.005, crash_after=0)
    started = time.monotonic()

    with pytest.raises(RuntimeError, match="exited before it produced a sample"):
        device_stream.wait_ready(_READY_TIMEOUT_S)

    assert time.monotonic() - started < 10.0  # not the full ready timeout


def test_a_missing_info_attribute_comes_back_as_the_construction_error() -> None:
    device_stream = DeviceStream(
        "counting", fakes.CountingDevice, {}, sample_method="read", stale_after_s=0.5, info_attrs=("no_such_attr",)
    )

    with pytest.raises(AttributeError, match="no_such_attr"):
        device_stream.wait_ready(_READY_TIMEOUT_S)


def test_the_child_leaks_no_inheritable_descriptors(stream: DeviceStream[fakes.Tick]) -> None:
    # Anything the device's native code forks and execs would otherwise hold the pipes open past the child.
    assert stream.info["inheritable_fds"] == []


def test_wait_ready_all_raises_the_first_failure_without_waiting_for_slower_devices() -> None:
    slow = _stream(init_sleep_s=30.0)
    broken = _stream(fail_init="camera refused 800x600")
    started = time.monotonic()

    with pytest.raises(ValueError, match="camera refused 800x600"):
        wait_ready_all([(slow, _READY_TIMEOUT_S, True), (broken, _READY_TIMEOUT_S, True)])

    assert time.monotonic() - started < 15.0
    assert not slow._process.is_alive() and not broken._process.is_alive()


def test_close_all_closes_streams_in_parallel() -> None:
    streams = [_stream(close_sleep_s=1.5) for _ in range(3)]
    wait_ready_all([(s, _READY_TIMEOUT_S, True) for s in streams])
    started = time.monotonic()

    close_all(streams)

    # One after another would take 4.5 s; together, about one device's close.
    assert time.monotonic() - started < 3.5
    assert all(s._process.exitcode == 0 for s in streams)
