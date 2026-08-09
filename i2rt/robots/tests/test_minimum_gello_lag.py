"""Regression tests for the minimum_gello follower teleop-lag fix.

Background: the follower's hardware worker used to drain its inter-process command queue
one item per ~2 ms loop in strict FIFO order, so any backlog was replayed instead of
skipped — the follower fell seconds behind the leader and the lag grew over time. The fix
makes the follower latest-value-wins on both hops:

  * ``_yam_polling_worker`` drains the queue to the newest command each loop (drops stale).
  * the follower command queue is bounded and ``_put_latest_command`` drops stale on overflow.

These tests pin both behaviours. The example lives outside the package, so it is loaded by
path. The worker is driven directly (no portal RPC server) so the tests bind no port and are
safe under ``pytest -n auto``.

Run with:
    uv run pytest i2rt/robots/tests/test_minimum_gello_lag.py -v
"""

import importlib.util
import pathlib
import queue as _queue
import threading
import time

import numpy as np
import portal
import pytest

# Load examples/minimum_gello/minimum_gello.py by path (it is not an importable package).
_MG_PATH = pathlib.Path(__file__).resolve().parents[3] / "examples" / "minimum_gello" / "minimum_gello.py"
_spec = importlib.util.spec_from_file_location("minimum_gello", _MG_PATH)
assert _spec is not None and _spec.loader is not None
mg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mg)


def test_put_latest_command_is_bounded_and_latest_wins() -> None:
    """``_put_latest_command`` must keep only the newest item, never grow past the bound,
    and never block — even when hammered far faster than anything drains it."""
    q: _queue.Queue = _queue.Queue(maxsize=1)
    n = 1000
    for i in range(n):
        mg._put_latest_command(q, np.array([float(i)], dtype=float))
        # Invariant: the queue never accumulates a backlog.
        assert q.qsize() <= 1

    # Only the most recently enqueued command survives.
    assert q.qsize() == 1
    survivor = q.get_nowait()
    assert survivor[0] == float(n - 1)
    assert q.empty()


@pytest.mark.sim
def test_follower_worker_applies_newest_setpoint_not_backlog() -> None:
    """The follower worker must jump to the NEWEST queued setpoint, not replay a FIFO backlog.

    We pre-load a large backlog of stale commands followed by one distinct final target into an
    (unbounded) queue and run the real ``_yam_polling_worker`` in sim. With drain-to-newest the
    worker reaches the final target within a couple of loop periods; the old one-pop-per-loop
    FIFO would spend ``backlog * _WORKER_LOOP_PERIOD_S`` (seconds) replaying stale commands first.
    """
    args = mg.Args(arm="yam", gripper="crank_4310", sim=True)
    pos_shared = portal.SharedArray(shape=(mg._MAX_DOFS,), dtype=np.float64)
    n_dofs_value = portal.mp.Value("i", 0)
    # Unbounded on purpose: simulate a backlog that already formed. The worker runs in a thread
    # here, and its drain logic only needs get_nowait()/queue.Empty, so a plain queue.Queue is
    # both faithful and free of multiprocessing.Queue's feeder-thread teardown hazard.
    cmd_queue: _queue.Queue = _queue.Queue()
    stop_event = portal.mp.Event()

    worker = threading.Thread(
        target=mg._yam_polling_worker,
        args=(args, pos_shared, n_dofs_value, cmd_queue, stop_event, "test-follower"),
        daemon=True,
    )
    worker.start()
    try:
        # Wait for the sim robot to come up and publish its DOF count.
        deadline = time.time() + 30.0
        while n_dofs_value.value == 0 and time.time() < deadline:
            time.sleep(0.02)
        n = n_dofs_value.value
        assert n == 7, f"sim follower did not start (num_dofs={n})"

        home = pos_shared.array[:n].copy()
        stale = home.copy()
        stale[0] = home[0] - 0.2
        final = home.copy()
        final[0] = home[0] + 0.3

        backlog = 1000  # 1000 * 2 ms = ~2 s of replay on the buggy FIFO path
        for _ in range(backlog):
            cmd_queue.put(stale.copy())
        cmd_queue.put(final.copy())

        # Drain-to-newest reaches `final` almost immediately; the old FIFO would take ~2 s.
        converge_deadline = time.time() + 0.8
        reached = False
        while time.time() < converge_deadline:
            if abs(pos_shared.array[0] - final[0]) < 1e-3:
                reached = True
                break
            time.sleep(0.005)

        assert reached, (
            f"follower did not converge to the newest setpoint within 0.8 s "
            f"(joint0={pos_shared.array[0]:.4f}, expected {final[0]:.4f}) — "
            f"indicates FIFO backlog replay instead of drain-to-newest"
        )
    finally:
        stop_event.set()
        worker.join(timeout=5.0)
        pos_shared.close()
