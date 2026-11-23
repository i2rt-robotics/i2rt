"""Run one sensor in its own process and stream every new sample to the parent over a pipe.

``Yumi`` puts each of its three sensors in a process of its own, for three reasons:

* **No shared GIL.** The camera's JPEG decode, the 250 Hz Vive poll and the 250 Hz CAN poll would otherwise
  compete for one interpreter, so a slow frame decode would jitter the gripper stream and vice versa.
* **A stuck driver stalls only itself.** A V4L2 ``select()`` timeout, a CAN retry loop or a SteamVR hiccup blocks
  the one process that owns that device; the other streams keep flowing, and the parent sees the stuck one go
  stale instead of freezing with it.
* **A native crash is contained.** libopenvr and the V4L2 stack are C code; a crash there kills its device
  process, and the parent reports "process exited" instead of dying mid-recording.

How it works. The parent spawns the child (``spawn``, never ``fork``: forking a process that already runs
driver threads can deadlock the child), and the child constructs the device from ``factory(**kwargs)``. The
child then calls ``sample_method`` every ``sample_period_s`` and **pushes each new sample** -- new meaning its
``timestamp`` attribute changed -- into a one-way data pipe, stamped with the child's ``time.monotonic()`` at
the moment it was taken. A receiver thread in the parent, running from construction on, drains that pipe and
keeps the latest sample. ``latest()`` returns it, or raises ``TimeoutError`` once the newest sample is older
than ``stale_after_s``.

Staleness is measured from the CHILD's stamp, not from when the parent happened to read the sample.
``CLOCK_MONOTONIC`` is one system-wide clock on Linux, so the two processes' readings are comparable. A sample
that sat in the pipe -- because the parent's main thread was hogging the GIL, or because it was read late --
therefore reads as old, not as fresh.

Messages, child -> parent (data pipe)::

    ("ready", info)                the device is up; info = {attr: getattr(device, attr) for attr in info_attrs}
    ("init_error", exc, tb)        construction (or reading info) raised; the child closes and exits
    ("sample", sample, stamp)      a new sample, and the child's time.monotonic() when it was taken
    ("error", text)                sampling raised (at most once a second), or the stream died; see last_error
    ("closed",)                    the device is closed; the child exits

and parent -> child (control pipe): ``("close",)``. The child ignores SIGINT, so a Ctrl-C in the terminal
reaches the parent's shutdown path instead of killing devices mid-close. It closes its device and exits when
the parent disappears (its control pipe hits EOF), and it marks every inherited descriptor close-on-exec, so a
helper the device's native code launches (SteamVR's server, for one) cannot hold this pipe open after the child
is gone.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import pickle
import signal
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from multiprocessing.connection import Connection
from types import TracebackType
from typing import Any, Generic, TypeVar

logger = logging.getLogger(__name__)

SampleT = TypeVar("SampleT")

_CONTEXT = multiprocessing.get_context("spawn")

# The child reports a failing sample call at most this often; the parent keeps the latest text for its errors.
_ERROR_REPORT_INTERVAL_S = 1.0
# How long close waits for the child to confirm, then for it to exit, before escalating to SIGTERM/SIGKILL.
_CLOSE_CONFIRM_TIMEOUT_S = 5.0
_JOIN_TIMEOUT_S = 2.0
# How often the receiver thread wakes to notice a close when no message arrives, and how often waits re-check.
_RECEIVER_POLL_S = 0.1
_WAIT_POLL_S = 0.01


class RemoteTraceback(Exception):
    """The child's formatted traceback, attached as ``__cause__`` of an exception re-raised in the parent."""


def _picklable(exc: BaseException) -> BaseException:
    """``exc`` itself if it survives pickling (so the parent re-raises the same type), else a RuntimeError."""
    try:
        pickle.loads(pickle.dumps(exc))
    except Exception:
        return RuntimeError(f"{type(exc).__name__}: {exc}")
    return exc


def _send_quietly(conn: Connection, message: tuple[Any, ...]) -> None:
    # Best effort: the pipe may be gone, or the payload unpicklable (TypeError as often as PicklingError).
    try:
        conn.send(message)
    except Exception:
        pass


def _make_fds_non_inheritable() -> None:
    # spawn hands the child its pipes (and multiprocessing's own bookkeeping pipes) as inheritable descriptors, so
    # anything native code in the child forks and execs would keep them open: the parent would then never see EOF
    # when the child dies, and closing would wait on a pipe no one will close. Linux-only, like the devices.
    try:
        fds = [int(name) for name in os.listdir("/proc/self/fd")]
    except OSError:
        return
    for fd in fds:
        if fd > 2:
            try:
                os.set_inheritable(fd, False)
            except OSError:
                pass  # e.g. the descriptor listdir itself used, already closed


def _configure_child_logging(name: str, level: int) -> None:
    # Importing a device module can run logging.basicConfig first (dm_driver does, at ERROR), which would make a
    # plain basicConfig here a no-op -- so replace the handlers and set the level explicitly.
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    logging.basicConfig(level=level, format=f"%(asctime)s [{name} process] %(levelname)s %(name)s: %(message)s")
    root.setLevel(level)


def _child_main(
    data: Connection,
    control: Connection,
    name: str,
    factory: Callable[..., Any],
    kwargs: dict[str, Any],
    sample_method: str,
    sample_period_s: float,
    info_attrs: tuple[str, ...],
    log_level: int,
) -> None:
    """Entry point of a device process: build the device, report ready, stream samples until told to stop."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _make_fds_non_inheritable()
    _configure_child_logging(name, log_level)
    try:
        device = factory(**kwargs)
    except BaseException as exc:
        _send_quietly(data, ("init_error", _picklable(exc), traceback.format_exc()))
        return
    try:
        try:
            info = {attr: getattr(device, attr) for attr in info_attrs}
            sample = getattr(device, sample_method)
        except BaseException as exc:
            _send_quietly(data, ("init_error", _picklable(exc), traceback.format_exc()))
            return
        data.send(("ready", info))
        _stream(data, control, sample, sample_period_s)
    except (EOFError, BrokenPipeError, ConnectionResetError):
        logger.warning("The parent process went away; closing the %s device.", name)
    except Exception as exc:
        # Anything else (a sample that cannot be pickled, say) ends the stream; say why before "closed".
        logger.exception("The %s stream failed", name)
        _send_quietly(data, ("error", f"stream stopped: {type(exc).__name__}: {exc}"))
    finally:
        try:
            device.close()
        except Exception:
            logger.exception("Closing the %s device failed", name)
        _send_quietly(data, ("closed",))


def _stream(data: Connection, control: Connection, sample: Callable[[], Any], sample_period_s: float) -> None:
    """Push every new sample until a close request (returns) or the parent disappears (EOFError/OSError)."""
    last_timestamp: object = object()
    last_error_report = float("-inf")
    while True:
        started = time.monotonic()
        # recv() raises EOFError once the parent is gone, which ends the stream (and closes the device).
        if control.poll(0) and control.recv()[0] == "close":
            return
        try:
            value = sample()
            stamp = time.monotonic()
        except Exception as exc:
            now = time.monotonic()
            if now - last_error_report >= _ERROR_REPORT_INTERVAL_S:
                last_error_report = now
                data.send(("error", f"{type(exc).__name__}: {exc}"))
        else:
            timestamp = getattr(value, "timestamp", None)
            if timestamp is None or timestamp != last_timestamp:
                last_timestamp = timestamp
                data.send(("sample", value, stamp))
        remaining = sample_period_s - (time.monotonic() - started)
        if remaining > 0:
            control.poll(remaining)  # sleeps, but wakes at once for a close request


class DeviceStream(Generic[SampleT]):
    """A device running in its own process, whose samples stream to this process over a pipe.

    Construction spawns the process, starts the receiver thread, and returns at once, so several devices can
    start in parallel. Call ``wait_ready()`` (or ``wait_ready_all()`` for several streams at once) before
    ``latest()``. The stream is closed with ``close()`` (idempotent; ``close_all()`` closes several in parallel),
    which asks the child to close its device and escalates to SIGTERM/SIGKILL only if it does not.

    Attributes:
        name: a short label used in process names, log lines and error messages.
        info: the ``info_attrs`` the device reported when it came up (empty until then).
        stale_after_s: ``latest()`` raises ``TimeoutError`` once its newest sample was taken longer ago than this.
    """

    def __init__(
        self,
        name: str,
        factory: Callable[..., Any],
        kwargs: dict[str, Any] | None = None,
        *,
        sample_method: str,
        stale_after_s: float,
        sample_period_s: float = 0.001,
        info_attrs: tuple[str, ...] = (),
    ) -> None:
        """Spawn the device process and start receiving from it (non-blocking).

        Args:
            name: label for logs, errors and the process name.
            factory: an importable callable (usually the device class) the child calls as ``factory(**kwargs)``.
                The child imports it by reference, so it must live at module level.
            kwargs: keyword arguments for ``factory``; must be picklable.
            sample_method: the no-argument device method the child polls; each call returns one sample.
            stale_after_s: how old (by the child's clock) the newest sample may be before ``latest()`` raises.
            sample_period_s: how often the child polls. Samples are deduplicated by their ``timestamp``
                attribute, so a device producing samples faster than this has some skipped.
            info_attrs: device attributes the child reports once, when it is up (see ``info``).
        """
        self.name = name
        self.stale_after_s = stale_after_s
        self.info: dict[str, Any] = {}

        self._lock = threading.Lock()
        self._latest: SampleT | None = None
        self._latest_monotonic = 0.0  # the child's stamp of the newest sample
        self._sample_count = 0
        self._last_error: str | None = None
        self._init_error: tuple[BaseException, str] | None = None
        self._closed = False
        self._close_requested_at: float | None = None
        self._finished = False
        self._came_up = threading.Event()  # "ready" or "init_error" arrived
        self._fresh_sample = threading.Event()  # a sample arrived that was fresh when it arrived
        self._child_done = threading.Event()  # "closed" arrived, or the data pipe hit EOF

        self._data, child_data = _CONTEXT.Pipe(duplex=False)
        child_control, self._control = _CONTEXT.Pipe(duplex=False)
        self._process = _CONTEXT.Process(
            target=_child_main,
            args=(
                child_data,
                child_control,
                name,
                factory,
                dict(kwargs or {}),
                sample_method,
                sample_period_s,
                tuple(info_attrs),
                logging.getLogger().getEffectiveLevel(),
            ),
            name=f"yumi-{name}",
            daemon=True,
        )
        self._process.start()
        # Drop this process's copies of the child's ends, so EOF on the data pipe means the child is gone.
        child_data.close()
        child_control.close()
        # Receive from the start: a pipe no one drains fills with samples that would later be read as current.
        self._receiver = threading.Thread(target=self._receive, name=f"yumi-{name}-receiver", daemon=True)
        self._receiver.start()

    @property
    def pid(self) -> int | None:
        """The device process's PID."""
        return self._process.pid

    @property
    def sample_count(self) -> int:
        """How many samples have arrived so far (useful for measuring a stream's rate)."""
        with self._lock:
            return self._sample_count

    @property
    def last_error(self) -> str | None:
        """The most recent error the child reported from its sample call, if any."""
        with self._lock:
            return self._last_error

    # ------------------------------------------------------------------ receiving

    def _receive(self) -> None:
        while True:
            try:
                if not self._data.poll(_RECEIVER_POLL_S):
                    continue
                message = self._data.recv()
            except (EOFError, OSError):
                self._child_done.set()
                return
            kind = message[0]
            if kind == "sample":
                _, sample, stamp = message
                with self._lock:
                    self._latest = sample
                    self._latest_monotonic = stamp
                    self._sample_count += 1
                if time.monotonic() - stamp <= self.stale_after_s:
                    self._fresh_sample.set()
            elif kind == "error":
                with self._lock:
                    self._last_error = message[1]
            elif kind == "ready":
                self.info = message[1]
                self._came_up.set()
            elif kind == "init_error":
                with self._lock:
                    self._init_error = (message[1], message[2])
                self._came_up.set()
            elif kind == "closed":
                self._child_done.set()
                return

    # ------------------------------------------------------------------ startup

    def poll_ready(self, first_sample: bool = True) -> bool:
        """Non-blocking readiness check: True once up (and, by default, once a fresh sample has arrived).

        Raises:
            Exception: the device's construction error, with the child's traceback as ``__cause__``.
            RuntimeError: the device process exited before it was ready.
        """
        with self._lock:
            init_error = self._init_error
        if init_error is not None:
            exc, tb = init_error
            raise exc from RemoteTraceback(tb)
        if self._came_up.is_set() and (not first_sample or self._fresh_sample.is_set()):
            return True
        if self._child_done.is_set() or not self._process.is_alive():
            phase = "produced a sample" if self._came_up.is_set() else "came up"
            raise RuntimeError(
                f"The {self.name} device process exited before it {phase} (exit code {self._process.exitcode}); "
                f"last error: {self.last_error}"
            )
        return False

    def describe_pending(self, timeout_s: float) -> str:
        """Why the stream is not ready yet, for a timeout message."""
        if not self._came_up.is_set():
            return f"The {self.name} device did not come up within {timeout_s:.1f} s"
        return (
            f"The {self.name} device came up but produced no fresh sample within {timeout_s:.1f} s; "
            f"last error: {self.last_error}"
        )

    def wait_ready(self, timeout_s: float, first_sample: bool = True) -> dict[str, Any]:
        """Block until the device is up (and, by default, its first fresh sample has arrived); return ``info``.

        Waiting for the first sample is what makes ``latest()`` usable the moment this returns: "up" only means the
        device was constructed. Pass ``first_sample=False`` for a stream whose first sample the caller waits for
        itself, with a better error (the tracker wait in ``Yumi``). On any failure the stream is closed before the
        exception propagates.

        Raises:
            TimeoutError: the device did not come up, or produced no fresh sample, within ``timeout_s``.
            RuntimeError: the device process exited first.
            Exception: whatever the device's constructor raised, with the child's traceback as ``__cause__``.
        """
        wait_ready_all([(self, timeout_s, first_sample)])
        return self.info

    # ------------------------------------------------------------------ reading

    def latest(self) -> SampleT:
        """The newest sample. It is this stream's own unpickled copy, shared by every caller until the next one.

        Raises:
            RuntimeError: the stream is closed, or its device process exited.
            TimeoutError: the newest sample was taken longer ago than ``stale_after_s`` (by the child's clock), or
                none has arrived; the message carries the child's last error.
        """
        with self._lock:
            closed = self._closed
            sample, stamp = self._latest, self._latest_monotonic
            count, last_error = self._sample_count, self._last_error
        if closed:
            raise RuntimeError(f"The {self.name} stream is closed")
        # Checked after "closed", so a child that is done here exited on its own.
        if self._child_done.is_set() or not self._process.is_alive():
            raise RuntimeError(
                f"The {self.name} device process exited (exit code {self._process.exitcode}); last error: {last_error}"
            )
        age = time.monotonic() - stamp if sample is not None else float("inf")
        if sample is None or age > self.stale_after_s:
            age_text = "no sample yet" if sample is None else f"newest sample taken {age:.3f} s ago"
            raise TimeoutError(
                f"The {self.name} stream is stale ({age_text}, limit {self.stale_after_s:.3f} s, {count} samples "
                f"so far); last error: {last_error}"
            )
        return sample

    # ------------------------------------------------------------------ closing

    def request_close(self) -> None:
        """Mark the stream closed and ask the child to stop (non-blocking). ``finish_close()`` completes it."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._close_requested_at = time.monotonic()
        if self._came_up.is_set() and not self._child_done.is_set():
            try:
                self._control.send(("close",))
            except OSError:
                pass

    def finish_close(self) -> None:
        """Wait for the child to close its device and exit, escalating to SIGTERM/SIGKILL. Idempotent."""
        self.request_close()
        if self._finished:
            return
        self._finished = True
        streaming = self._came_up.is_set()
        if streaming:
            # The receiver keeps draining the data pipe meanwhile, so a child blocked sending a frame can finish,
            # see the request and close its device.
            assert self._close_requested_at is not None
            remaining = _CLOSE_CONFIRM_TIMEOUT_S - (time.monotonic() - self._close_requested_at)
            if not self._child_done.wait(max(0.0, remaining)):
                logger.warning(
                    "The %s device did not confirm close within %.1f s", self.name, _CLOSE_CONFIRM_TIMEOUT_S
                )
        # A device still being constructed is not reading its control pipe, so it is terminated outright; the OS
        # releases its CAN socket, video node and SteamVR connection.
        self._process.join(_JOIN_TIMEOUT_S if streaming else 0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(_JOIN_TIMEOUT_S)
        if self._process.is_alive():
            logger.warning("The %s device process ignored SIGTERM; killing it", self.name)
            self._process.kill()
            self._process.join(_JOIN_TIMEOUT_S)
        self._control.close()
        self._receiver.join(_RECEIVER_POLL_S * 5)
        self._data.close()

    def close(self) -> None:
        """Ask the child to close its device and exit; escalate to SIGTERM, then SIGKILL, if it does not. Idempotent."""
        self.request_close()
        self.finish_close()

    def __enter__(self) -> DeviceStream[SampleT]:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def wait_ready_all(requests: Sequence[tuple[DeviceStream[Any], float, bool]]) -> None:
    """Wait for several streams at once; each request is ``(stream, timeout_s, first_sample)``.

    The streams start in parallel, so they are waited on in parallel too: the first failure -- a construction
    error, a dead process, or one stream's own timeout -- is raised as soon as it happens, not after the slower
    streams come up. On any failure every stream in ``requests`` is closed before the exception propagates.
    """
    started = time.monotonic()
    pending = list(requests)
    try:
        while pending:
            still_pending = []
            for stream, timeout_s, first_sample in pending:
                if stream.poll_ready(first_sample):
                    continue
                if time.monotonic() - started >= timeout_s:
                    raise TimeoutError(stream.describe_pending(timeout_s))
                still_pending.append((stream, timeout_s, first_sample))
            pending = still_pending
            if pending:
                time.sleep(_WAIT_POLL_S)
    except BaseException:
        close_all([stream for stream, _, _ in requests])
        raise


def close_all(streams: Sequence[DeviceStream[Any]]) -> None:
    """Close several streams in parallel: ask every child to stop first, then wait for each in ``streams`` order.

    Every stream is closed even if an earlier one fails; the first failure is re-raised at the end.
    """
    first_error: BaseException | None = None
    for stream in streams:
        try:
            stream.request_close()
        except BaseException as exc:
            first_error = first_error or exc
    for stream in streams:
        try:
            stream.finish_close()
        except BaseException as exc:
            first_error = first_error or exc
    if first_error is not None:
        raise first_error
