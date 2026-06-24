"""Modern PyQt control panel for the LeRobot recorder.

Two stages in one window:

* a **Setup page** (landing): dataset name/dir/task, record **source** (teleop /
  dagger / eval), a **Continue collecting** (resume) toggle, and a status line
  (cameras detected, whether the dataset already exists). No robot connection yet.
* a **Collect page**: the live agentview (with wrist insets), a color status banner,
  a health strip (robot / cameras / disk / queue), live stats, and a review panel.

Pressing **START** resolves the dataset (create / resume / confirmed-overwrite),
opens the cameras + robot + dataset, switches to the Collect page, and (when
``auto_arm``) arms immediately so the next teleop engage records.

A **log panel** at the bottom is shared by both pages and shows the recorder's
important messages (link up/down, camera fallback, dataset saved, faults).
"""

from __future__ import annotations

import fcntl
import logging
import os
import tempfile
from collections import deque
from typing import IO, Optional

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from workstation.lerobot_recorder import theme
from workstation.lerobot_recorder.cameras import detect_cameras
from workstation.lerobot_recorder.config import RecorderConfig
from workstation.lerobot_recorder.dataset_writer import dataset_dir, dataset_info, remove_dataset_root
from workstation.lerobot_recorder.recorder import Recorder
from workstation.lerobot_recorder.sound import Cues
from workstation.lerobot_recorder.views import compose_agentview

logger = logging.getLogger(__name__)


_LOCK_PATH = os.path.join(tempfile.gettempdir(), "yam_recorder.lock")


def _acquire_singleton_lock() -> Optional[IO]:
    """Hold an exclusive lock so a second recorder can't fight over the cameras.

    Returns the open file (keep a reference to hold the lock for the process), or
    None if another live recorder already holds it."""
    f = open(_LOCK_PATH, "w")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    f.write(str(os.getpid()))
    f.flush()
    return f


def _np_to_pixmap(img: np.ndarray) -> QtGui.QPixmap:
    img = np.ascontiguousarray(img)
    h, w, _ = img.shape
    return QtGui.QPixmap.fromImage(QtGui.QImage(img.tobytes(), w, h, 3 * w, QtGui.QImage.Format_RGB888))


class _LogBuffer(logging.Handler):
    """Collect recorder log records (from any thread) for the GUI to drain.

    The recorder's components log from background threads, so we only buffer here
    (a deque is safe to append/popleft across threads) and let the GUI thread render
    them on its timer — never touching Qt widgets off the GUI thread.
    """

    def __init__(self, maxlen: int = 1000) -> None:
        super().__init__()
        self.records: "deque" = deque(maxlen=maxlen)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.records.append((record.levelno, self.format(record)))
        except Exception:
            pass


class RecorderGUI(QtWidgets.QWidget):
    def __init__(self, cfg: RecorderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.recorder: "Recorder | None" = None  # created on START (so the source picker applies)
        self.cues = Cues(enabled=True)
        self._review_idx = 0
        self._prev: dict = {}

        # Surface the recorder's important log messages in-window. Attach to the
        # package logger so cameras / robot link / dataset writer all flow in.
        self._logbuf = _LogBuffer()
        self._logbuf.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
        pkg_logger = logging.getLogger("workstation.lerobot_recorder")
        pkg_logger.setLevel(logging.INFO)
        pkg_logger.propagate = False  # don't also emit via the root logger (double lines)
        if not any(isinstance(h, logging.StreamHandler) for h in pkg_logger.handlers):
            pkg_logger.addHandler(logging.StreamHandler())  # keep console output too
        pkg_logger.addHandler(self._logbuf)

        self.setWindowTitle("YAM · LeRobot Recorder")
        self.setStyleSheet(theme.QSS)
        self._build()
        self._update_setup_status(rescan=True)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(100)  # 10 Hz UI
        self._review_timer = QtCore.QTimer(self)
        self._review_timer.timeout.connect(self._advance_review)
        self._review_timer.start(50)

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        self.banner = QtWidgets.QLabel("SETUP")
        self.banner.setAlignment(QtCore.Qt.AlignCenter)
        self.banner.setStyleSheet(theme.banner_style(theme.IDLE))

        self.stack = QtWidgets.QStackedWidget()
        self.setup_page = self._build_setup_page()
        self.collect_page = self._build_collect_page()
        self.stack.addWidget(self.setup_page)
        self.stack.addWidget(self.collect_page)

        # log panel (shared by both pages): recorder's important messages
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)  # cap memory; old lines scroll off
        self.log_view.setMinimumHeight(220)
        self.log_view.setStyleSheet(
            f"background:#0d1117;border:1px solid #30363d;border-radius:6px;color:{theme.MUTED};"
            "font-family:monospace;font-size:18px;"
        )

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)
        root.addWidget(self.banner)
        root.addWidget(self.stack, 2)
        root.addWidget(self.log_view, 1)

    def _build_setup_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        box = QtWidgets.QGroupBox("Session setup")
        form = QtWidgets.QFormLayout(box)
        form.setSpacing(10)

        self.repo_edit = QtWidgets.QLineEdit(self.cfg.repo_id)
        self.root_edit = QtWidgets.QLineEdit(self.cfg.root)
        self.repo_edit.textChanged.connect(lambda *_: self._update_setup_status())
        self.root_edit.textChanged.connect(lambda *_: self._update_setup_status())

        self.task_combo = QtWidgets.QComboBox()
        self.task_combo.setEditable(True)
        seen = []
        for t in [self.cfg.task, *self.cfg.tasks]:
            if t and t not in seen:
                seen.append(t)
        self.task_combo.addItems(seen)
        self.task_combo.setCurrentText(self.cfg.task)

        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.addItems(["teleop", "dagger", "eval"])
        idx = self.source_combo.findText(self.cfg.record_source)
        self.source_combo.setCurrentIndex(idx if idx >= 0 else 0)

        self.resume_check = QtWidgets.QCheckBox("Continue collecting (append to the existing dataset)")
        self.resume_check.setChecked(self.cfg.resume)
        self.resume_check.toggled.connect(lambda *_: self._update_setup_status())

        form.addRow("repo_id", self.repo_edit)
        form.addRow("root", self.root_edit)
        form.addRow("task", self.task_combo)
        form.addRow("source", self.source_combo)
        form.addRow("", self.resume_check)

        self.setup_status = QtWidgets.QLabel()
        self.setup_status.setTextFormat(QtCore.Qt.RichText)
        self.setup_status.setWordWrap(True)

        self.rescan_btn = QtWidgets.QPushButton("Re-scan cameras")
        self.rescan_btn.clicked.connect(lambda: self._update_setup_status(rescan=True))
        self.start_btn = QtWidgets.QPushButton("START")
        self.start_btn.setObjectName("start")
        self.start_btn.clicked.connect(self._on_start)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.rescan_btn)
        row.addStretch(1)
        row.addWidget(self.start_btn)

        lay = QtWidgets.QVBoxLayout(page)
        lay.setSpacing(12)
        lay.addWidget(box)
        lay.addWidget(self.setup_status)
        lay.addStretch(1)
        lay.addLayout(row)
        return page

    def _build_collect_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()

        self.health = QtWidgets.QLabel()
        self.health.setTextFormat(QtCore.Qt.RichText)
        self.health.setStyleSheet("font-size:22px;")  # bigger health/queue strip
        self.stats = QtWidgets.QLabel("episodes 0")
        self.stats.setStyleSheet(f"color:{theme.MUTED};font-size:21px;")
        self.estop_btn = QtWidgets.QPushButton("■ E-STOP")
        self.estop_btn.setObjectName("estop")
        self.estop_btn.setStyleSheet(f"background:{theme.BAD};color:white;font-weight:700;")
        self.estop_btn.setCheckable(True)
        self.estop_btn.toggled.connect(self._on_estop)
        strip = QtWidgets.QHBoxLayout()
        strip.addWidget(self.health)
        strip.addStretch(1)
        strip.addWidget(self.stats)
        strip.addStretch(1)
        strip.addWidget(self.estop_btn)

        self.collect_btn = QtWidgets.QPushButton("Start collection")
        self.collect_btn.setCheckable(True)
        self.collect_btn.setEnabled(False)
        self.collect_btn.clicked.connect(self._on_collect)

        # primary operator view: agentview + wrist insets composited into one frame
        self.live_lbl = QtWidgets.QLabel("camera view")
        self.live_lbl.setMinimumHeight(360)
        self.live_lbl.setAlignment(QtCore.Qt.AlignCenter)
        # Ignored size policy so the scaled pixmap never feeds back into the layout
        # (otherwise the label keeps growing each frame).
        self.live_lbl.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.live_lbl.setStyleSheet(f"background:#000;border:1px solid #30363d;border-radius:8px;color:{theme.MUTED};")

        # review panel
        self.review_box = QtWidgets.QGroupBox("Episode review")
        rv = QtWidgets.QVBoxLayout(self.review_box)
        self.review_lbl = QtWidgets.QLabel("(no episode pending)")
        self.review_lbl.setMinimumHeight(180)
        self.review_lbl.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.review_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.review_lbl.setStyleSheet(
            f"background:#000;border:1px solid #30363d;border-radius:6px;color:{theme.MUTED};"
        )
        self.review_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.review_slider.setEnabled(False)
        self.review_slider.sliderMoved.connect(self._scrub)
        self.keep_btn = QtWidgets.QPushButton("✓ Keep success  [S]")
        self.keepfail_btn = QtWidgets.QPushButton("Keep fail  [F]")
        self.del_btn = QtWidgets.QPushButton("✗ Delete  [D]")
        self.keep_btn.clicked.connect(lambda: self._on_keep("success"))
        self.keepfail_btn.clicked.connect(lambda: self._on_keep("fail"))
        self.del_btn.clicked.connect(self._on_delete)
        rb = QtWidgets.QHBoxLayout()
        for b in (self.keep_btn, self.keepfail_btn, self.del_btn):
            b.setEnabled(False)
            rb.addWidget(b)
        rv.addWidget(self.review_lbl)
        rv.addWidget(self.review_slider)
        rv.addLayout(rb)
        # Auto-save sessions (review_before_save: false) never use this panel — hide it
        # and give the room to the log. It stays for review-mode sessions.
        self.review_box.setVisible(self.cfg.review_before_save)

        self.hint = QtWidgets.QLabel("space toggles collection · S/F keep · D delete")
        self.hint.setStyleSheet(f"color:{theme.MUTED};")

        lay = QtWidgets.QVBoxLayout(page)
        lay.setSpacing(12)
        lay.addLayout(strip)
        lay.addWidget(self.collect_btn)
        lay.addWidget(self.live_lbl, 1)
        lay.addWidget(self.review_box)
        lay.addWidget(self.hint)
        return page

    # ------------------------------------------------------------------ setup status
    def _update_setup_status(self, rescan: bool = False) -> None:
        """Refresh the camera-detected + dataset-exists line on the setup page."""
        if rescan or not hasattr(self, "_cam_detect"):
            self._cam_detect = detect_cameras(self.cfg)
        cam = self._cam_detect
        cam_ok = cam["found"] == cam["total"]
        cam_col = theme.OK if cam_ok else theme.BAD
        cam_txt = f'<span style="color:{cam_col};">●</span> cameras {cam["found"]}/{cam["total"]}'
        if cam.get("missing"):
            cam_txt += f' (missing: {", ".join(cam["missing"])})'

        root = self.root_edit.text().strip() if hasattr(self, "root_edit") else self.cfg.root
        repo = self.repo_edit.text().strip() if hasattr(self, "repo_edit") else self.cfg.repo_id
        ds_dir = dataset_dir(root, repo)
        info = dataset_info(ds_dir)
        where = f' <span style="color:{theme.MUTED};">→ {ds_dir}</span>'
        if not info["exists"]:
            ds_txt = f'<span style="color:{theme.OK};">●</span> dataset: new (will be created){where}'
        else:
            n = info["episodes"]
            ntxt = f"{n} episode(s)" if n is not None else "existing data"
            if self.resume_check.isChecked():
                ds_txt = f'<span style="color:{theme.ACCENT};">●</span> dataset: exists ({ntxt}) — will append{where}'
            else:
                ds_txt = f'<span style="color:{theme.WARN};">●</span> dataset: exists ({ntxt}) — START will offer to overwrite{where}'
        self.setup_status.setText(cam_txt + "<br>" + ds_txt)

    # ------------------------------------------------------------------ actions
    def _on_start(self) -> None:
        cfg = self.cfg
        cfg.repo_id = self.repo_edit.text().strip()
        cfg.root = self.root_edit.text().strip()
        cfg.task = self.task_combo.currentText().strip()
        cfg.record_source = self.source_combo.currentText().strip()
        cfg.resume = self.resume_check.isChecked()

        # Single-instance guard: two recorders fighting over the cameras causes the
        # flapping/freeze, so refuse to start a second one with a clear message.
        if not cfg.mock and getattr(self, "_lock", None) is None:
            self._lock = _acquire_singleton_lock()
            if self._lock is None:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Recorder already running",
                    "Another YAM recorder appears to be running and holds the cameras.\n"
                    "Close it first, then press START again.",
                )
                return

        ds_dir = dataset_dir(cfg.root, cfg.repo_id)
        info = dataset_info(ds_dir)
        if info["exists"] and not cfg.resume:
            if not self._confirm_overwrite(ds_dir, info["episodes"]):
                return
            try:
                remove_dataset_root(ds_dir)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Delete failed", str(e))
                return
        elif not info["exists"]:
            cfg.resume = False  # nothing to resume

        self.recorder = Recorder(cfg)
        self.start_btn.setEnabled(False)
        try:
            self.recorder.start()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Start failed", str(e))
            self.recorder = None
            self.start_btn.setEnabled(True)
            return

        self._prev = self.recorder.get_status()
        self.stack.setCurrentWidget(self.collect_page)
        self.collect_btn.setEnabled(True)
        if cfg.auto_arm:  # arm immediately so the next teleop engage records
            self.collect_btn.setChecked(True)
            self._on_collect()

    def _confirm_overwrite(self, root: str, episodes: Optional[int]) -> bool:
        """Two sequential confirms before deleting an existing dataset."""
        ntxt = f"{episodes} episode(s)" if episodes is not None else "existing data"
        first = QtWidgets.QMessageBox.warning(
            self,
            "Folder already exists",
            f"{os.path.expanduser(root)}\nalready contains {ntxt}.\n\n"
            "Overwrite it and start a fresh dataset?\n(Check 'Continue collecting' instead to append.)",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Cancel,
        )
        if first != QtWidgets.QMessageBox.Yes:
            return False
        second = QtWidgets.QMessageBox.critical(
            self,
            "Confirm overwrite",
            "This permanently DELETES the existing dataset. This cannot be undone.\n\nAre you sure?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Cancel,
        )
        return second == QtWidgets.QMessageBox.Yes

    def _on_collect(self) -> None:
        if self.recorder is None:
            return
        if self.collect_btn.isChecked():
            self.cfg.task = self.task_combo.currentText().strip()
            self.recorder.arm()
            self.collect_btn.setText("Stop collection")
        else:
            self.recorder.disarm()
            self.collect_btn.setText("Start collection")

    def _on_keep(self, outcome: str) -> None:
        if self.recorder is not None:
            self.recorder.keep_episode(outcome=outcome)

    def _on_delete(self) -> None:
        if self.recorder is not None:
            self.recorder.delete_episode()

    def _on_estop(self, engaged: bool) -> None:
        if self.recorder is not None:
            self.recorder.set_estop(engaged)
        self.estop_btn.setText("■ E-STOP (engaged)" if engaged else "■ E-STOP")

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key = event.key()
        if self.recorder is not None and self.stack.currentWidget() is self.collect_page:
            if key == QtCore.Qt.Key_Space and self.collect_btn.isEnabled():
                self.collect_btn.toggle()
                self._on_collect()
            elif self.recorder.get_status()["pending"]:
                if key == QtCore.Qt.Key_S:
                    self._on_keep("success")
                elif key == QtCore.Qt.Key_F:
                    self._on_keep("fail")
                elif key == QtCore.Qt.Key_D:
                    self._on_delete()
        super().keyPressEvent(event)

    # ------------------------------------------------------------------ refresh
    def _refresh(self) -> None:
        self._drain_log()
        if self.recorder is None:
            return  # setup page: status updated on field-change / re-scan
        st = self.recorder.get_status()
        self._update_banner(st)
        self._update_health(st)
        self._update_stats(st)
        self._cue_transitions(self._prev, st)

        for b in (self.keep_btn, self.keepfail_btn, self.del_btn):
            b.setEnabled(st["pending"])
        if not st["pending"]:
            self.review_lbl.setText("(no episode pending)")
            self.review_slider.setEnabled(False)

        images = self.recorder.get_last_images()
        composite = compose_agentview(images, agent_key=self.cfg.review_cam)
        if isinstance(composite, np.ndarray) and composite.ndim == 3:
            self.live_lbl.setPixmap(_np_to_pixmap(composite).scaled(self.live_lbl.size(), QtCore.Qt.KeepAspectRatio))

        self._prev = st

    def _drain_log(self) -> None:
        """Render any buffered log records (called on the GUI thread)."""
        colors = {logging.ERROR: theme.BAD, logging.WARNING: theme.WARN}
        appended = False
        while self._logbuf.records:
            try:
                level, msg = self._logbuf.records.popleft()
            except IndexError:
                break
            color = colors.get(level, theme.MUTED)
            self.log_view.appendHtml(f'<span style="color:{color};">{msg}</span>')
            appended = True
        if appended:
            self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

    def _update_banner(self, st: dict) -> None:
        if not st.get("disk_ok", True):
            text, color = "⚠ LOW DISK — not saving", theme.STATE_COLORS["ERROR"]
        elif not (st["cam_ok"] and st.get("robot_ok", True)):
            text, color = "⚠ DEVICE FAULT", theme.STATE_COLORS["ERROR"]
        elif st["pending"]:
            text, color = "REVIEW — keep [S/F] or delete [D]", theme.STATE_COLORS["REVIEW"]
        elif st["recording"]:
            text, color = "● REC", theme.STATE_COLORS["REC"]
        elif st["armed"]:
            text, color = f"ARMED · {st['teleop']}", theme.STATE_COLORS["ARMED"]
        else:
            text, color = "IDLE" if st["running"] else "not started", theme.STATE_COLORS["IDLE"]
        self.banner.setText(text)
        self.banner.setStyleSheet(theme.banner_style(color))

    def _update_health(self, st: dict) -> None:
        cam = theme.dot(st["cam_ok"])
        rob = theme.dot(st.get("robot_ok", False))
        disk = theme.dot(st.get("disk_ok", True))
        w = st.get("writer") or {}
        workers = w.get("workers", 1)
        saved = w.get("saved", 0)
        queued = w.get("queued", st.get("queue", 0))
        # idle (green) vs encoding-one-now (amber ⟳, with episode + frame count) vs backlog.
        if w.get("saving"):
            idx, fr = w.get("saving_index"), w.get("saving_frames", 0)
            save_txt = f'<span style="color:{theme.WARN};">⟳ encoding #{idx} ({fr} frames)</span> · queued {queued}'
        elif queued:
            save_txt = f'<span style="color:{theme.WARN};">● queued {queued}</span>'
        else:
            save_txt = f'<span style="color:{theme.OK};">● idle</span>'
        self.health.setText(
            f"{rob} robot &nbsp;&nbsp; {cam} cameras &nbsp;&nbsp; {disk} disk "
            f"&nbsp;&nbsp;|&nbsp;&nbsp; writer: {workers} worker · saved {saved} · {save_txt}"
        )

    def _update_stats(self, st: dict) -> None:
        kept, suc, fail, disc = st["kept"], st["success"], st["fail"], st["discarded"]
        rate = f"{100 * suc / max(suc + fail, 1):.0f}%" if (suc + fail) else "—"
        self.stats.setText(
            f"episodes {st['episodes']} · kept {kept} (✓{suc} ✗{fail}) · discarded {disc} · "
            f"success {rate} · frames {st['frames']}"
        )

    def _cue_transitions(self, prev: dict, cur: dict) -> None:
        if cur["recording"] and not prev.get("recording"):
            self.cues.play("start")
        if cur["success"] > prev.get("success", 0):
            self.cues.play("success")
        if cur["fail"] > prev.get("fail", 0):
            self.cues.play("fail")
        if cur["discarded"] > prev.get("discarded", 0):
            self.cues.play("delete")
        healthy_now = cur["cam_ok"] and cur.get("robot_ok", True)
        healthy_was = prev.get("cam_ok", True) and prev.get("robot_ok", True)
        if healthy_was and not healthy_now:
            self.cues.play("error")

    def _scrub(self, value: int) -> None:
        if self.recorder is None:
            return
        frames = self.recorder.get_review_frames()
        if frames:
            self._review_idx = max(0, min(value, len(frames) - 1))
            self._show_review(frames[self._review_idx])

    def _advance_review(self) -> None:
        if self.recorder is None or not self.recorder.get_status()["pending"]:
            self._review_idx = 0
            return
        frames = self.recorder.get_review_frames()
        if not frames:
            return
        self.review_slider.setEnabled(True)
        self.review_slider.setMaximum(len(frames) - 1)
        if not self.review_slider.isSliderDown():
            self._review_idx = (self._review_idx + 1) % len(frames)
            self.review_slider.setValue(self._review_idx)
            self._show_review(frames[self._review_idx])

    def _show_review(self, img: np.ndarray) -> None:
        if isinstance(img, np.ndarray) and img.ndim == 3:
            self.review_lbl.setPixmap(_np_to_pixmap(img).scaled(self.review_lbl.size(), QtCore.Qt.KeepAspectRatio))

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.recorder is not None:
            self.recorder.shutdown()
        lock = getattr(self, "_lock", None)
        if lock is not None:
            try:
                lock.close()  # releases the flock
            except Exception:
                pass
        super().closeEvent(event)
