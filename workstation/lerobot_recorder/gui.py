"""Modern PyQt control panel for the LeRobot recorder.

Designed for an operator who watches the **robot**, not the screen:

* a big **color status banner** (IDLE / ARMED / REC / REVIEW / fault) for peripheral
  feedback, plus **audio cues** on episode start and keep/fail/delete,
* a **health strip** (robot link / cameras / save queue),
* **live stats** (kept ✓/✗, discarded, success rate),
* the **agentview composited with the wrist insets** as the primary view,
* a **task template** combo for quick language-instruction switching (the active
  task persists until you change it),
* labeling by mouse, **keyboard** (S=success, F=fail, D=delete), or leader buttons,
  with a review scrubber.
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from workstation.lerobot_recorder import theme
from workstation.lerobot_recorder.config import RecorderConfig
from workstation.lerobot_recorder.recorder import Recorder
from workstation.lerobot_recorder.sound import Cues
from workstation.lerobot_recorder.views import compose_agentview


def _np_to_pixmap(img: np.ndarray) -> QtGui.QPixmap:
    img = np.ascontiguousarray(img)
    h, w, _ = img.shape
    return QtGui.QPixmap.fromImage(QtGui.QImage(img.tobytes(), w, h, 3 * w, QtGui.QImage.Format_RGB888))


class RecorderGUI(QtWidgets.QWidget):
    def __init__(self, cfg: RecorderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.recorder = Recorder(cfg)
        self.cues = Cues(enabled=True)
        self._previews: dict = {}
        self._review_idx = 0
        self._prev = self.recorder.get_status()
        self.setWindowTitle("YAM · LeRobot Recorder")
        self.setStyleSheet(theme.QSS)
        self._build()

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(100)  # 10 Hz UI
        self._review_timer = QtCore.QTimer(self)
        self._review_timer.timeout.connect(self._advance_review)
        self._review_timer.start(50)

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        self.banner = QtWidgets.QLabel("IDLE")
        self.banner.setAlignment(QtCore.Qt.AlignCenter)
        self.banner.setStyleSheet(theme.banner_style(theme.IDLE))

        self.health = QtWidgets.QLabel()
        self.health.setTextFormat(QtCore.Qt.RichText)
        self.stats = QtWidgets.QLabel("episodes 0")
        self.stats.setStyleSheet(f"color:{theme.MUTED};")
        strip = QtWidgets.QHBoxLayout()
        strip.addWidget(self.health)
        strip.addStretch(1)
        strip.addWidget(self.stats)

        # session config + task templates
        form = QtWidgets.QFormLayout()
        self.repo_edit = QtWidgets.QLineEdit(self.cfg.repo_id)
        self.root_edit = QtWidgets.QLineEdit(self.cfg.root)
        self.task_combo = QtWidgets.QComboBox()
        self.task_combo.setEditable(True)
        seen = []
        for t in [self.cfg.task, *self.cfg.tasks]:
            if t and t not in seen:
                seen.append(t)
        self.task_combo.addItems(seen)
        self.task_combo.setCurrentText(self.cfg.task)
        self.task_combo.currentTextChanged.connect(self._on_task)
        form.addRow("repo_id", self.repo_edit)
        form.addRow("root", self.root_edit)
        form.addRow("task", self.task_combo)

        self.start_btn = QtWidgets.QPushButton("Start")
        self.start_btn.clicked.connect(self._on_start)
        self.collect_btn = QtWidgets.QPushButton("Start collection")
        self.collect_btn.setCheckable(True)
        self.collect_btn.setEnabled(False)
        self.collect_btn.clicked.connect(self._on_collect)
        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(self.start_btn)
        btns.addWidget(self.collect_btn)

        # primary operator view: agentview + wrist insets
        self.live_lbl = QtWidgets.QLabel("camera view")
        self.live_lbl.setMinimumHeight(320)
        self.live_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.live_lbl.setStyleSheet(f"background:#000;border:1px solid #30363d;border-radius:8px;color:{theme.MUTED};")

        previews = QtWidgets.QHBoxLayout()
        for cam in self.cfg.cameras:
            box = QtWidgets.QVBoxLayout()
            cap = QtWidgets.QLabel(cam.key)
            cap.setStyleSheet(f"color:{theme.MUTED};")
            lbl = QtWidgets.QLabel()
            lbl.setFixedSize(176, 132)
            lbl.setStyleSheet("background:#000;border:1px solid #30363d;border-radius:6px;")
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            box.addWidget(cap)
            box.addWidget(lbl)
            previews.addLayout(box)
            self._previews[cam.key] = lbl
        previews.addStretch(1)

        # review panel
        self.review_box = QtWidgets.QGroupBox("Episode review")
        rv = QtWidgets.QVBoxLayout(self.review_box)
        self.review_lbl = QtWidgets.QLabel("(no episode pending)")
        self.review_lbl.setMinimumHeight(180)
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

        self.hint = QtWidgets.QLabel("space toggles collection · S/F keep · D delete")
        self.hint.setStyleSheet(f"color:{theme.MUTED};")

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)
        root.addWidget(self.banner)
        root.addLayout(strip)
        root.addLayout(form)
        root.addLayout(btns)
        root.addWidget(self.live_lbl)
        root.addLayout(previews)
        root.addWidget(self.review_box)
        root.addWidget(self.hint)

    # ------------------------------------------------------------------ actions
    def _on_task(self, text: str) -> None:
        self.cfg.task = text.strip()  # active instruction persists until changed again

    def _on_start(self) -> None:
        self.cfg.repo_id = self.repo_edit.text().strip()
        self.cfg.root = self.root_edit.text().strip()
        self.cfg.task = self.task_combo.currentText().strip()
        for w in (self.repo_edit, self.root_edit, self.start_btn):
            w.setEnabled(False)
        try:
            self.recorder.start()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Start failed", str(e))
            for w in (self.repo_edit, self.root_edit, self.start_btn):
                w.setEnabled(True)
            return
        self.collect_btn.setEnabled(True)

    def _on_collect(self) -> None:
        if self.collect_btn.isChecked():
            self.cfg.task = self.task_combo.currentText().strip()
            self.recorder.arm()
            self.collect_btn.setText("Stop collection")
        else:
            self.recorder.disarm()
            self.collect_btn.setText("Start collection")

    def _on_keep(self, outcome: str) -> None:
        self.recorder.keep_episode(outcome=outcome)

    def _on_delete(self) -> None:
        self.recorder.delete_episode()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key = event.key()
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
        for key, img in images.items():
            lbl = self._previews.get(key)
            if lbl is not None and isinstance(img, np.ndarray) and img.ndim == 3:
                lbl.setPixmap(_np_to_pixmap(img).scaled(lbl.size(), QtCore.Qt.KeepAspectRatio))
        composite = compose_agentview(images, agent_key=self.cfg.review_cam)
        if isinstance(composite, np.ndarray) and composite.ndim == 3:
            self.live_lbl.setPixmap(_np_to_pixmap(composite).scaled(self.live_lbl.size(), QtCore.Qt.KeepAspectRatio))

        self._prev = st

    def _update_banner(self, st: dict) -> None:
        if not (st["cam_ok"] and st.get("robot_ok", True)):
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
        q = st.get("queue", 0)
        qcol = theme.OK if q <= 2 else theme.WARN
        self.health.setText(
            f'{rob} robot &nbsp;&nbsp; {cam} cameras &nbsp;&nbsp; <span style="color:{qcol};">●</span> save queue {q}'
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
        frames = self.recorder.get_review_frames()
        if frames:
            self._review_idx = max(0, min(value, len(frames) - 1))
            self._show_review(frames[self._review_idx])

    def _advance_review(self) -> None:
        if not self.recorder.get_status()["pending"]:
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
        self.recorder.shutdown()
        super().closeEvent(event)
