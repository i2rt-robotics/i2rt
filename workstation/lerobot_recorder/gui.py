"""PyQt control panel for the LeRobot recorder.

Workflow:

1. Set ``repo_id`` / ``root`` / ``task`` (language instruction).
2. **Start** — opens cameras, the remote ROS subscriber, and the dataset; the
   record loop runs but does not save yet.
3. **Start collection** — arms the auto-gate: an episode now begins the instant
   teleop becomes ENGAGED and saves when homing returns to IDLE. **Stop
   collection** disarms (a partial episode is discarded).

Live status shows the teleop state, a REC indicator, the episode count, and a
small preview of each camera.
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from workstation.lerobot_recorder.config import RecorderConfig
from workstation.lerobot_recorder.recorder import Recorder


class RecorderGUI(QtWidgets.QWidget):
    def __init__(self, cfg: RecorderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.recorder = Recorder(cfg)
        self._previews: dict = {}
        self.setWindowTitle("YAM ↔ LeRobot Recorder")
        self._build()

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(100)  # 10 Hz UI refresh

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        form = QtWidgets.QFormLayout()
        self.repo_edit = QtWidgets.QLineEdit(self.cfg.repo_id)
        self.root_edit = QtWidgets.QLineEdit(self.cfg.root)
        self.task_edit = QtWidgets.QLineEdit(self.cfg.task)
        form.addRow("repo_id", self.repo_edit)
        form.addRow("root", self.root_edit)
        form.addRow("task (instruction)", self.task_edit)
        form.addRow("fps", QtWidgets.QLabel(f"{self.cfg.fps}  (mock={self.cfg.mock})"))

        self.start_btn = QtWidgets.QPushButton("Start")
        self.start_btn.clicked.connect(self._on_start)
        self.collect_btn = QtWidgets.QPushButton("Start collection")
        self.collect_btn.setEnabled(False)
        self.collect_btn.setCheckable(True)
        self.collect_btn.clicked.connect(self._on_collect)
        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(self.start_btn)
        btns.addWidget(self.collect_btn)

        self.state_lbl = QtWidgets.QLabel("teleop: —")
        self.rec_lbl = QtWidgets.QLabel("● IDLE")
        self.rec_lbl.setStyleSheet("font-weight:bold;")
        self.eps_lbl = QtWidgets.QLabel("episodes: 0   frames: 0")
        status = QtWidgets.QHBoxLayout()
        status.addWidget(self.state_lbl)
        status.addStretch(1)
        status.addWidget(self.rec_lbl)
        status.addStretch(1)
        status.addWidget(self.eps_lbl)

        previews = QtWidgets.QHBoxLayout()
        for cam in self.cfg.cameras:
            box = QtWidgets.QVBoxLayout()
            lbl = QtWidgets.QLabel()
            lbl.setFixedSize(213, 160)
            lbl.setStyleSheet("background:#111;border:1px solid #333;")
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            box.addWidget(QtWidgets.QLabel(cam.key))
            box.addWidget(lbl)
            previews.addLayout(box)
            self._previews[cam.key] = lbl

        root = QtWidgets.QVBoxLayout(self)
        root.addLayout(form)
        root.addLayout(btns)
        root.addWidget(self._hline())
        root.addLayout(status)
        root.addLayout(previews)

    @staticmethod
    def _hline() -> QtWidgets.QFrame:
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        return line

    # ------------------------------------------------------------------ actions
    def _on_start(self) -> None:
        # push edited fields into the config, lock them, open everything
        self.cfg.repo_id = self.repo_edit.text().strip()
        self.cfg.root = self.root_edit.text().strip()
        self.cfg.task = self.task_edit.text().strip()
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
            self.cfg.task = self.task_edit.text().strip()  # allow re-editing instruction between episodes
            self.recorder.arm()
            self.collect_btn.setText("Stop collection")
        else:
            self.recorder.disarm()
            self.collect_btn.setText("Start collection")

    # ------------------------------------------------------------------ refresh
    def _refresh(self) -> None:
        st = self.recorder.get_status()
        self.state_lbl.setText(f"teleop: {st['teleop']}")
        self.eps_lbl.setText(f"episodes: {st['episodes']}   frames: {st['frames']}")
        if st["recording"]:
            self.rec_lbl.setText("● REC")
            self.rec_lbl.setStyleSheet("color:#e44;font-weight:bold;")
        else:
            self.rec_lbl.setText("● ARMED" if st["armed"] else "● IDLE")
            self.rec_lbl.setStyleSheet("color:#888;font-weight:bold;")

        for key, img in self.recorder.get_last_images().items():
            lbl = self._previews.get(key)
            if lbl is not None and isinstance(img, np.ndarray) and img.ndim == 3:
                h, w, _ = img.shape
                qimg = QtGui.QImage(img.tobytes(), w, h, 3 * w, QtGui.QImage.Format_RGB888)
                lbl.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(lbl.size(), QtCore.Qt.KeepAspectRatio))

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.recorder.shutdown()
        super().closeEvent(event)
