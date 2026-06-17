"""PyQt panel to review a recorded dataset and replay it onto the robot.

Load a dataset, pick an episode, and Play — the frames are shown and (if "Send to
robot" is on) each frame's action is sent to the YAM wrapper server (over portal)
so the robot follows the dataset. The robot side must be running
``scripts/yam wrapper``.
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from workstation.lerobot_recorder.config import RecorderConfig
from workstation.lerobot_recorder.dataset_reader import DatasetReader
from workstation.lerobot_recorder.replay_controller import ReplayController


def _np_to_pixmap(img: np.ndarray) -> QtGui.QPixmap:
    img = np.ascontiguousarray(img)
    h, w, _ = img.shape
    return QtGui.QPixmap.fromImage(QtGui.QImage(img.tobytes(), w, h, 3 * w, QtGui.QImage.Format_RGB888))


class ReplayGUI(QtWidgets.QWidget):
    def __init__(self, cfg: RecorderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.reader = DatasetReader(cfg.repo_id, cfg.root, display_cam=cfg.review_cam, mock=cfg.mock)
        self.controller: ReplayController | None = None
        self.setWindowTitle("YAM ↔ LeRobot Replay")
        self._build()
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(33)  # ~30 Hz UI refresh

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        form = QtWidgets.QFormLayout()
        self.repo_edit = QtWidgets.QLineEdit(self.cfg.repo_id)
        self.root_edit = QtWidgets.QLineEdit(self.cfg.root)
        form.addRow("repo_id", self.repo_edit)
        form.addRow("root", self.root_edit)

        self.load_btn = QtWidgets.QPushButton("Load")
        self.load_btn.clicked.connect(self._on_load)
        self.episode_cb = QtWidgets.QComboBox()
        self.episode_cb.setEnabled(False)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.load_btn)
        top.addWidget(QtWidgets.QLabel("episode"))
        top.addWidget(self.episode_cb, 1)

        self.view = QtWidgets.QLabel("(load a dataset)")
        self.view.setFixedHeight(280)
        self.view.setAlignment(QtCore.Qt.AlignCenter)
        self.view.setStyleSheet("background:#111;border:1px solid #333;")

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setEnabled(False)

        self.play_btn = QtWidgets.QPushButton("Play")
        self.pause_btn = QtWidgets.QPushButton("Pause")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.play_btn.clicked.connect(self._on_play)
        self.pause_btn.clicked.connect(self._on_pause)
        self.stop_btn.clicked.connect(self._on_stop)
        self.send_cb = QtWidgets.QCheckBox("Send to robot")
        self.speed = QtWidgets.QDoubleSpinBox()
        self.speed.setRange(0.1, 4.0)
        self.speed.setSingleStep(0.1)
        self.speed.setValue(1.0)
        self.speed.setPrefix("x ")
        ctl = QtWidgets.QHBoxLayout()
        for w in (self.play_btn, self.pause_btn, self.stop_btn):
            ctl.addWidget(w)
        ctl.addStretch(1)
        ctl.addWidget(QtWidgets.QLabel("speed"))
        ctl.addWidget(self.speed)
        ctl.addWidget(self.send_cb)

        self.status = QtWidgets.QLabel("—")
        for w in (self.play_btn, self.pause_btn, self.stop_btn, self.send_cb, self.speed):
            w.setEnabled(False)

        root = QtWidgets.QVBoxLayout(self)
        root.addLayout(form)
        root.addLayout(top)
        root.addWidget(self.view)
        root.addWidget(self.slider)
        root.addLayout(ctl)
        root.addWidget(self.status)

    # ------------------------------------------------------------------ actions
    def _on_load(self) -> None:
        self.cfg.repo_id = self.repo_edit.text().strip()
        self.cfg.root = self.root_edit.text().strip()
        self.reader = DatasetReader(self.cfg.repo_id, self.cfg.root, display_cam=self.cfg.review_cam, mock=self.cfg.mock)
        try:
            self.reader.load()
            self.controller = ReplayController(self.reader, self.cfg)
            self.controller.connect()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(e))
            return
        self.episode_cb.clear()
        self.episode_cb.addItems([f"ep {e}  ({self.reader.episode_length(e)} frames)" for e in range(self.reader.num_episodes)])
        for w in (self.episode_cb, self.play_btn, self.pause_btn, self.stop_btn, self.send_cb, self.speed):
            w.setEnabled(True)
        self.status.setText(f"loaded {self.reader.num_episodes} episodes @ {self.reader.fps} fps")

    def _episode(self) -> int:
        return max(0, self.episode_cb.currentIndex())

    def _on_play(self) -> None:
        if self.controller is None:
            return
        if self.controller.playing:
            self.controller.resume()
            return
        e = self._episode()
        self.slider.setMaximum(max(self.reader.episode_length(e) - 1, 0))
        self.controller.set_speed(self.speed.value())
        if self.send_cb.isChecked():
            QtWidgets.QMessageBox.information(self, "Replay", "Sending to robot. Ensure the wrapper is running and the area is clear.")
        self.controller.play_episode(e, send_to_robot=self.send_cb.isChecked())

    def _on_pause(self) -> None:
        if self.controller:
            self.controller.pause()

    def _on_stop(self) -> None:
        if self.controller:
            self.controller.stop()

    # ------------------------------------------------------------------ refresh
    def _refresh(self) -> None:
        if self.controller is None:
            return
        self.controller.set_speed(self.speed.value())
        e = self._episode()
        f = self.controller.frame
        n = self.reader.episode_length(e)
        if n:
            self.slider.setMaximum(max(n - 1, 0))
            self.slider.setValue(min(f, n - 1))
            img = self.reader.get_image(e, min(f, n - 1))
            if img is not None and img.ndim == 3:
                self.view.setPixmap(_np_to_pixmap(img).scaled(self.view.size(), QtCore.Qt.KeepAspectRatio))
            self.status.setText(f"ep {e}  frame {f}/{n}  {'▶ playing' if self.controller.playing else '⏸ stopped'}")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.controller:
            self.controller.shutdown()
        super().closeEvent(event)
