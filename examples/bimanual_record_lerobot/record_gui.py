#!/usr/bin/env python3
# Copyright 2025
# Record a LeRobot-format dataset from a custom portal-based robot with PyQt GUI.

from __future__ import annotations
from dataclasses import dataclass
import sys
import os
import shutil
from pathlib import Path
import threading
import time
import signal
import atexit

# PyQt imports
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QTextEdit, 
                             QProgressBar, QGroupBox, QGridLayout, QMessageBox,
                             QStatusBar, QFrame)
from PyQt6.QtCore import QTimer, QThread, pyqtSignal, Qt
from PyQt6.QtGui import QFont, QPalette, QColor

# ---- LeRobot imports ----
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.processor import make_default_processors
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.constants import OBS_STR
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

from i2rt import I2RTRobot, PortalLeaderTeleop
from config import I2RTFollowerConfig, i2rtLeaderConfig, RecordingConfig

def _clear_local_lerobot_cache(repo_id: str) -> None:
    # expands to ~/.cache/huggingface/lerobot/<namespace>/<name>
    cache_root = Path(os.path.expanduser("~")) / ".cache" / "huggingface" / "lerobot"
    cache_dir = cache_root / repo_id  # repo_id like "user/name"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

def create_events_dict() -> dict:
    """Create a simple events dictionary for recording control."""
    return {
        "stop_recording": False,
        "rerecord_episode": False,
        "exit_early": False,
        "pause": False,
        "resume": False,
        "start_requested": False
    }

class RecordingWorker(QThread):
    """Worker thread for recording operations."""
    status_update = pyqtSignal(str)
    episode_complete = pyqtSignal(int, int)  # current, total
    recording_started = pyqtSignal()
    recording_stopped = pyqtSignal()
    saving_started = pyqtSignal()
    saving_finished = pyqtSignal()
    session_finished = pyqtSignal(int)  # number of episodes recorded
    error_occurred = pyqtSignal(str)
    
    def __init__(self, events, recording_cfg, robot, teleop, dataset, processors):
        super().__init__()
        self.events = events
        self.recording_cfg = recording_cfg
        self.robot = robot
        self.teleop = teleop
        self.dataset = dataset
        self.teleop_action_processor, self.robot_action_processor, self.robot_observation_processor = processors
        
        self.recorded_episodes = 0
        self.current_episode = 0
        self.waiting_for_start = True
        
    def run(self):
        """Main recording loop."""
        try:
            self.status_update.emit("Starting recording session...")
            
            while (self.recording_cfg.num_episodes == 0 or self.recorded_episodes < self.recording_cfg.num_episodes) and not self.events["stop_recording"]:
                if self.waiting_for_start:
                    max_episodes_text = f" (max {self.recording_cfg.num_episodes})" if self.recording_cfg.num_episodes > 0 else ""
                    self.status_update.emit(f"Ready to record episode {self.current_episode + 1}{max_episodes_text}")
                    # Wait for start command
                    while self.waiting_for_start and not self.events["stop_recording"]:
                        time.sleep(0.1)
                        if self.events.get("start_requested", False):
                            self.events["start_requested"] = False
                            self.events["rerecord_episode"] = False
                            self.waiting_for_start = False
                            break
                
                if self.events["stop_recording"]:
                    break
                    
                self.status_update.emit(f"Recording episode {self.current_episode + 1}")
                self.recording_started.emit()
                
                # Reset flags before recording
                self.events["exit_early"] = False
                
                # Main record loop
                record_loop(
                    robot=self.robot,
                    events=self.events,
                    fps=self.recording_cfg.fps,
                    dataset=self.dataset,
                    teleop=self.teleop,
                    control_time_s=self.recording_cfg.episode_time_sec,
                    single_task=self.recording_cfg.task_description,
                    display_data=True,
                    teleop_action_processor=self.teleop_action_processor,
                    robot_action_processor=self.robot_action_processor,
                    robot_observation_processor=self.robot_observation_processor,
                )
                
                self.recording_stopped.emit()
                
                # Check if episode was exited early
                if self.events.get("exit_early", False):
                    self.status_update.emit("Episode recording ended early!")
                    self.events["exit_early"] = False
                
                # Handle rerecord
                if self.events["rerecord_episode"]:
                    self.status_update.emit("Re-record episode")
                    self.events["rerecord_episode"] = False
                    self.events["exit_early"] = False
                    self.dataset.clear_episode_buffer()
                    self.waiting_for_start = True
                    continue
                
                # Save episode
                self.saving_started.emit()
                self.status_update.emit(f"Saving episode {self.current_episode+1}...")
                self.dataset.save_episode()
                self.saving_finished.emit()
                self.status_update.emit(f"Episode {self.current_episode+1} saved successfully")
                self.recorded_episodes += 1
                self.current_episode += 1
                self.episode_complete.emit(self.recorded_episodes, self.recording_cfg.num_episodes)
                
                # Reset for next episode
                self.waiting_for_start = True
                
        except Exception as e:
            self.error_occurred.emit(f"Recording error: {str(e)}")
        finally:
            self.status_update.emit(f"Recording session ended - {self.recorded_episodes} episodes recorded")
            self.session_finished.emit(self.recorded_episodes)

class RecordingGUI(QMainWindow):
    """Main GUI window for recording control."""
    
    def __init__(self):
        super().__init__()
        self.events = create_events_dict()
        self.recording_worker = None
        self.recording_cfg = None
        self.robot = None
        self.teleop = None
        self.dataset = None
        self.processors = None
        self.is_recording = False
        self.is_connected = False
        
        self.init_ui()
        self.setup_connections()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("I2RT Recording Control")
        self.setGeometry(100, 100, 800, 600)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Status group
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        
        self.status_label = QLabel("Not connected")
        self.status_label.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
        status_layout.addWidget(self.status_label)
        
        self.recording_indicator = QLabel("● NOT RECORDING")
        self.recording_indicator.setStyleSheet("font-size: 16px; font-weight: bold; color: red; padding: 5px;")
        status_layout.addWidget(self.recording_indicator)
        
        self.saving_indicator = QLabel("")
        self.saving_indicator.setStyleSheet("font-size: 14px; font-weight: bold; color: orange; padding: 5px;")
        self.saving_indicator.setVisible(False)  # Hidden by default
        status_layout.addWidget(self.saving_indicator)
        
        # Saving progress bar (initially hidden)
        self.saving_progress = QProgressBar()
        self.saving_progress.setVisible(False)
        self.saving_progress.setRange(0, 0)  # Indeterminate progress
        self.saving_progress.setStyleSheet("QProgressBar { height: 8px; }")
        status_layout.addWidget(self.saving_progress)
        
        # Session results display
        self.session_results = QLabel("")
        self.session_results.setStyleSheet("font-size: 14px; font-weight: bold; color: blue; padding: 5px;")
        self.session_results.setVisible(False)
        status_layout.addWidget(self.session_results)
        
        # Progress group
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_label = QLabel("Episodes: 0/0")
        progress_layout.addWidget(self.progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        # Control buttons group
        control_group = QGroupBox("Recording Controls")
        control_layout = QGridLayout(control_group)
        
        # Connection buttons
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }")
        control_layout.addWidget(self.connect_btn, 0, 0)
        
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.setEnabled(False)
        self.disconnect_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 10px; }")
        control_layout.addWidget(self.disconnect_btn, 0, 1)
        
        # Recording buttons
        self.start_episode_btn = QPushButton("Start Episode")
        self.start_episode_btn.setEnabled(False)
        self.start_episode_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 10px; }")
        control_layout.addWidget(self.start_episode_btn, 1, 0)
        
        self.exit_early_btn = QPushButton("Finish Episode")
        self.exit_early_btn.setEnabled(False)
        self.exit_early_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; font-weight: bold; padding: 10px; }")
        control_layout.addWidget(self.exit_early_btn, 1, 1)
        
        self.rerecord_btn = QPushButton("Rerecord Episode")
        self.rerecord_btn.setEnabled(False)
        self.rerecord_btn.setStyleSheet("QPushButton { background-color: #9C27B0; color: white; font-weight: bold; padding: 10px; }")
        control_layout.addWidget(self.rerecord_btn, 2, 0)
        
        self.stop_recording_btn = QPushButton("Stop & Save All")
        self.stop_recording_btn.setEnabled(False)
        self.stop_recording_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 10px; }")
        control_layout.addWidget(self.stop_recording_btn, 2, 1)
        
        # Log group
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)
        
        # Add groups to main layout
        layout.addWidget(status_group)
        layout.addWidget(progress_group)
        layout.addWidget(control_group)
        layout.addWidget(log_group)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
    def setup_connections(self):
        """Connect signals and slots."""
        self.connect_btn.clicked.connect(self.connect_robot)
        self.disconnect_btn.clicked.connect(self.disconnect_robot)
        self.start_episode_btn.clicked.connect(self.start_episode)
        self.exit_early_btn.clicked.connect(self.exit_early)
        self.rerecord_btn.clicked.connect(self.rerecord_episode)
        self.stop_recording_btn.clicked.connect(self.stop_recording)
        
    def log_message(self, message):
        """Add message to log."""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        self.log_text.ensureCursorVisible()
        
    def connect_robot(self):
        """Connect to robot and teleop."""
        try:
            self.log_message("Connecting to robot and teleop...")
            self.status_label.setText("Connecting...")
            
            # Initialize configuration
            self.recording_cfg = RecordingConfig()
            _clear_local_lerobot_cache(self.recording_cfg.hf_repo_id)
            
            # Build robot (followers)
            robot_cfg = I2RTFollowerConfig()
            self.robot = I2RTRobot(robot_cfg)
            
            # Teleop (leader)
            teleop_cfg = i2rtLeaderConfig()
            self.teleop = PortalLeaderTeleop(teleop_cfg)
            
            # Connect endpoints
            self.robot.connect()
            self.teleop.connect()
            
            if not self.robot.is_connected or not self.teleop.is_connected:
                raise RuntimeError("Robot or teleop is not connected!")
            
            # Processors
            self.processors = make_default_processors()
            
            # Dataset features
            action_features = hw_to_dataset_features(self.robot.action_features, "action")
            obs_features = hw_to_dataset_features(self.robot.observation_features, OBS_STR)
            dataset_features = {**action_features, **obs_features}
            
            # Create dataset
            self.dataset = LeRobotDataset.create(
                repo_id=self.recording_cfg.hf_repo_id,
                fps=self.recording_cfg.fps,
                features=dataset_features,
                robot_type=self.robot.name,
                use_videos=self.recording_cfg.use_videos,
                image_writer_threads=4,
                batch_encoding_size=self.recording_cfg.batch_encoding_size,
            )
            
            # Update UI
            self.is_connected = True
            self.status_label.setText("Connected")
            self.status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: green; padding: 10px;")
            
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            self.start_episode_btn.setEnabled(True)
            self.stop_recording_btn.setEnabled(True)
            
            if self.recording_cfg.num_episodes > 0:
                self.progress_label.setText(f"Episodes: 0/{self.recording_cfg.num_episodes}")
                self.progress_bar.setMaximum(self.recording_cfg.num_episodes)
            else:
                self.progress_label.setText("Episodes: 0 (unlimited)")
                self.progress_bar.setMaximum(0)  # Indeterminate progress
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)
            
            # Reset session results
            self.session_results.setVisible(False)
            
            self.log_message("Successfully connected to robot and teleop!")
            if self.recording_cfg.num_episodes > 0:
                self.log_message(f"Recording a maximum of {self.recording_cfg.num_episodes} episodes of max {self.recording_cfg.episode_time_sec}s each")
            else:
                self.log_message(f"Recording unlimited episodes of max {self.recording_cfg.episode_time_sec}s each")
            
        except Exception as e:
            self.log_message(f"Connection error: {str(e)}")
            QMessageBox.critical(self, "Connection Error", f"Failed to connect: {str(e)}")
            
    def disconnect_robot(self):
        """Disconnect from robot and teleop."""
        try:
            self.log_message("Disconnecting...")
            
            if self.recording_worker and self.recording_worker.isRunning():
                self.stop_recording()
                
            if self.teleop:
                self.teleop.disconnect()
            if self.robot:
                self.robot.disconnect()
                
            self.is_connected = False
            self.status_label.setText("Disconnected")
            self.status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: red; padding: 10px;")
            
            self.connect_btn.setEnabled(True)
            self.disconnect_btn.setEnabled(False)
            self.start_episode_btn.setEnabled(False)
            self.exit_early_btn.setEnabled(False)
            self.rerecord_btn.setEnabled(False)
            self.stop_recording_btn.setEnabled(False)
            
            self.log_message("Disconnected successfully")
            
        except Exception as e:
            self.log_message(f"Disconnect error: {str(e)}")
            
    def start_episode(self):
        """Start recording an episode."""
        if not self.is_connected:
            return
            
        if not self.recording_worker or not self.recording_worker.isRunning():
            # Start new recording session
            self.recording_worker = RecordingWorker(
                self.events, self.recording_cfg, self.robot, self.teleop, 
                self.dataset, self.processors
            )
            self.recording_worker.status_update.connect(self.log_message)
            self.recording_worker.episode_complete.connect(self.update_progress)
            self.recording_worker.recording_started.connect(self.on_recording_started)
            self.recording_worker.recording_stopped.connect(self.on_recording_stopped)
            self.recording_worker.saving_started.connect(self.on_saving_started)
            self.recording_worker.saving_finished.connect(self.on_saving_finished)
            self.recording_worker.session_finished.connect(self.on_session_finished)
            self.recording_worker.error_occurred.connect(self.on_error)
            self.recording_worker.start()
            
        # Request start
        self.events["start_requested"] = True
        self.start_episode_btn.setEnabled(False)
        self.exit_early_btn.setEnabled(True)
        self.rerecord_btn.setEnabled(True)
        
    def exit_early(self):
        """Exit current episode early."""
        self.events["exit_early"] = True
        self.log_message("Exit early requested")
        
    def rerecord_episode(self):
        """Rerecord current episode."""
        self.events["rerecord_episode"] = True
        self.events["exit_early"] = True
        self.log_message("Rerecord episode requested")
        
    def stop_recording(self):
        """Stop the recording session and save all episodes."""
        self.events["stop_recording"] = True
        self.log_message("Stop recording requested - all episodes will be saved")
        
        if self.recording_worker and self.recording_worker.isRunning():
            self.recording_worker.wait(3000)  # Wait up to 3 seconds
            
        self.reset_ui()
        
    def on_recording_started(self):
        """Called when recording starts."""
        self.is_recording = True
        self.recording_indicator.setText("● RECORDING")
        self.recording_indicator.setStyleSheet("font-size: 16px; font-weight: bold; color: green; padding: 5px;")
        self.exit_early_btn.setEnabled(True)
        self.rerecord_btn.setEnabled(True)
        
    def on_recording_stopped(self):
        """Called when recording stops."""
        self.is_recording = False
        self.recording_indicator.setText("● NOT RECORDING")
        self.recording_indicator.setStyleSheet("font-size: 16px; font-weight: bold; color: red; padding: 5px;")
        self.exit_early_btn.setEnabled(False)
        self.rerecord_btn.setEnabled(False)
        self.start_episode_btn.setEnabled(True)
        
    def on_saving_started(self):
        """Called when episode saving starts."""
        self.saving_indicator.setText("💾 SAVING...")
        self.saving_indicator.setVisible(True)
        self.saving_progress.setVisible(True)
        # Disable buttons during saving
        self.start_episode_btn.setEnabled(False)
        self.exit_early_btn.setEnabled(False)
        self.rerecord_btn.setEnabled(False)
        
    def on_saving_finished(self):
        """Called when episode saving finishes."""
        self.saving_indicator.setVisible(False)
        self.saving_progress.setVisible(False)
        # Re-enable buttons after saving
        if not self.is_recording:
            self.start_episode_btn.setEnabled(self.is_connected)
        
    def update_progress(self, current, total):
        """Update progress display."""
        if total > 0:
            self.progress_label.setText(f"Episodes: {current}/{total}")
            self.progress_bar.setValue(current)
        else:
            self.progress_label.setText(f"Episodes: {current} (unlimited)")
            self.progress_bar.setValue(0)  # Keep at 0 for unlimited
            
    def on_session_finished(self, episodes_recorded):
        """Called when recording session finishes."""
        self.session_results.setText(f"✅ Session Complete: {episodes_recorded} episodes recorded")
        self.session_results.setVisible(True)
        self.log_message(f"🎉 Recording session completed! {episodes_recorded} episodes recorded successfully.")
        
    def on_error(self, error_msg):
        """Handle errors."""
        self.log_message(f"ERROR: {error_msg}")
        QMessageBox.critical(self, "Error", error_msg)
        self.reset_ui()
        
    def reset_ui(self):
        """Reset UI to initial state."""
        self.is_recording = False
        self.recording_indicator.setText("● NOT RECORDING")
        self.recording_indicator.setStyleSheet("font-size: 16px; font-weight: bold; color: red; padding: 5px;")
        self.saving_indicator.setVisible(False)
        self.saving_progress.setVisible(False)
        self.session_results.setVisible(False)
        self.start_episode_btn.setEnabled(self.is_connected)
        self.exit_early_btn.setEnabled(False)
        self.rerecord_btn.setEnabled(False)
        
    def closeEvent(self, event):
        """Handle window close event."""
        if self.recording_worker and self.recording_worker.isRunning():
            reply = QMessageBox.question(self, 'Quit', 'Recording is in progress. Are you sure you want to quit?',
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
            else:
                self.stop_recording()
        
        self.disconnect_robot()
        event.accept()

def main():
    """Main function."""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = RecordingGUI()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()