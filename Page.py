"""
Page.py - Main UI Manager for PyAV NeuroLab System

This script provides a main UI menu that allows users to:
- Start recording sessions with RecordingManager
- Replay sessions with ReplayManager
- Configure settings for recording and replaying
- Manage saved sessions (view, delete, export)
- Save and load custom configurations

Requirements:
    pip install PyQt5 pyqtgraph opencv-python numpy pandas matplotlib
"""

import sys
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
    QGroupBox, QMessageBox, QFileDialog, QListWidget, QDialog,
    QDialogButtonBox, QTextEdit, QComboBox, QCheckBox, QTabWidget,
    QScrollArea, QFormLayout, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon

from RecordingManager import RecordingManager
from ReplayManager import ReplayManagerQt
from DataManager import DataManager


# ============================================================================
# Recording Thread (for non-blocking recording)
# ============================================================================

class RecordingThread(QThread):
    """Thread for running recording session without blocking UI."""

    finished = pyqtSignal(bool, dict)
    progress = pyqtSignal(str)

    def __init__(self, recorder: RecordingManager, duration: int, event_name: str):
        super().__init__()
        self.recorder = recorder
        self.duration = duration
        self.event_name = event_name

    def run(self):
        """Run recording in background thread."""
        try:
            self.progress.emit("Recording started...")
            success = self.recorder.start_recording(
                duration=self.duration,
                event_name=self.event_name
            )

            if success:
                self.progress.emit("Saving data...")
                results = self.recorder.stop_recording(save=True)
                self.finished.emit(True, results)
            else:
                self.finished.emit(False, {})

        except Exception as e:
            logging.error(f"Recording thread error: {e}")
            self.finished.emit(False, {"error": str(e)})


# ============================================================================
# Settings Dialog
# ============================================================================

class SettingsDialog(QDialog):
    """Dialog for configuring recording and replay settings."""

    def __init__(self, current_settings: Dict, parent=None):
        super().__init__(parent)
        self.settings = current_settings.copy()
        self.init_ui()

    def init_ui(self):
        """Initialize settings dialog UI."""
        self.setWindowTitle("Settings")
        self.setMinimumSize(600, 500)

        layout = QVBoxLayout()

        # Create tabs for different settings categories
        tabs = QTabWidget()

        # Recording settings tab
        recording_tab = self._create_recording_settings_tab()
        tabs.addTab(recording_tab, "Recording")

        # Replay settings tab
        replay_tab = self._create_replay_settings_tab()
        tabs.addTab(replay_tab, "Replay")

        # General settings tab
        general_tab = self._create_general_settings_tab()
        tabs.addTab(general_tab, "General")

        layout.addWidget(tabs)

        # Buttons
        button_layout = QHBoxLayout()

        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self.save_settings)

        load_btn = QPushButton("Load Settings")
        load_btn.clicked.connect(self.load_settings)

        button_layout.addWidget(save_btn)
        button_layout.addWidget(load_btn)
        button_layout.addStretch()

        layout.addLayout(button_layout)

        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def _create_recording_settings_tab(self) -> QWidget:
        """Create recording settings tab."""
        widget = QWidget()
        layout = QFormLayout()

        # Data folder
        self.folder_input = QLineEdit(self.settings.get("folder_path", "./recordings"))
        folder_layout = QHBoxLayout()
        folder_layout.addWidget(self.folder_input)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_folder)
        folder_layout.addWidget(browse_btn)

        layout.addRow("Data Folder:", folder_layout)

        # File prefix name
        self.prefix_name_input = QLineEdit(self.settings.get("file_prefix_name", "NeuroLab"))
        layout.addRow("File Prefix:", self.prefix_name_input)

        # Event name
        self.event_name_input = QLineEdit(self.settings.get("file_prefix_event", "Experiment"))
        layout.addRow("Default Event Name:", self.event_name_input)

        # Preview option
        self.preview_check = QCheckBox("Enable Camera Preview")
        self.preview_check.setChecked(self.settings.get("preview", True))
        layout.addRow("Camera Preview:", self.preview_check)

        # Default duration
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(5, 3600)
        self.duration_spin.setValue(self.settings.get("default_duration", 60))
        self.duration_spin.setSuffix(" seconds")
        layout.addRow("Default Duration:", self.duration_spin)

        widget.setLayout(layout)
        return widget

    def _create_replay_settings_tab(self) -> QWidget:
        """Create replay settings tab."""
        widget = QWidget()
        layout = QFormLayout()

        # Default playback speed
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "1.0x", "2.0x", "4.0x"])
        default_speed = self.settings.get("default_playback_speed", "1.0x")
        self.speed_combo.setCurrentText(default_speed)
        layout.addRow("Default Playback Speed:", self.speed_combo)

        # EEG window size
        self.eeg_window_spin = QDoubleSpinBox()
        self.eeg_window_spin.setRange(0.5, 10.0)
        self.eeg_window_spin.setValue(self.settings.get("eeg_window_size", 2.0))
        self.eeg_window_spin.setSuffix(" seconds")
        self.eeg_window_spin.setSingleStep(0.5)
        layout.addRow("EEG Window Size:", self.eeg_window_spin)

        # Number of EEG channels
        self.eeg_channels_spin = QSpinBox()
        self.eeg_channels_spin.setRange(1, 32)
        self.eeg_channels_spin.setValue(self.settings.get("eeg_channels", 16))
        layout.addRow("EEG Channels:", self.eeg_channels_spin)

        widget.setLayout(layout)
        return widget

    def _create_general_settings_tab(self) -> QWidget:
        """Create general settings tab."""
        widget = QWidget()
        layout = QFormLayout()

        # Logging level
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        log_level = self.settings.get("log_level", "INFO")
        self.log_level_combo.setCurrentText(log_level)
        layout.addRow("Logging Level:", self.log_level_combo)

        # Auto-save settings
        self.autosave_check = QCheckBox("Auto-save settings on exit")
        self.autosave_check.setChecked(self.settings.get("autosave_settings", True))
        layout.addRow("Auto-save:", self.autosave_check)

        widget.setLayout(layout)
        return widget

    def browse_folder(self):
        """Browse for data folder."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Data Folder",
            self.folder_input.text()
        )
        if folder:
            self.folder_input.setText(folder)

    def get_settings(self) -> Dict:
        """Get current settings from dialog."""
        return {
            "folder_path": self.folder_input.text(),
            "file_prefix_name": self.prefix_name_input.text(),
            "file_prefix_event": self.event_name_input.text(),
            "preview": self.preview_check.isChecked(),
            "default_duration": self.duration_spin.value(),
            "default_playback_speed": self.speed_combo.currentText(),
            "eeg_window_size": self.eeg_window_spin.value(),
            "eeg_channels": self.eeg_channels_spin.value(),
            "log_level": self.log_level_combo.currentText(),
            "autosave_settings": self.autosave_check.isChecked()
        }

    def save_settings(self):
        """Save settings to file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Settings",
            "config.json",
            "JSON Files (*.json)"
        )

        if file_path:
            try:
                settings = self.get_settings()
                with open(file_path, 'w') as f:
                    json.dump(settings, f, indent=2)
                QMessageBox.information(self, "Success", "Settings saved successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save settings:\n{str(e)}")

    def load_settings(self):
        """Load settings from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Settings",
            "",
            "JSON Files (*.json)"
        )

        if file_path:
            try:
                with open(file_path, 'r') as f:
                    settings = json.load(f)
                self.settings = settings

                # Update UI with loaded settings
                self.folder_input.setText(settings.get("folder_path", "./recordings"))
                self.prefix_name_input.setText(settings.get("file_prefix_name", "NeuroLab"))
                self.event_name_input.setText(settings.get("file_prefix_event", "Experiment"))
                self.preview_check.setChecked(settings.get("preview", True))
                self.duration_spin.setValue(settings.get("default_duration", 60))
                self.speed_combo.setCurrentText(settings.get("default_playback_speed", "1.0x"))
                self.eeg_window_spin.setValue(settings.get("eeg_window_size", 2.0))
                self.eeg_channels_spin.setValue(settings.get("eeg_channels", 16))
                self.log_level_combo.setCurrentText(settings.get("log_level", "INFO"))
                self.autosave_check.setChecked(settings.get("autosave_settings", True))

                QMessageBox.information(self, "Success", "Settings loaded successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load settings:\n{str(e)}")


# ============================================================================
# Session Manager Dialog
# ============================================================================

class SessionManagerDialog(QDialog):
    """Dialog for managing saved sessions."""

    def __init__(self, data_manager: DataManager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.init_ui()
        self.load_sessions()

    def init_ui(self):
        """Initialize session manager UI."""
        self.setWindowTitle("Session Manager")
        self.setMinimumSize(800, 600)

        layout = QVBoxLayout()

        # Info label
        self.info_label = QLabel("Available Sessions")
        self.info_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(self.info_label)

        # Session table
        self.session_table = QTableWidget()
        self.session_table.setColumnCount(5)
        self.session_table.setHorizontalHeaderLabels([
            "Session Name", "Date", "Video", "EEG", "Landmarks"
        ])
        self.session_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        layout.addWidget(self.session_table)

        # Buttons
        button_layout = QHBoxLayout()

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.load_sessions)
        button_layout.addWidget(refresh_btn)

        delete_btn = QPushButton("Delete Selected")
        delete_btn.clicked.connect(self.delete_session)
        button_layout.addWidget(delete_btn)

        button_layout.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def load_sessions(self):
        """Load and display all sessions."""
        try:
            sessions = self.data_manager.list_sessions()
            self.session_table.setRowCount(len(sessions))

            for i, session_name in enumerate(sessions):
                # Session name
                self.session_table.setItem(i, 0, QTableWidgetItem(session_name))

                # Extract date from session name (assuming format includes date)
                try:
                    date_str = session_name.split('_')[1] + "_" + session_name.split('_')[2]
                    date = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
                    date_formatted = date.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    date_formatted = "Unknown"

                self.session_table.setItem(i, 1, QTableWidgetItem(date_formatted))

                # Check data availability
                folder_path = Path(self.data_manager.folder_path)
                has_video = (folder_path / f"{session_name}_video.mp4").exists()
                has_eeg = (folder_path / f"{session_name}_eeg.csv").exists()
                has_landmarks = (folder_path / f"{session_name}_landmarks.json").exists()

                self.session_table.setItem(i, 2, QTableWidgetItem("✓" if has_video else "✗"))
                self.session_table.setItem(i, 3, QTableWidgetItem("✓" if has_eeg else "✗"))
                self.session_table.setItem(i, 4, QTableWidgetItem("✓" if has_landmarks else "✗"))

            self.info_label.setText(f"Available Sessions ({len(sessions)} total)")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load sessions:\n{str(e)}")

    def delete_session(self):
        """Delete selected session."""
        current_row = self.session_table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "No Selection", "Please select a session to delete.")
            return

        session_name = self.session_table.item(current_row, 0).text()

        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete session:\n{session_name}?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                # Delete all files associated with this session
                folder_path = Path(self.data_manager.folder_path)
                files_to_delete = [
                    folder_path / f"{session_name}_video.mp4",
                    folder_path / f"{session_name}_video_timestamps.json",
                    folder_path / f"{session_name}_landmarks.json",
                    folder_path / f"{session_name}_eeg.csv",
                    folder_path / f"{session_name}_eeg.json",
                    folder_path / f"{session_name}_metadata.json"
                ]

                deleted_count = 0
                for file_path in files_to_delete:
                    if file_path.exists():
                        file_path.unlink()
                        deleted_count += 1

                QMessageBox.information(
                    self,
                    "Success",
                    f"Deleted {deleted_count} files for session: {session_name}"
                )

                # Refresh table
                self.load_sessions()

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete session:\n{str(e)}")


# ============================================================================
# Main Page UI
# ============================================================================

class PageUI(QMainWindow):
    """
    Main UI Manager for PyAV NeuroLab System.

    Provides a main menu for:
    - Starting recording sessions
    - Replaying sessions
    - Managing sessions
    - Configuring settings
    """

    def __init__(self):
        super().__init__()

        # Load settings
        self.settings = self.load_default_settings()

        # Initialize managers
        self.data_manager = DataManager(self.settings)
        self.recorder = None
        self.recording_thread = None
        self.replay_window = None

        # Setup logging
        log_level = getattr(logging, self.settings.get("log_level", "INFO"))
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Initialize UI
        self.init_ui()

        self.logger.info("PageUI initialized")

    def init_ui(self):
        """Initialize main UI."""
        self.setWindowTitle("PyAV NeuroLab - Main Menu")
        self.setGeometry(100, 100, 900, 700)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Title
        title = QLabel("PyAV NeuroLab System")
        title.setFont(QFont("Arial", 24, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        subtitle = QLabel("Recording and Replay Manager")
        subtitle.setFont(QFont("Arial", 12))
        subtitle.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(subtitle)

        main_layout.addSpacing(30)

        # Create scroll area for main content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        # Recording section
        recording_group = self._create_recording_section()
        scroll_layout.addWidget(recording_group)

        # Replay section
        replay_group = self._create_replay_section()
        scroll_layout.addWidget(replay_group)

        # Management section
        management_group = self._create_management_section()
        scroll_layout.addWidget(management_group)

        scroll_layout.addStretch()

        scroll.setWidget(scroll_widget)
        main_layout.addWidget(scroll)

        # Status bar
        self.statusBar().showMessage("Ready")

        # Menu bar
        self.create_menu_bar()

    def create_menu_bar(self):
        """Create application menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('&File')

        settings_action = file_menu.addAction('&Settings')
        settings_action.triggered.connect(self.open_settings)
        settings_action.setShortcut('Ctrl+,')

        file_menu.addSeparator()

        exit_action = file_menu.addAction('E&xit')
        exit_action.triggered.connect(self.close)
        exit_action.setShortcut('Ctrl+Q')

        # Help menu
        help_menu = menubar.addMenu('&Help')

        about_action = help_menu.addAction('&About')
        about_action.triggered.connect(self.show_about)

    def _create_recording_section(self) -> QGroupBox:
        """Create recording control section."""
        group = QGroupBox("Recording")
        group.setFont(QFont("Arial", 14, QFont.Bold))
        layout = QVBoxLayout()

        # Description
        desc = QLabel("Start a new recording session with EEG device and camera.")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Recording parameters
        params_layout = QFormLayout()

        self.event_name_input = QLineEdit(self.settings.get("file_prefix_event", "Experiment"))
        params_layout.addRow("Event Name:", self.event_name_input)

        self.duration_input = QSpinBox()
        self.duration_input.setRange(5, 3600)
        self.duration_input.setValue(self.settings.get("default_duration", 60))
        self.duration_input.setSuffix(" seconds")
        params_layout.addRow("Duration:", self.duration_input)

        layout.addLayout(params_layout)

        # Buttons
        button_layout = QHBoxLayout()

        self.start_recording_btn = QPushButton("▶ Start Recording")
        self.start_recording_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-size: 14px; padding: 10px; }")
        self.start_recording_btn.clicked.connect(self.start_recording)
        button_layout.addWidget(self.start_recording_btn)

        self.check_devices_btn = QPushButton("🔍 Check Devices")
        self.check_devices_btn.clicked.connect(self.check_devices)
        button_layout.addWidget(self.check_devices_btn)

        layout.addLayout(button_layout)

        # Status label
        self.recording_status_label = QLabel("Status: Ready")
        self.recording_status_label.setStyleSheet("QLabel { color: blue; font-weight: bold; }")
        layout.addWidget(self.recording_status_label)

        group.setLayout(layout)
        return group

    def _create_replay_section(self) -> QGroupBox:
        """Create replay control section."""
        group = QGroupBox("Replay")
        group.setFont(QFont("Arial", 14, QFont.Bold))
        layout = QVBoxLayout()

        # Description
        desc = QLabel("Open replay manager to view and analyze recorded sessions.")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Button
        self.open_replay_btn = QPushButton("📺 Open Replay Manager")
        self.open_replay_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-size: 14px; padding: 10px; }")
        self.open_replay_btn.clicked.connect(self.open_replay_manager)
        layout.addWidget(self.open_replay_btn)

        group.setLayout(layout)
        return group

    def _create_management_section(self) -> QGroupBox:
        """Create session management section."""
        group = QGroupBox("Session Management")
        group.setFont(QFont("Arial", 14, QFont.Bold))
        layout = QVBoxLayout()

        # Description
        desc = QLabel("View, manage, and delete recorded sessions.")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Buttons
        button_layout = QHBoxLayout()

        self.manage_sessions_btn = QPushButton("📁 Manage Sessions")
        self.manage_sessions_btn.clicked.connect(self.open_session_manager)
        button_layout.addWidget(self.manage_sessions_btn)

        self.view_stats_btn = QPushButton("📊 View Statistics")
        self.view_stats_btn.clicked.connect(self.show_statistics)
        button_layout.addWidget(self.view_stats_btn)

        layout.addLayout(button_layout)

        group.setLayout(layout)
        return group

    # ========================================================================
    # Settings Methods
    # ========================================================================

    def load_default_settings(self) -> Dict:
        """Load default settings or from config file."""
        config_file = Path("config.json")

        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    settings = json.load(f)
                self.logger.info("Loaded settings from config.json")
                return settings
            except Exception as e:
                self.logger.warning(f"Failed to load config.json: {e}")

        # Default settings
        return {
            "folder_path": "./recordings",
            "file_prefix_name": "NeuroLab",
            "file_prefix_event": "Experiment",
            "preview": True,
            "default_duration": 60,
            "default_playback_speed": "1.0x",
            "eeg_window_size": 2.0,
            "eeg_channels": 16,
            "log_level": "INFO",
            "autosave_settings": True
        }

    def save_settings_to_file(self):
        """Save current settings to config file."""
        try:
            with open("config.json", 'w') as f:
                json.dump(self.settings, f, indent=2)
            self.logger.info("Settings saved to config.json")
        except Exception as e:
            self.logger.error(f"Failed to save settings: {e}")

    def open_settings(self):
        """Open settings dialog."""
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec_() == QDialog.Accepted:
            self.settings = dialog.get_settings()

            # Update managers with new settings
            self.data_manager = DataManager(self.settings)

            # Update log level
            log_level = getattr(logging, self.settings.get("log_level", "INFO"))
            logging.getLogger().setLevel(log_level)

            QMessageBox.information(self, "Success", "Settings updated successfully!")
            self.logger.info("Settings updated")

    # ========================================================================
    # Recording Methods
    # ========================================================================

    def check_devices(self):
        """Check device availability."""
        self.statusBar().showMessage("Checking devices...")
        self.recording_status_label.setText("Status: Checking devices...")

        try:
            # Initialize recorder if not exists
            if self.recorder is None:
                self.recorder = RecordingManager(self.settings)

            device_status = self.recorder.check_devices()

            # Display results
            status_text = "Device Check Results:\n\n"
            status_text += f"EEG Device: {'✓ Available' if device_status['eeg'] else '✗ Not Available'}\n"
            status_text += f"Camera: {'✓ Available' if device_status['camera'] else '✗ Not Available'}\n"

            if device_status['eeg']:
                status_text += "\n✓ You can start recording!"
                self.recording_status_label.setText("Status: Devices ready")
                self.recording_status_label.setStyleSheet("QLabel { color: green; font-weight: bold; }")
            else:
                status_text += "\n✗ EEG device required for recording"
                self.recording_status_label.setText("Status: EEG device not available")
                self.recording_status_label.setStyleSheet("QLabel { color: red; font-weight: bold; }")

            QMessageBox.information(self, "Device Status", status_text)
            self.statusBar().showMessage("Device check completed")

        except Exception as e:
            error_msg = f"Device check failed:\n{str(e)}"
            QMessageBox.critical(self, "Error", error_msg)
            self.recording_status_label.setText("Status: Device check failed")
            self.recording_status_label.setStyleSheet("QLabel { color: red; font-weight: bold; }")
            self.statusBar().showMessage("Device check failed")

    def start_recording(self):
        """Start recording session."""
        event_name = self.event_name_input.text().strip()
        duration = self.duration_input.value()

        if not event_name:
            QMessageBox.warning(self, "Invalid Input", "Please enter an event name.")
            return

        # Confirm start
        reply = QMessageBox.question(
            self,
            "Start Recording",
            f"Start recording session:\n\nEvent: {event_name}\nDuration: {duration} seconds",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.No:
            return

        try:
            # Initialize recorder
            if self.recorder is None:
                self.recorder = RecordingManager(self.settings)

            # Disable start button
            self.start_recording_btn.setEnabled(False)
            self.recording_status_label.setText("Status: Recording in progress...")
            self.recording_status_label.setStyleSheet("QLabel { color: orange; font-weight: bold; }")
            self.statusBar().showMessage(f"Recording for {duration} seconds...")

            # Start recording in thread
            self.recording_thread = RecordingThread(self.recorder, duration, event_name)
            self.recording_thread.finished.connect(self.on_recording_finished)
            self.recording_thread.progress.connect(self.on_recording_progress)
            self.recording_thread.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start recording:\n{str(e)}")
            self.start_recording_btn.setEnabled(True)
            self.recording_status_label.setText("Status: Recording failed")
            self.recording_status_label.setStyleSheet("QLabel { color: red; font-weight: bold; }")

    def on_recording_progress(self, message: str):
        """Handle recording progress updates."""
        self.statusBar().showMessage(message)
        self.logger.info(message)

    def on_recording_finished(self, success: bool, results: Dict):
        """Handle recording completion."""
        self.start_recording_btn.setEnabled(True)

        if success:
            result_text = "Recording completed successfully!\n\nSaved files:\n"
            for key, value in results.items():
                result_text += f"  {key}: {'✓' if value else '✗'}\n"

            QMessageBox.information(self, "Success", result_text)
            self.recording_status_label.setText("Status: Recording completed")
            self.recording_status_label.setStyleSheet("QLabel { color: green; font-weight: bold; }")
            self.statusBar().showMessage("Recording completed successfully")
        else:
            error_msg = results.get("error", "Unknown error")
            QMessageBox.critical(self, "Error", f"Recording failed:\n{error_msg}")
            self.recording_status_label.setText("Status: Recording failed")
            self.recording_status_label.setStyleSheet("QLabel { color: red; font-weight: bold; }")
            self.statusBar().showMessage("Recording failed")

    # ========================================================================
    # Replay Methods
    # ========================================================================

    def open_replay_manager(self):
        """Open replay manager window."""
        try:
            if self.replay_window is None or not self.replay_window.isVisible():
                data_folder = self.settings.get("folder_path", "./recordings")
                self.replay_window = ReplayManagerQt(data_folder=data_folder)
                self.replay_window.show()
                self.logger.info("Replay manager opened")
            else:
                self.replay_window.raise_()
                self.replay_window.activateWindow()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open replay manager:\n{str(e)}")

    # ========================================================================
    # Management Methods
    # ========================================================================

    def open_session_manager(self):
        """Open session manager dialog."""
        try:
            dialog = SessionManagerDialog(self.data_manager, self)
            dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open session manager:\n{str(e)}")

    def show_statistics(self):
        """Show statistics for all sessions."""
        try:
            sessions = self.data_manager.list_sessions()

            if not sessions:
                QMessageBox.information(self, "Statistics", "No sessions found.")
                return

            # Calculate statistics
            total_sessions = len(sessions)
            folder_path = Path(self.data_manager.folder_path)

            video_count = sum(1 for s in sessions if (folder_path / f"{s}_video.mp4").exists())
            eeg_count = sum(1 for s in sessions if (folder_path / f"{s}_eeg.csv").exists())
            landmarks_count = sum(1 for s in sessions if (folder_path / f"{s}_landmarks.json").exists())

            # Calculate total size
            total_size = 0
            for session in sessions:
                for suffix in ["_video.mp4", "_eeg.csv", "_eeg.json", "_landmarks.json", "_metadata.json", "_video_timestamps.json"]:
                    file_path = folder_path / f"{session}{suffix}"
                    if file_path.exists():
                        total_size += file_path.stat().st_size

            size_mb = total_size / (1024 * 1024)

            stats_text = f"Session Statistics:\n\n"
            stats_text += f"Total Sessions: {total_sessions}\n"
            stats_text += f"Sessions with Video: {video_count}\n"
            stats_text += f"Sessions with EEG: {eeg_count}\n"
            stats_text += f"Sessions with Landmarks: {landmarks_count}\n"
            stats_text += f"\nTotal Storage: {size_mb:.2f} MB\n"
            stats_text += f"Data Folder: {folder_path}\n"

            QMessageBox.information(self, "Statistics", stats_text)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to calculate statistics:\n{str(e)}")

    # ========================================================================
    # Help Methods
    # ========================================================================

    def show_about(self):
        """Show about dialog."""
        about_text = """
        <h2>PyAV NeuroLab System</h2>
        <p><b>Version:</b> 1.0</p>

        <p>A comprehensive recording and replay system for multi-modal neuroscience data.</p>

        <h3>Features:</h3>
        <ul>
            <li>EEG data recording and visualization</li>
            <li>Video recording with pose detection</li>
            <li>Synchronized multi-modal replay</li>
            <li>Session management and analysis</li>
        </ul>

        <p><b>Components:</b></p>
        <ul>
            <li>RecordingManager - Data acquisition</li>
            <li>ReplayManager - Data visualization</li>
            <li>DataManager - Data storage and loading</li>
        </ul>
        """
        QMessageBox.about(self, "About PyAV NeuroLab", about_text)

    # ========================================================================
    # Window Close Event
    # ========================================================================

    def closeEvent(self, event):
        """Handle window close event."""
        # Auto-save settings if enabled
        if self.settings.get("autosave_settings", True):
            self.save_settings_to_file()

        # Close replay window if open
        if self.replay_window:
            self.replay_window.close()

        event.accept()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Set application info
    app.setApplicationName("PyAV NeuroLab")
    app.setApplicationVersion("1.0")

    # Create and show main window
    window = PageUI()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
