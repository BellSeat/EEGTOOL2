"""
ReplayManagerQt - PyQt5-based Replay Manager with Timestamp Synchronization

Complete GUI application for replaying multi-modal neuroscience data with:
- Video playback with landmark overlay
- Real-time EEG visualization (PyQtGraph)
- Timestamp-based synchronization
- Advanced session search and filtering
- Multiple playback modes (video/EEG/landmarks)

Requirements:
    pip install PyQt5 pyqtgraph opencv-python numpy pandas matplotlib
"""

import sys
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QComboBox, QSplitter, QFileDialog,
    QLineEdit, QGroupBox, QMessageBox, QListWidget, QDialog, QDialogButtonBox
)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont

from DataManager import DataManager


# ============================================================================
# Data Classes
# ============================================================================

class ReplaySession:
    """Container for replay session data."""
    
    def __init__(self, session_name: str):
        self.session_name = session_name
        self.video_cap: Optional[cv2.VideoCapture] = None
        self.eeg_data: Optional[pd.DataFrame] = None
        self.landmarks_data: Optional[List[Dict]] = None
        self.metadata: Optional[Dict] = None
    
    def is_valid(self) -> Dict[str, bool]:
        """
        Check validity of each data type.
        
        Returns:
            Dict[str, bool]: Validity status for video, EEG, and landmarks
        """
        return {
            "video": self.video_cap is not None and self.video_cap.isOpened(),
            "eeg": self.eeg_data is not None and len(self.eeg_data) > 0,
            "landmarks": self.landmarks_data is not None and len(self.landmarks_data) > 0
        }
    
    def has_any_data(self) -> bool:
        """Check if session has at least one data type."""
        validity = self.is_valid()
        return any(validity.values())
    
    def get_available_data(self) -> List[str]:
        """Get list of available data types."""
        validity = self.is_valid()
        return [data_type for data_type, is_valid in validity.items() if is_valid]


# ============================================================================
# EEG Plot Widget (PyQtGraph)
# ============================================================================

class EEGPlotWidget(pg.GraphicsLayoutWidget):
    """Real-time EEG visualization widget using PyQtGraph."""
    
    def __init__(self, num_channels: int = 16):
        super().__init__()
        
        self.num_channels = num_channels
        self.plots = []
        self.curves = []
        
        # Setup plots for each channel
        for i in range(num_channels):
            plot = self.addPlot(row=i, col=0)
            plot.setLabel('left', f'Ch{i+1}')
            plot.showGrid(x=True, y=True, alpha=0.3)
            plot.setMenuEnabled(False)
            
            # Create curve for this channel
            curve = plot.plot(pen=pg.mkPen(color=(0, 150, 255), width=1))
            
            self.plots.append(plot)
            self.curves.append(curve)
        
        # Configure layout
        self.ci.layout.setRowStretchFactor(num_channels-1, 1)
    
    def update_data(self, eeg_window: pd.DataFrame):
        """
        Update EEG display with new data.
        
        Args:
            eeg_window: DataFrame with timestamp and channel columns
        """
        if eeg_window is None or len(eeg_window) == 0:
            return
        
        # Get channel columns (exclude timestamp)
        eeg_columns = [col for col in eeg_window.columns if col != 'Timestamp']
        
        x = np.arange(len(eeg_window))
        
        for i in range(min(self.num_channels, len(eeg_columns))):
            y = eeg_window[eeg_columns[i]].values
            self.curves[i].setData(x, y)
            
            # Auto-scale y-axis
            if len(y) > 0:
                y_min, y_max = y.min(), y.max()
                margin = max((y_max - y_min) * 0.1, 10)
                self.plots[i].setYRange(y_min - margin, y_max + margin)


# ============================================================================
# Video Display Widget
# ============================================================================

class VideoWidget(QLabel):
    """Video display widget with landmark overlay support."""
    
    def __init__(self):
        super().__init__()
        self.setScaledContents(True)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("background-color: black;")
    
    def display_frame(self, frame: np.ndarray, landmarks: Optional[Dict] = None):
        """
        Display video frame with optional landmark overlay.
        
        Args:
            frame: Video frame in BGR format
            landmarks: Optional landmark data to overlay
        """
        if frame is None:
            return
        
        # Draw landmarks if available
        if landmarks:
            frame = self._draw_landmarks(frame.copy(), landmarks)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to QImage
        height, width, channel = rgb_frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Display
        pixmap = QPixmap.fromImage(q_image)
        self.setPixmap(pixmap)
    
    def _draw_landmarks(self, frame: np.ndarray, landmarks: Dict) -> np.ndarray:
        """Draw pose and hand landmarks on frame."""
        height, width = frame.shape[:2]
        
        # Draw pose landmarks
        pose_lms = landmarks.get('pose_landmarks', [])
        for lm in pose_lms:
            x = int(lm['x'] * width)
            y = int(lm['y'] * height)
            visibility = lm.get('visibility', 1.0)
            
            # Color based on visibility
            color = (0, 255, 0) if visibility > 0.5 else (0, 255, 255)
            cv2.circle(frame, (x, y), 3, color, -1)
        
        # Draw hand landmarks
        hands_lms = landmarks.get('hands_landmarks', [])
        for hand_idx, hand in enumerate(hands_lms):
            color = (255, 0, 0) if hand_idx == 0 else (0, 0, 255)
            for lm in hand:
                x = int(lm['x'] * width)
                y = int(lm['y'] * height)
                cv2.circle(frame, (x, y), 3, color, -1)
        
        return frame


# ============================================================================
# Control Panel
# ============================================================================

class ControlPanel(QWidget):
    """Playback control panel with buttons and sliders."""
    
    # Signals
    play_clicked = pyqtSignal()
    pause_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    seek_backward_clicked = pyqtSignal()
    seek_forward_clicked = pyqtSignal()
    slider_moved = pyqtSignal(int)
    speed_changed = pyqtSignal(float)
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout()
        
        # Progress slider
        slider_layout = QVBoxLayout()
        slider_layout.addWidget(QLabel("Progress:"))
        
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setMinimum(0)
        self.progress_slider.setMaximum(100)
        self.progress_slider.setValue(0)
        self.progress_slider.sliderMoved.connect(self.slider_moved.emit)
        slider_layout.addWidget(self.progress_slider)
        
        layout.addLayout(slider_layout)
        
        # Time label
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setFont(QFont("Arial", 12))
        layout.addWidget(self.time_label)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.backward_btn = QPushButton("⏪ -30s")
        self.play_btn = QPushButton("▶ Play")
        self.pause_btn = QPushButton("⏸ Pause")
        self.stop_btn = QPushButton("⏹ Stop")
        self.forward_btn = QPushButton("⏩ +30s")
        
        # Connect signals
        self.backward_btn.clicked.connect(self.seek_backward_clicked.emit)
        self.play_btn.clicked.connect(self.play_clicked.emit)
        self.pause_btn.clicked.connect(self.pause_clicked.emit)
        self.stop_btn.clicked.connect(self.stop_clicked.emit)
        self.forward_btn.clicked.connect(self.seek_forward_clicked.emit)
        
        # Add buttons to layout
        button_layout.addWidget(self.backward_btn)
        button_layout.addWidget(self.play_btn)
        button_layout.addWidget(self.pause_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addWidget(self.forward_btn)
        
        layout.addLayout(button_layout)
        
        # Speed control
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "1.0x", "2.0x", "4.0x"])
        self.speed_combo.setCurrentText("1.0x")
        self.speed_combo.currentTextChanged.connect(self._on_speed_changed)
        
        speed_layout.addWidget(self.speed_combo)
        speed_layout.addStretch()
        
        layout.addLayout(speed_layout)
        
        self.setLayout(layout)
    
    def _on_speed_changed(self, text: str):
        """Handle speed selection change."""
        speed_map = {
            "0.25x": 0.25,
            "0.5x": 0.5,
            "1.0x": 1.0,
            "2.0x": 2.0,
            "4.0x": 4.0
        }
        speed = speed_map.get(text, 1.0)
        self.speed_changed.emit(speed)
    
    def update_time(self, current_time: float, total_time: float):
        """Update time display."""
        current_str = self._format_time(current_time)
        total_str = self._format_time(total_time)
        self.time_label.setText(f"{current_str} / {total_str}")
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format time as MM:SS."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"


# ============================================================================
# Session Selection Dialog
# ============================================================================

class SessionSelectionDialog(QDialog):
    """Dialog for selecting sessions with search and filter options."""
    
    def __init__(self, data_manager: DataManager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.selected_session = None
        self.all_sessions = []  # Store all sessions
        self.init_ui()
        
        # Load sessions after UI is initialized
        self.load_sessions()
    
    def init_ui(self):
        """Initialize UI components."""
        self.setWindowTitle("Select Session")
        self.setMinimumSize(600, 500)
        
        layout = QVBoxLayout()
        
        # Info label
        self.info_label = QLabel("Loading sessions...")
        layout.addWidget(self.info_label)
        
        # Search box
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search:"))
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter keyword to search...")
        self.search_input.textChanged.connect(self.update_session_list)
        search_layout.addWidget(self.search_input)
        
        layout.addLayout(search_layout)
        
        # Filter options
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter by:"))
        
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All Sessions", "Has Video", "Has EEG", "Has Landmarks"])
        self.filter_combo.currentTextChanged.connect(self.update_session_list)
        filter_layout.addWidget(self.filter_combo)
        filter_layout.addStretch()
        
        layout.addLayout(filter_layout)
        
        # Session list
        self.session_list = QListWidget()
        self.session_list.itemDoubleClicked.connect(self.accept)
        layout.addWidget(self.session_list)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
    
    def load_sessions(self):
        """Load all sessions from DataManager."""
        try:
            # Get all sessions
            self.all_sessions = self.data_manager.list_sessions()
            
            if not self.all_sessions:
                self.info_label.setText("No sessions found in data folder")
                logging.warning("No sessions found")
            else:
                self.info_label.setText(f"Found {len(self.all_sessions)} sessions")
                logging.info(f"Loaded {len(self.all_sessions)} sessions")
            
            # Update the list display
            self.update_session_list()
            
        except Exception as e:
            error_msg = f"Error loading sessions: {str(e)}"
            self.info_label.setText(error_msg)
            logging.error(error_msg)
            QMessageBox.warning(self, "Error", error_msg)
    
    def update_session_list(self):
        """Update session list based on search and filter."""
        self.session_list.clear()
        
        if not self.all_sessions:
            return
        
        # Start with all sessions
        filtered_sessions = self.all_sessions.copy()
        
        # Apply search filter
        search_text = self.search_input.text().strip().lower()
        if search_text:
            filtered_sessions = [s for s in filtered_sessions if search_text in s.lower()]
        
        # Apply data type filter
        filter_type = self.filter_combo.currentText()
        # Note: For now, just show all matching sessions
        # Full filtering would require loading each session's validity
        
        # Display sessions
        if filtered_sessions:
            for session in filtered_sessions:
                self.session_list.addItem(session)
            self.info_label.setText(f"Showing {len(filtered_sessions)} of {len(self.all_sessions)} sessions")
        else:
            self.info_label.setText("No sessions match your search")
    
    def accept(self):
        """Handle OK button or double-click."""
        current_item = self.session_list.currentItem()
        if current_item:
            self.selected_session = current_item.text()
            logging.info(f"Selected session: {self.selected_session}")
            super().accept()
        else:
            QMessageBox.warning(self, "No Selection", "Please select a session first")



# ============================================================================
# Main Replay Manager Window
# ============================================================================

class ReplayManagerQt(QMainWindow):
    """
    Main replay manager window with PyQt5 interface.
    
    Features:
    - Video playback with landmark overlay
    - Real-time EEG visualization (PyQtGraph)
    - Timestamp-based synchronization
    - Session search and filtering
    - Multiple playback modes
    """
    
    def __init__(self, data_folder: str = "./data"):
        super().__init__()
        
        # Data manager
        self.data_manager = DataManager({"folder_path": data_folder})
        self.data_folder = Path(data_folder)
        
        # Current session
        self.current_session: Optional[ReplaySession] = None
        
        # Playback state
        self.current_frame_idx = 0
        self.total_frames = 0
        self.fps = 30.0
        self.is_playing = False
        self.playback_speed = 1.0
        
        # Timestamp synchronization
        self.video_start_time = None
        self.eeg_start_time = None
        self.current_playback_time = 0.0
        self.time_offset = 0.0

        self.video_timestamps: Optional[List[float]] = None
        
        # Timer for playback
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Initialize UI
        self.init_ui()
        
        logging.info("ReplayManagerQt initialized")
    
    def init_ui(self):
        """Initialize main window UI."""
        self.setWindowTitle("NeuroLab Replay Manager")
        self.setGeometry(100, 100, 1600, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Menu bar
        self.create_menu_bar()
        
        # Main splitter (video left, EEG right)
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel: Video and controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Video widget
        self.video_widget = VideoWidget()
        left_layout.addWidget(self.video_widget, stretch=3)
        
        # Control panel
        self.control_panel = ControlPanel()
        left_layout.addWidget(self.control_panel, stretch=1)
        
        splitter.addWidget(left_widget)
        
        # Right panel: EEG plot
        self.eeg_widget = EEGPlotWidget(num_channels=16)
        splitter.addWidget(self.eeg_widget)
        
        # Set splitter sizes
        splitter.setSizes([800, 800])
        
        main_layout.addWidget(splitter)
        
        # Connect signals
        self.connect_signals()
    
    def create_menu_bar(self):
        """Create application menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('&File')
        
        load_action = file_menu.addAction('&Load Session')
        load_action.triggered.connect(self.load_session_dialog)
        load_action.setShortcut('Ctrl+O')
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction('E&xit')
        exit_action.triggered.connect(self.close)
        exit_action.setShortcut('Ctrl+Q')
        
        # View menu
        view_menu = menubar.addMenu('&View')
        
        info_action = view_menu.addAction('Session &Info')
        info_action.triggered.connect(self.show_session_info)
    
    def connect_signals(self):
        """Connect all signals to slots."""
        self.control_panel.play_clicked.connect(self.play)
        self.control_panel.pause_clicked.connect(self.pause)
        self.control_panel.stop_clicked.connect(self.stop)
        self.control_panel.seek_backward_clicked.connect(self.seek_backward)
        self.control_panel.seek_forward_clicked.connect(self.seek_forward)
        self.control_panel.slider_moved.connect(self.seek_to_position)
        self.control_panel.speed_changed.connect(self.change_speed)
    
    # ========================================================================
    # Session Loading Methods
    # ========================================================================
    
    def load_session_dialog(self):
        """Show session selection dialog."""
        try:
            logging.info("Opening session selection dialog...")
            
            # Check if data folder exists
            if not self.data_folder.exists():
                error_msg = f"Data folder not found: {self.data_folder}"
                logging.error(error_msg)
                QMessageBox.warning(self, "Error", error_msg)
                return
            
            # Create and show dialog
            dialog = SessionSelectionDialog(self.data_manager, self)
            
            # Set window modality for macOS
            dialog.setModal(True)
            dialog.raise_()
            dialog.activateWindow()
            
            result = dialog.exec_()
            
            if result == QDialog.Accepted and dialog.selected_session:
                logging.info(f"Dialog accepted, loading: {dialog.selected_session}")
                self.load_session(dialog.selected_session)
            else:
                logging.info("Dialog cancelled or no selection made")
                
        except Exception as e:
            error_msg = f"Error opening session dialog: {str(e)}"
            logging.error(error_msg)
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", error_msg)
    
    def load_session(self, session_name: str) -> bool:
        """
        Load a session by name.
        
        Args:
            session_name: Name of the session to load
        
        Returns:
            bool: True if loaded successfully
        """
        logging.info(f"Loading session: {session_name}")
        
        try:
            # Load data using DataManager
            loaded_data = self.data_manager.load_session(session_name)
            
            if not loaded_data:
                QMessageBox.warning(self, "Load Error", 
                                  f"No data found for session: {session_name}")
                return False
            
            # Create session object
            session = ReplaySession(session_name)
            
            # Load video
            if 'video' in loaded_data:
                session.video_cap = loaded_data['video']
                if session.video_cap.isOpened():
                    self.total_frames = int(session.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    self.fps = session.video_cap.get(cv2.CAP_PROP_FPS)
                    logging.info(f"✓ Video: {self.total_frames} frames @ {self.fps} fps")
            
            # Load EEG
            if 'eeg' in loaded_data:
                session.eeg_data = loaded_data['eeg']
                logging.info(f"✓ EEG: {len(session.eeg_data)} samples")
            
            # Load landmarks
            if 'landmarks' in loaded_data:
                session.landmarks_data = loaded_data['landmarks']
                logging.info(f"✓ Landmarks: {len(session.landmarks_data)} frames")
            # Load video timestamps
            if 'video_timestamps' in loaded_data:
                self.video_timestamps = loaded_data['video_timestamps']
                logging.info(f"✓ Video Timestamps: {len(self.video_timestamps)} entries")
            else:
                self.video_timestamps = None
                logging.warning("✗ Video Timestamps not found")
            # Load metadata
            if 'metadata' in loaded_data:
                session.metadata = loaded_data['metadata']
            
            # Check validity
            validity = session.is_valid()
            available = session.get_available_data()
            
            if not session.has_any_data():
                QMessageBox.warning(self, "Load Error", "Session has no valid data")
                return False
            
            logging.info(f"Available data: {', '.join(available)}")
            
            # Set current session
            self.current_session = session
            self.current_frame_idx = 0
            
            # Initialize timestamps
            self._initialize_timestamps()
            
            # Update UI
            self.control_panel.progress_slider.setMaximum(self.total_frames - 1)
            self.setWindowTitle(f"NeuroLab Replay - {session_name}")
            
            # Show first frame
            self.update_frame()
            
            QMessageBox.information(self, "Success", 
                                   f"Session loaded: {session_name}\n"
                                   f"Available: {', '.join(available)}")
            
            logging.info(f"✓ Session loaded: {session_name}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load session: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load session:\n{str(e)}")
            return False
    
    def _initialize_timestamps(self):
        """Initialize timestamp information for synchronization."""
        if self.current_session is None:
            return
        
        # Use video timestamps if available
        if self.video_timestamps is not None and len(self.video_timestamps) > 0:
            self.video_start_time = self.video_timestamps[0]
            logging.info(f"Video start (from timestamps): {self.video_start_time:.3f}s")
        if self.video_start_time is None and self.current_session.landmarks_data:
            self.video_start_time = self.current_session.landmarks_data[0].get('timestamp', 0.0)
            logging.info(f"Video start (from landmarks): {self.video_start_time:.3f}s")
        
        # Extract EEG start time
        if self.current_session.eeg_data is not None:
            eeg_df = self.current_session.eeg_data
            timestamp_col = None
            for candidate in ('Timestamp', 'timestamp'):
                if candidate in eeg_df.columns:
                    timestamp_col = candidate
                    break

            if timestamp_col is not None:
                self.eeg_start_time = eeg_df[timestamp_col].iloc[0]
                logging.info(f"EEG start: {self.eeg_start_time:.3f}s")
            else:
                self.eeg_start_time = None
                logging.warning("EEG data missing timestamp column; cannot determine start time")
        
        # Calculate time offset
        if self.video_start_time is not None and self.eeg_start_time is not None:
            self.time_offset = self.video_start_time - self.eeg_start_time
            logging.info(f"Time offset: {self.time_offset:.3f}s")
    
    # ========================================================================
    # Playback Control Methods
    # ========================================================================
    
    def play(self):
        """Start playback."""
        if self.current_session is None:
            QMessageBox.warning(self, "No Session", "Please load a session first")
            return
        
        self.is_playing = True
        interval = int(1000 / (self.fps * self.playback_speed))
        self.timer.start(interval)
        logging.info("Playback started")
    
    def pause(self):
        """Pause playback."""
        self.is_playing = False
        self.timer.stop()
        logging.info("Playback paused")
    
    def stop(self):
        """Stop playback and reset to beginning."""
        self.pause()
        self.current_frame_idx = 0
        self.current_playback_time = 0.0
        self.update_frame()
        logging.info("Playback stopped")
    
    def seek_backward(self):
        """Seek backward 30 frames."""
        self.current_frame_idx = max(0, self.current_frame_idx - 30)
        self.update_frame()
    
    def seek_forward(self):
        """Seek forward 30 frames."""
        self.current_frame_idx = min(self.total_frames - 1, self.current_frame_idx + 30)
        self.update_frame()
    
    def seek_to_position(self, position: int):
        """Seek to specific frame position."""
        self.current_frame_idx = position
        self.update_frame()
    
    def change_speed(self, speed: float):
        """Change playback speed."""
        self.playback_speed = speed
        if self.is_playing:
            self.pause()
            self.play()
        logging.info(f"Speed changed to {speed}x")
    
    # ========================================================================
    # Frame Update Method
    # ========================================================================
    
    def update_frame(self):
        """Update display with current frame (called by timer or seek)."""
        if self.current_session is None:
            return
        
        validity = self.current_session.is_valid()
        
        # Update video if available
        if validity['video']:
            self._update_video_frame()
        
        # Update EEG if available
        if validity['eeg']:
            self._update_eeg_display()
        
        # Update controls
        self._update_controls()
        
        # Advance frame if playing
        if self.is_playing:
            self.current_frame_idx += 1
            if self.current_frame_idx >= self.total_frames:
                self.stop()
    
    def _update_video_frame(self):
        """Update video display with timestamp-synchronized landmarks."""
        if not self.current_session.video_cap:
            return
        
        # Read frame
        self.current_session.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.current_session.video_cap.read()
        
        if not ret:
            return
        
        # Calculate current timestamp
        frame_time = self.current_frame_idx / self.fps
        self.current_playback_time = frame_time
        absolute_time = self._get_absolute_time()
        
        # Get synchronized landmarks
        landmarks = self._get_landmarks_at_time(absolute_time)
        
        # Display frame with landmarks
        self.video_widget.display_frame(frame, landmarks)
    
    def _update_eeg_display(self):
        """Update EEG display with timestamp-synchronized data."""
        if self.current_session.eeg_data is None:
            return
        
        absolute_time = self._get_absolute_time()
        eeg_window = self._get_eeg_at_time(absolute_time, window_size=2.0)
        
        if eeg_window is not None and len(eeg_window) > 0:
            self.eeg_widget.update_data(eeg_window)
    
    def _update_controls(self):
        """Update control panel displays."""
        current_time = self.current_frame_idx / self.fps
        total_time = self.total_frames / self.fps
        
        self.control_panel.update_time(current_time, total_time)
        self.control_panel.progress_slider.setValue(self.current_frame_idx)
    
    # ========================================================================
    # Timestamp Synchronization Methods
    # ========================================================================
    
    def _get_absolute_time(self) -> float:
        """Get current absolute timestamp."""
        if self.video_timestamps is not None and self.current_frame_idx < len(self.video_timestamps):
            return self.video_timestamps[self.current_frame_idx]
        elif self.video_start_time is not None:
            return self.video_start_time + self.current_playback_time
        elif self.eeg_start_time is not None:
            return self.eeg_start_time + self.current_playback_time
        else:
            return self.current_playback_time
    
    def _get_eeg_at_time(self, timestamp: float, window_size: float = 1.0) -> Optional[pd.DataFrame]:
        """Get EEG data around specific timestamp."""
        if self.current_session is None or self.current_session.eeg_data is None:
            return None
        
        eeg_df = self.current_session.eeg_data
        timestamp_col = None

        # Support both legacy 'Timestamp' and lowercase 'timestamp' column names.
        for candidate in ('Timestamp', 'timestamp'):
            if candidate in eeg_df.columns:
                timestamp_col = candidate
                break

        if timestamp_col is None:
            logging.warning("EEG dataframe missing timestamp column")
            return None

        mask = (eeg_df[timestamp_col] >= timestamp - window_size) & \
               (eeg_df[timestamp_col] <= timestamp)
        
        return eeg_df[mask]
    
    def _get_landmarks_at_time(self, timestamp: float, tolerance: float = 0.05) -> Optional[Dict]:
        """Get landmarks closest to specific timestamp."""
        if self.current_session is None or not self.current_session.landmarks_data:
            return None
        
        # Find closest landmark
        min_diff = float('inf')
        closest_landmark = None
        
        for landmark in self.current_session.landmarks_data:
            lm_time = landmark.get('timestamp', 0.0)
            diff = abs(lm_time - timestamp)
            
            if diff < min_diff and diff < tolerance:
                min_diff = diff
                closest_landmark = landmark
        
        return closest_landmark
    
    # ========================================================================
    # Info Display
    # ========================================================================
    
    def show_session_info(self):
        """Show current session information."""
        if self.current_session is None:
            QMessageBox.information(self, "Session Info", "No session loaded")
            return
        
        validity = self.current_session.is_valid()
        available = self.current_session.get_available_data()
        
        info_text = f"Session: {self.current_session.session_name}\n\n"
        info_text += "Data Availability:\n"
        
        for data_type, is_valid in validity.items():
            status = "✓" if is_valid else "✗"
            info_text += f"  {status} {data_type.capitalize()}\n"
        
        info_text += "\n"
        
        # Video info
        if validity['video']:
            info_text += f"Video:\n"
            info_text += f"  Total frames: {self.total_frames}\n"
            info_text += f"  FPS: {self.fps:.2f}\n"
            info_text += f"  Duration: {self.total_frames / self.fps:.2f}s\n\n"
        
        # EEG info
        if validity['eeg']:
            eeg_df = self.current_session.eeg_data
            eeg_cols = [c for c in eeg_df.columns if c != 'Timestamp']
            info_text += f"EEG:\n"
            info_text += f"  Samples: {len(eeg_df)}\n"
            info_text += f"  Channels: {len(eeg_cols)}\n\n"
        
        # Landmarks info
        if validity['landmarks']:
            info_text += f"Landmarks:\n"
            info_text += f"  Frames: {len(self.current_session.landmarks_data)}\n\n"
        
        # Timestamp info
        if self.video_start_time:
            info_text += f"Video start: {self.video_start_time:.3f}s\n"
        if self.eeg_start_time:
            info_text += f"EEG start: {self.eeg_start_time:.3f}s\n"
        if self.time_offset != 0.0:
            info_text += f"Time offset: {self.time_offset:.3f}s\n"
        
        QMessageBox.information(self, "Session Info", info_text)
    
    # ========================================================================
    # Window Close Event
    # ========================================================================
    
    def closeEvent(self, event):
        """Handle window close event."""
        self.pause()
        if self.current_session and self.current_session.video_cap:
            self.current_session.video_cap.release()
        event.accept()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    
    # Set application info
    app.setApplicationName("NeuroLab Replay Manager")
    app.setApplicationVersion("2.0")
    
    # Create and show main window
    window = ReplayManagerQt(data_folder="./recordings")
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
