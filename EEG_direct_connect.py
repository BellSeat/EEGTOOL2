"""
EEG Direct Connect Controller - BrainFlow EEG Device Interface
Handles direct WiFi connection to OpenBCI Cyton/Daisy boards via BrainFlow library.

Key features:
- Single BoardShim instance (no recreation)
- Separate connection and streaming control
- Comprehensive error handling and logging
- State management for connection and streaming
"""

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
import time
import logging
from typing import Optional, Dict, Tuple
import numpy as np


class EEGDevice:
    """
    Controller for direct WiFi connection to EEG devices via BrainFlow.
    
    This class manages the lifecycle of EEG data acquisition including:
    - Device connection/disconnection
    - Stream start/stop
    - Data retrieval
    - State management
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the EEG controller.
        
        Args:
            config: Configuration dictionary containing:
                - board_id: BrainFlow board ID (default: CYTON_DAISY_WIFI_BOARD)
                - ip_address: Device IP address (default: "192.168.4.1")
                - ip_port: Device port (default: 6987)
                - serial_port: Serial port if using USB
                - Other BrainFlow parameters
        """
        self.config = config or {}
        self.board_id = self.config.get("board_id", BoardIds.CYTON_DAISY_WIFI_BOARD)
        
        # Initialize BrainFlow input parameters
        self.params = BrainFlowInputParams()
        self.params.serial_port = self.config.get("serial_port", "")
        self.params.ip_port = self.config.get("ip_port", 6987)
        self.params.ip_address = self.config.get("ip_address", "192.168.4.1")
        self.params.mac_address = self.config.get("mac_address", "")
        self.params.other_info = self.config.get("other_info", "")
        self.params.timeout = self.config.get("timeout", 0)
        self.params.file = self.config.get("file", "")
        
        # Create BoardShim instance ONCE - do not recreate in wifi_connect()
        self.board = BoardShim(self.board_id, self.params)
        
        # State flags
        self.is_streaming = False
        self.connected = False

        # Timestamp and conveted to perf_counter
        self.stream_start_time_perf = None
        self.stream_start_time_board = None
        
        # Get board information
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.timestamp_channel = BoardShim.get_timestamp_channel(self.board_id)
        
        logging.info(f"EEGDirectConnectController initialized: "
                    f"{len(self.eeg_channels)} channels @ {self.sampling_rate}Hz")
    
    def wifi_connect(self) -> bool:
        """
        Connect to the EEG device via WiFi.
        
        This method only establishes the connection without starting data streaming.
        Call start_stream() separately to begin data acquisition.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        if self.connected:
            logging.warning("Already connected to EEG device")
            return True
        
        try:
            # Prepare session - do NOT recreate BoardShim here
            self.board.prepare_session()
            self.connected = True
            logging.info("EEG device connected successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to connect EEG device: {e}")
            self.connected = False
            return False
    
    def start_stream(self) -> bool:
        """
        Start data streaming from the connected device.
        
        Configures the board settings and begins streaming data.
        Device must be connected first (call wifi_connect()).
        
        Returns:
            bool: True if stream started successfully, False otherwise
        """
        if not self.connected:
            logging.error("Device not connected. Call wifi_connect() first.")
            return False
        
        if self.is_streaming:
            logging.warning("Stream already started")
            return True
        
        try:
            # Configure board settings
            self.board.config_board("~4")   # Enable 16 channels mode
            self.board.config_board("d")    # Set default settings
            self.board.config_board("s500") # Set 500Hz sampling rate
            
            # Start streaming
            self.board.start_stream()
            self.stream_start_time_perf = time.perf_counter()
            time.sleep(0.1)  # Allow buffer to fill
            try:
                calibration_data = self.board.get_board_data(1)
                if calibration_data.size > 0:
                    self.stream_start_time_board = calibration_data[self.timestamp_channel, 0]
                    logging.info(f"Stream calibration: board_ts={self.stream_start_time_board:.3f}, "
                               f"perf_ts={self.stream_start_time_perf:.3f}")
            except Exception as e:
                logging.warning(f"Could not get initial timestamp from board: {e}")
                self.stream_start_time_board = None

            self.is_streaming = True
            logging.info("EEG stream started")
            return True
        except Exception as e:
            logging.error(f"Failed to start stream: {e}")
            self.is_streaming = False
            return False
    
    def stop_stream(self) -> bool:
        """
        Stop data streaming.
        
        Stops the data stream but maintains the connection.
        Can call start_stream() again without reconnecting.
        
        Returns:
            bool: True if stream stopped successfully, False otherwise
        """
        if not self.is_streaming:
            logging.warning("Stream is not running")
            return True
        
        try:
            self.board.stop_stream()
            self.is_streaming = False
            self.stream_start_time_board = None
            self.stream_start_time_perf = None
            logging.info("EEG stream stopped")
            return True
        except Exception as e:
            logging.error(f"Error stopping stream: {e}")
            # Mark as stopped even on error to prevent state inconsistency
            self.is_streaming = False
            return False
    
    def wifi_disconnect(self) -> bool:
        """
        Disconnect from the EEG device.
        
        Stops streaming if active and releases the device session.
        After disconnection, must call wifi_connect() again to reconnect.
        
        Returns:
            bool: True if disconnected successfully, False otherwise
        """
        if not self.connected:
            logging.warning("Device is not connected")
            return True
        
        try:
            # Ensure stream is stopped before disconnecting
            if self.is_streaming:
                self.stop_stream()
                time.sleep(0.5)  # Wait for stream to fully stop
            
            # Release the session
            self.board.release_session()
            self.connected = False
            logging.info("EEG device disconnected")
            return True
        except Exception as e:
            logging.error(f"Error disconnecting device: {e}")
            # Mark as disconnected even on error to prevent state inconsistency
            self.connected = False
            self.is_streaming = False
            return False
    
    def get_data(self, num_samples: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve EEG data from the device buffer.
        
        Args:
            num_samples: Number of samples to retrieve from buffer
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - eeg_data: EEG channel data, shape (num_channels, num_samples)
                - timestamps: Corresponding timestamps, shape (num_samples,)
        
        Raises:
            RuntimeError: If stream is not started
        """
        if not self.is_streaming:
            raise RuntimeError("Stream is not started. Call start_stream() first.")
        
        try:
            # Get data from board buffer
            data = self.board.get_board_data(num_samples)
            
            # Extract EEG channels and timestamps
            eeg_data = data[self.eeg_channels, :]
            timestamps = data[self.timestamp_channel, :]
            if self.stream_start_time_board is not None and self.stream_start_time_perf is not None:
                time_offset = timestamps - self.stream_start_time_board
                comverted_timestamps = self.stream_start_time_perf + time_offset
                return eeg_data, comverted_timestamps
            else:
                logging.warning("Stream start timestamps not available, returning raw board timestamps")
                return eeg_data, timestamps
        except Exception as e:
            logging.error(f"Error getting data: {e}")
            return np.array([]), np.array([])
    
    def get_is_streaming(self) -> bool:
        """
        Check if data streaming is active.
        
        Returns:
            bool: True if streaming, False otherwise
        """
        return self.is_streaming
    
    def get_is_connected(self) -> bool:
        """
        Check if device is connected.
        
        Returns:
            bool: True if connected, False otherwise
        """
        return self.connected
    
    def get_board_info(self) -> Dict:
        """
        Get information about the board configuration.
        
        Returns:
            Dict: Board information including channels, sampling rate, etc.
        """
        return {
            'board_id': self.board_id,
            'num_channels': len(self.eeg_channels),
            'eeg_channels': self.eeg_channels.tolist(),
            'sampling_rate': self.sampling_rate,
            'timestamp_channel': self.timestamp_channel,
            'is_connected': self.connected,
            'is_streaming': self.is_streaming
        }
    
    def close(self):
        """
        Clean up and close the device connection.
        
        This is a convenience method that stops streaming and disconnects.
        Should be called when done using the device.
        """
        logging.info("Closing EEG device...")
        self.stop_stream()
        time.sleep(0.5)
        self.wifi_disconnect()


# ===== Example Usage =====
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Device configuration
    cfg = {
        "board_id": BoardIds.CYTON_DAISY_WIFI_BOARD,
        "serial_port": "",
        "ip_port": 6987,
        "ip_address": "192.168.4.1"
    }
    
    # Create controller instance
    controller = EEGDevice(cfg)
    
    try:
        # Step 1: Connect to device
        if not controller.wifi_connect():
            print("Failed to connect")
            exit(1)
        
        # Step 2: Start streaming
        if not controller.start_stream():
            print("Failed to start stream")
            exit(1)
        
        # Step 3: Wait for data to accumulate
        print("Collecting data...")
        time.sleep(2)
        
        # Step 4: Retrieve data
        eeg_data, timestamps = controller.get_data(512)
        print(f"Got data: {eeg_data.shape}")
        
        # Step 5: Visualize data (first 16 channels)
        num_channels = min(16, eeg_data.shape[0])
        fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 10))
        
        for i in range(num_channels):
            row = i // 4
            col = i % 4
            
            # Plot time-domain signal
            axes[row, col].plot(timestamps, eeg_data[i, :])
            axes[row, col].set_title(f'Channel {i+1}')
            axes[row, col].set_xlabel('Time')
            axes[row, col].set_ylabel('Amplitude')
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Step 6: Clean up
        controller.close()