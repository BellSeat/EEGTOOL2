from DataManager import DataManager
from EEG_direct_connect import EEGDevice
from blazepose import BlazePoseCam
from EEGMatplotDisplay import EEGMatplotDisplay
from qt_preview import QtPreview
from typing import Optional, Dict, List, Generator

import numpy as np
import time
import logging
import threading
import queue

class RecordingManager:

    def __init__(self,config: Optional[Dict] = None) -> None:
        self.config = config or {}
        
        # init EEG plotter
        self.data_manager = DataManager(config)
        self.eeg_device = EEGDevice(config)
        self.pose_detector = BlazePoseCam()
        
        # init device status
        self.has_camera = False
        self.has_eeg = False
        
        # queue for communication between producer and consumer
        self.frame_queue = queue.Queue(maxsize=100)
        self.eeg_queue = queue.Queue(maxsize=100)
        
        # Threading control
        self.stop_event = threading.Event()
        self.producer_threads = []
        self.consumer_thread = None
        
        # logging setup
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        preview_default = self.config.get("preview", True)
        self.previewing = bool(preview_default)
        self.preview = None
        if self.previewing:
            self.preview = QtPreview(window_title="Camera Preview")
            if not self.preview.is_available:
                self.logger.warning("Qt preview disabled: no Qt backend found.")
                self.previewing = False

    def check_devices(self) -> Dict[str, bool]:
        device_status = {
            "eeg": self._check_eeg(),
            "camera": self._check_cammera()
        }
        return device_status

    def _check_cammera(self) -> bool:
        try:
            self.pose_detector.open()
            self.has_camera = True
            self.pose_detector.close()
            self.logger.info("Camera check: PASS")
            return True
        except Exception as e:
            self.logger.error(f"Camera check failed: {e}")
            self.has_camera = False
            return False
            
        
    def _check_eeg(self) -> bool:

        try:
            self.logger.info("Testing EEG connection...")
    
            if not self.eeg_device.wifi_connect():
                self.logger.error("Failed to connect to EEG device")
                self.has_eeg = False
                return False

            if not self.eeg_device.start_stream():
                self.logger.error("Failed to start EEG stream")
                self.eeg_device.wifi_disconnect()
                self.has_eeg = False
                return False
 
            timeout = 5
            start = time.time()
            eeg_data = None
            
            while time.time() - start < timeout:
                eeg_data, timestamps = self.eeg_device.get_data(256)
                if eeg_data is not None and len(eeg_data) > 0:
                    break
                time.sleep(0.1)
            
            self.eeg_device.stop_stream()
            time.sleep(0.5)
            self.eeg_device.wifi_disconnect()
            
            if eeg_data is None or len(eeg_data) == 0:
                self.logger.error("No EEG data received during test")
                self.has_eeg = False
                return False
            
            self.logger.info("EEG check: PASS")
            self.has_eeg = True
            return True
            
        except Exception as e:
            self.logger.error(f"EEG check failed: {e}")
            import traceback
            traceback.print_exc()
            self.has_eeg = False
            try:
                self.eeg_device.stop_stream()
                time.sleep(0.5)
                self.eeg_device.wifi_disconnect()
            except:
                pass
            return False
    
    #  --- use producer and consumer pattern to handle data recording ---

    def start_recording(self, duration: int = 60, event_name: Optional[str] = None) -> bool:
        device_status = self.check_devices()
        if not device_status["eeg"]:
            self.logger.error("Cannot start recording: EEG device not available.")
            return False
        if device_status["camera"]:
            self.logger.warning("Camera detected and will be used for recording.")
            self.has_camera = True
        elif not device_status["camera"]:
            self.logger.warning("Camera not detected. Proceeding without camera.")
            self.has_camera = False

        session_name = self.data_manager.start_new_session(event_name)
        self.logger.info(f"Recording session '{session_name}' started for duration {duration} seconds.")

        self.stop_event.clear()

        try:
            self._initialize_devices()
            self._start_threads(duration)

            self.logger.info("Recording in progress...")
            for thread in self.producer_threads:
                thread.join()

            if self.consumer_thread:
                self.consumer_thread.join()
            self.logger.info("Recording completed successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Recording error: {e}")
            self.stop_event.set()
            return False
        finally:
            self._cleanup_devices()

    def _initialize_devices(self) -> None:
        self.logger.info("Initializing devices...")
        if not self.eeg_device.wifi_connect():
            raise RuntimeError("Failed to connect to EEG device.")
        if not self.eeg_device.start_stream():
            self.eeg_device.wifi_disconnect()
            raise RuntimeError("Failed to start EEG stream.")
        time.sleep(2)  # Allow time for data accumulation

        if self.has_camera:
            self.logger.info("Initializing camera...")
            self.pose_detector.open()
            time.sleep(2)
            if self.previewing and self.preview:
                self.preview.start()
        
    def _cleanup_devices(self) -> None:
        self.logger.info("Cleaning up devices...")
        try:
            self.eeg_device.stop_stream()
            time.sleep(0.5)
            self.eeg_device.wifi_disconnect()
        except Exception as e:
            self.logger.error(f"Error disconnecting EEG device: {e}")

        if self.has_camera:
            try:
                self.pose_detector.close()
            except Exception as e:
                self.logger.error(f"Error closing camera: {e}")
        if self.preview:
            self.preview.stop()

    def _start_threads(self, duration: int) -> None:
        eeg_thread = threading.Thread(
            target = self._eeg_produceer,
            args = (duration,),
            name = "EEGProducerThread"
        )
        eeg_thread.start()
        self.producer_threads.append(eeg_thread)
        if self.has_camera:
            camera_thread = threading.Thread(
                target = self._camera_produceer,
                args = (duration,),
                name = "CameraProducerThread"
            )
            camera_thread.start()
            self.producer_threads.append(camera_thread)

        self.consumer_thread = threading.Thread(
            target = self._consumer,
            args = (duration,),
            name = "ConsumerThread"
        )
        self.consumer_thread.start()
    
    def _eeg_produceer(self, duration: int = 60) -> None:
        self.logger.info("EEG producer started.")
        start_time = time.time()
        batch_count = 0

        while time.time() - start_time < duration and not self.stop_event.is_set():
            try:
                eeg_data, timestamps = self.eeg_device.get_data(256)
                if eeg_data is not None and len(eeg_data) > 0:
                    self.eeg_queue.put(('eeg',eeg_data, timestamps))
                    batch_count += 1
            except Exception as e:
                self.logger.error(f"EEG producer error: {e}")
                break
        self.logger.info(f"EEG producer finished. Total batches produced: {batch_count}")

    def _camera_produceer(self, duration: int = 60) -> None:
        self.logger.info("Camera producer started.")
        start_time = time.time()
        frame_count = 0

        try:
            for frame_data in self.pose_detector.process_frames():
                if time.time() - start_time >= duration or self.stop_event.is_set():
                    break
                if not self.frame_queue.full():
                    self.frame_queue.put(('video',frame_data))
                    frame_count += 1
                else:
                    self.logger.warning("Frame queue full, skipping frame.")
                if self.previewing and self.preview:
                    self.preview.show(frame_data.raw_frame)
        except Exception as e:
            self.logger.error(f"Camera producer error: {e}")

        self.logger.info(f"Camera producer finished. Total frames produced: {frame_count}")


    def _consumer(self, duration: int = 60) -> None:
        self.logger.info("Consumer started.")
        start_time = time.time()

        frame_count = 0
        eeg_count = 0

        while True:
            processed = False

            try:
                data_type, frame_data = self.frame_queue.get(timeout=0.1)
                if data_type == 'video':
                    self.data_manager.collect_frame_data(frame_data)
                    frame_count += 1
                else:
                    self.logger.warning(f"Unknown frame queue data type: {data_type}")
                self.frame_queue.task_done()
                processed = True
            except queue.Empty:
                pass
            except Exception as e:
                self.logger.error(f"Consumer frame processing error: {e}")

            try:
                data_type, eeg_data, timestamps = self.eeg_queue.get_nowait()
                if data_type == 'eeg':
                    self.data_manager.collect_eeg_data(eeg_data, timestamps)
                    eeg_count += 1
                else:
                    self.logger.warning(f"Unknown EEG queue data type: {data_type}")
                self.eeg_queue.task_done()
                processed = True
            except queue.Empty:
                pass
            except Exception as e:
                self.logger.error(f"Consumer EEG processing error: {e}")

            if self.stop_event.is_set() and self.frame_queue.empty() and self.eeg_queue.empty():
                break

            if not processed:
                if time.time() - start_time >= duration and self.frame_queue.empty() and self.eeg_queue.empty():
                    break

        self.logger.info(f"Consumer finished. Total frames consumed: {frame_count}, EEG batches consumed: {eeg_count}")
        


    def stop_recording(self, save: bool = True) -> Dict[str, bool]:

        self.logger.info("Stopping recording...")
        self.stop_event.set()
        
        for thread in self.producer_threads:
            if thread.is_alive():
                thread.join(timeout=5)
        
        if self.consumer_thread and self.consumer_thread.is_alive():
            self.consumer_thread.join(timeout=5)
        
        self._cleanup_devices()

        save_results = {}
        if save:
            self.logger.info("Saving data...")
            stats = self.data_manager.get_stats()
            self.logger.info(f"Data stats: {stats}")
            save_results = self.data_manager.save_all()

        self.data_manager.reset()
        self.producer_threads.clear()
        self.consumer_thread = None
        
        self.logger.info("Recording stopped")
        return save_results

if __name__ == "__main__":
    config = {
        "folder_path": "./recordings",
        "file_prefix_name": "NeuroLab",
        "file_prefix_event": "Test"
    }
    
    recorder = RecordingManager(config)
    
    try:
        # 开始录制
        success = recorder.start_recording(duration=30, event_name="Demo")
        
        if success:
            # 保存并停止
            results = recorder.stop_recording(save=True)
            print(f"Save results: {results}")
        else:
            print("Failed to start recording")

            
    except KeyboardInterrupt:
        print("\nStopped by user")
        recorder.stop_recording(save=True)
    
    dm = DataManager({
        "folder_path": "./recordings",
    })
    sessions = dm.list_sessions()
    print(f"Sessions recorded: {sessions}")

    
        
