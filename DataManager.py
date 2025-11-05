# in this script, the DataManager class will mangage the following datat:
# - loading eeg data from CSV files
# - lodaing eeg data from json files
# - loading video data
# - loading annotations data from json files for example {"landmakr_data.json"}

# - mange raw json data in the folder name "raw_data"
# - attach date after all data file_name 
# - able to auto find data wifth same file name prefix. for example "Qingran_eeg_<date>_<envent_name>_<type of data>_<order>.<file type>"
import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Generator, Tuple
from datetime import datetime
from pathlib import Path
import glob
import cv2
import numpy as np
import sqlite3
import logging

class DataCollector:
    def __init__(self):
        self.data = []

    def add(self, item: Any) -> None:
        self.data.append(item)

    def clear(self) -> None:
        self.data.clear()
    def get_all(self) -> List[Any]:
        return self.data
    
    def size(self) -> int:
        return len(self.data)
    
class VideoCollector(DataCollector):
    def add_frame(self, frame: np.ndarray, timestamp: float) -> None:
        self.add((frame, timestamp))

    def get_timestamps(self) -> List[float]:
        return [ts for _, ts in self.data]
    
    def save_timestamps(self, output_path: str) -> bool:
        if not self.data:
            logging.warning("No video frames to save timestamps.")
            return False
        timestamps = self.get_timestamps()
        try:
            with open(output_path, 'w') as f:
                json.dump({"Timestamps": timestamps}, f, indent=2)
            logging.info(f"Saved video timestamps to {output_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to save video timestamps to {output_path}: {e}")
            return False
    def save_to_video(self, output_path: str, fps: float = 30.0) -> bool:
        if not self.data:
            logging.warning("No video frames to save.")
            return False
        
        first_frame, _ = self.data[0]
        height, width= first_frame.shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            logging.error(f"Failed to open video writer for {output_path}")
            return False
        
        for frame, _ in self.data:
            out.write(frame)

        out.release()
        logging.info(f"Saved {len(self.data)} frames to {output_path}")
        return True
class LandmarkCollector(DataCollector):

    def add_landmarks(self, landmarks_dict: Dict) -> None:
        self.add(landmarks_dict)

    def save_to_json(self, output_path: str) -> bool:
        if not self.data:
            logging.warning("No landmark data to save.")
            return False
        
        try:
            with open(output_path, 'w') as f:
                json.dump(self.data, f, indent=4)
            logging.info(f"Saved landmark data to {output_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to save landmark data to {output_path}: {e}")
            return False
        
class EEGCollector(DataCollector):
    def add_eeg_data(self, eeg_data: np.ndarray, timestamps: np.ndarray) -> None:
        self.add((eeg_data, timestamps))

    def to_dataframe(self) -> Optional[pd.DataFrame]:
        if not self.data:
            logging.warning("No EEG data to convert.")
            return None
        
        all_eeg = []
        all_timestamps = []

        for eeg_data, timestamp in self.data:
            all_eeg.append(eeg_data)
            all_timestamps.append(timestamp)
        
        # merege data
        combined_eeg = np.concatenate(all_eeg, axis=1)
        combined_timestamps = np.concatenate(all_timestamps)

        # create dataframe
        df = pd.DataFrame(combined_eeg.T)
        df.insert(0, 'Timestamp', combined_timestamps)

        return df
    
    def save_to_csv(self,output_path:str) -> bool:
        df = self.to_dataframe()
        if df is None:
            logging.warning("No EEG data to save.")
            return False
        try:
            df.to_csv(output_path, index=False)
            logging.info(f"Saved EEG data to {output_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to save EEG data to {output_path}: {e}")
            return False
        
    def save_to_json(self,output_path:str) -> bool:
        df = self.to_dataframe()
        if df is None:
            logging.warning("No EEG data to save.")
            return False
        try:
            df.to_json(output_path, orient='records', indent=2)
            logging.info(f"Saved EEG data ({len(df)} samples) to {output_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to save EEG data to {output_path}: {e}")
            return False
        
class DataManager:
    """
    DataManager -- manage all data collection and storeing operations
    """
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.folder_path = Path(self.config.get("folder_path", "./data"))

        # create folder if not exist
        self.folder_path.mkdir(parents=True, exist_ok=True)

        # file name config
        self.prefix_name = self.config.get("file_prefix_name", "Neurolab")
        self.prefix_event = self.config.get("file_prefix_event", "Experiment")

        # set-up collectors
        self.video_collector = VideoCollector()
        self.landmark_collector = LandmarkCollector()
        self.eeg_collector = EEGCollector()

        # status
        self.current_session_name = None

        logging.info(f"DataManager initialized with folder: {self.folder_path}")

    def start_new_session(self, event_name: Optional[str] = None) -> str:
        """
        start a new data collection session
        Args:
            event_name (Optional[str]): name of the event, default to self.prefix_event
        Returns:
            str: generated session name
        """
        if event_name:
            self.prefix_event = event_name
        data_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{self.prefix_name}_{data_str}_{self.prefix_event}"

        # checsk existing files with same prefix, then add order number
        counter = 0
        session_name = base_name
        while self._session_exists(session_name):
            counter += 1
            session_name = f"{base_name}_{counter:02d}"

        self.current_session_name = session_name
        logging.info(f"Started new session: {self.current_session_name}")
        return session_name

    def _session_exists(self, session_name: str) -> bool:
        """
        check if any data file exists with the given session name prefix
        Args:
            session_name (str): session name prefix
        Returns:
            bool: True if any file exists, False otherwise
        """
        pattern = str(self.folder_path / f"{session_name}_*.*")
        existing_files = glob.glob(pattern)
        return len(existing_files) > 0
    
    # ====== data colloection methods ======

    def collect_video_frame(self, frame: np.ndarray, timestamp: float) -> None:
        self.video_collector.add_frame(frame, timestamp)

    def collect_landmarks(self, landmarks_dict: Dict) -> None:
        self.landmark_collector.add_landmarks(landmarks_dict)

    def collect_eeg_data(self, eeg_data: np.ndarray, timestamps: np.ndarray) -> None:
        self.eeg_collector.add_eeg_data(eeg_data, timestamps)

    # ====== general collections methods ======

    def collect_frame_data(self, frame_data) -> None:
        """
        Collect data from blazepose as FrameData
        Args:
            frame_data: FrameData object from blazepose
        """
        self.collect_video_frame(frame_data.raw_frame, frame_data.timestamp)

        if hasattr(frame_data, "to_dict"):
            landmarks_dict = frame_data.to_dict()
        else:
            landmarks_dict = {
                "frame_id": getattr(frame_data, "frame_id", 0),
                "timestamp": getattr(frame_data, "timestamp", 0.0)
            }
        self.collect_landmarks(landmarks_dict)

    # ====== data saving methods ======
    
    def save_all(self, video_fps: float = 30.0) -> Dict[str, bool]:
        """
        Save all collected data to files
        Args:
            video_fps (float): frame per second for video saving
        Returns:
            Dict[str, bool]: save status for each data type
        """
        if not self.current_session_name:
            logging.error("No active session. Call start_new_session first.")
            return {}
        results = {}

        # save video
        if self.video_collector.size() > 0:
            video_path = str(self.folder_path / f"{self.current_session_name}_video.mp4")
            results['video'] = self.video_collector.save_to_video(video_path, video_fps)
            timestamps_path = str(self.folder_path / f"{self.current_session_name}_video_timestamps.json")
            results['video_timestamps'] = self.video_collector.save_timestamps(timestamps_path)
        # save landmarks
        if self.landmark_collector.size() > 0:
            landmark_path = str(self.folder_path / f"{self.current_session_name}_landmarks.json")
            results['landmarks'] = self.landmark_collector.save_to_json(landmark_path)

        # save EEG data
        if self.eeg_collector.size() > 0:
            eeg_csv_path = str(self.folder_path / f"{self.current_session_name}_eeg.csv")
            eeg_json_path = str(self.folder_path / f"{self.current_session_name}_eeg.json")
            results['eeg_csv'] = self.eeg_collector.save_to_csv(eeg_csv_path)
            results['eeg_json'] = self.eeg_collector.save_to_json(eeg_json_path)
        
        metadata_path = self.folder_path / f"{self.current_session_name}_metadata.json"
        results['metadata'] = self._save_metadata(metadata_path)
        logging.info(f"Data saving results: {results}")
        return results
    
    def _save_metadata(self, output_path: Path) -> bool:

        metadata = {
            'session_name': self.current_session_name,
            'prefix_name': self.prefix_name,
            'prefix_event': self.prefix_event,
            'created_at': datetime.now().isoformat(),
            'video_frames': self.video_collector.size(),
            'landmark_frames': self.landmark_collector.size(),
            'eeg_batches': self.eeg_collector.size()
        }
        try:
            with open(output_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logging.info(f"Saved metadata to {output_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to save metadata to {output_path}: {e}")
            return False
    
    # ====== data status checking ======
    def get_stats(self) -> Dict[str, int]:
        return {
            'video_frames': self.video_collector.size(),
            'landmark_frames': self.landmark_collector.size(),
            'eeg_batches': self.eeg_collector.size()
        }
    
    def reset(self) -> None:
        self.video_collector.clear()
        self.landmark_collector.clear()
        self.eeg_collector.clear()
        self.current_session_name = None
        logging.info("DataManager reset: all collectors cleared and session name reset.")

    def load_session(self, session_name: str) -> Dict[str, Any]:
        """
        load seesion data from seesion name

        Args:
            session_name (str): session name prefix
        Returns:
            Dict[str, Any]: loaded data for each type
        """

        data = {}

        # load video
        video_path = self.folder_path / f"{session_name}_video.mp4"
        if video_path.exists():
            data['video'] = cv2.VideoCapture(str(video_path))
            logging.info(f"Loaded video data from {video_path}")
        else:
            logging.warning(f"Video file not found: {video_path}")

        # load video timestamps
        video_timestamps_path = self.folder_path / f"{session_name}_video_timestamps.json"
        if video_timestamps_path.exists():
            with open(video_timestamps_path, 'r') as f:
                timestamps_payload = json.load(f)

            timestamps = None
            if isinstance(timestamps_payload, dict):
                timestamps = timestamps_payload.get("Timestamps")
            elif isinstance(timestamps_payload, list):
                timestamps = timestamps_payload

            if timestamps is None:
                logging.warning(f"Video timestamps file malformed: {video_timestamps_path}")
            else:
                data['video_timestamps'] = timestamps
                logging.info(f"Loaded video timestamps from {video_timestamps_path} ({len(timestamps)} entries)")
        else:
            logging.warning(f"Video timestamps file not found: {video_timestamps_path}")

        # load landmarks
        landmark_path = self.folder_path / f"{session_name}_landmarks.json"
        if landmark_path.exists():
            with open(landmark_path, 'r') as f:
                data['landmarks'] = json.load(f)
            logging.info(f"Loaded landmark data from {landmark_path}")
        else:
            logging.warning(f"Landmark file not found: {landmark_path}")


        # load EEG data
        eeg_csv_path = self.folder_path / f"{session_name}_eeg.csv"
        if eeg_csv_path.exists():
            data['eeg'] = pd.read_csv(eeg_csv_path)  #
            logging.info(f"Loaded EEG data from {eeg_csv_path}")
        else:
            logging.warning(f"EEG CSV file not found: {eeg_csv_path}")

            eeg_json_path = self.folder_path / f"{session_name}_eeg.json"
            if eeg_json_path.exists():
                with open(eeg_json_path, 'r') as f:
                    eeg_json = json.load(f)
                    data['eeg'] = pd.DataFrame(eeg_json)  # 
                logging.info(f"Loaded EEG data from {eeg_json_path}")
            else:
                logging.warning(f"EEG JSON file not found: {eeg_json_path}")


        # load metadata
        metadata_path = self.folder_path / f"{session_name}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                data['metadata'] = json.load(f)
            logging.info(f"Loaded metadata from {metadata_path}")

        if data:
            logging.info(f"Loaded session data for {session_name}: {list(data.keys())}")
        else:
            logging.warning(f"No data found for session: {session_name}")
        return data
    
    def list_sessions(self) -> List[str]:
        pattern = str(self.folder_path / f"*_metadata.json")
        metadata_files = glob.glob(pattern)

        sessions = []
        for f in metadata_files:
            session_name = Path(f).stem.replace("_metadata", "")
            sessions.append(session_name)
        
        sessions.sort(reverse=True)
        return sessions
    def find_session_by_prefix(self, prefix: str) -> List[str]:
        all_sessions = self.list_sessions()
        matched_sessions = [s for s in all_sessions if s.startswith(prefix)]
        logging.info(f"Found {len(matched_sessions)} sessions with prefix '{prefix}'")
        return matched_sessions
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dm = DataManager({
        "folder_path": "./data",
        "file_prefix_name": "Test",
        "file_prefix_event": "Demo"
    })
    

    session = dm.start_new_session("TestRun")
    print(f"Session: {session}")
    

    for i in range(10):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dm.collect_video_frame(frame, i * 0.033)
        
        landmarks = {
            'frame_id': i,
            'timestamp': i * 0.033,
            'pose_landmarks': [{'name': f'point_{j}', 'x': 0.5, 'y': 0.5, 'z': 0.0} for j in range(33)]
        }
        dm.collect_landmarks(landmarks)
        
        eeg_data = np.random.randn(16, 128)
        timestamps = np.linspace(i * 0.033, (i + 1) * 0.033, 128)
        dm.collect_eeg_data(eeg_data, timestamps)
    
    print("Stats:", dm.get_stats())
    
    results = dm.save_all()
    print("Save results:", results)
    
    sessions = dm.list_sessions()
    print("Available sessions:", sessions)
    
    if sessions:
        loaded_data = dm.load_session(sessions[0])
        print("Loaded data keys:", loaded_data.keys())
