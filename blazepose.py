import mediapipe as mp
import json 
import cv2
import numpy as np
import pyvirtualcam
import logging
import os
from typing import Generator, Tuple, Optional,List
from dataclasses import dataclass


from camera import Camera
# import stream.youtube_live as yt_live

# Suppress MediaPipe warnings
os.environ['GLOG_minloglevel'] = '2'

@dataclass
class PoseLandmark:
    name: str
    x: float
    y: float
    z: float
    visibility: Optional[float] = 1.0

@dataclass
class HandLandmark:
    name: str
    x: float
    y: float
    z: float

@dataclass
class FrameData:
    frame_id: int
    timestamp: float
    raw_frame: np.ndarray
    processed_frame: np.ndarray
    pose_landmarks: Optional[List[PoseLandmark]]
    hands_landmarks: Optional[List[List[HandLandmark]]]
    width: int
    height: int
    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "width": self.width,
            "height": self.height,
            "pose_landmarks":[
                {'name': lm.name, 'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility}
                for lm in (self.pose_landmarks or [])
            ],
            "hands_landmarks":[
                [
                    {'name': lm.name, 'x': lm.x, 'y': lm.y, 'z': lm.z}
                    for lm in hand_lm
                ]
                for hand_lm in (self.hands_landmarks or [])
            ]
        }

class BlazePoseCam:
    
    def __init__(self, config_path: str = 'config.json') -> None:
        self._stop_flag = False
        self.frame_queue = []
        self.config_path = config_path
        self.cam = None  # Don't initialize camera yet
        self.is_open = False

        # Load configuration
        try:
            with open(self.config_path, 'r') as f:
                self.cfg = json.load(f)
        except Exception as e:
            logging.error(f"Failed to load config file: {e}")
            self.cfg = {"blazePoseCig": {}}
        # Read MediaPipe parameters from config
        self.image_mod = self.cfg.get("blazePoseCig", {}).get("static_image_mode", False)
        self.image_mod = self.cfg["blazePoseCig"].get("static_image_mode", False)
        self.min_detection_confidence = self.cfg["blazePoseCig"].get("min_detection_confidence", 0.5)
        self.min_tracking_confidence = self.cfg["blazePoseCig"].get("min_tracking_confidence", 0.5)
        self.model_complexity = self.cfg["blazePoseCig"].get("model_complexity", 1)

        # Initialize MediaPipe Pose with additional parameters to reduce warnings
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=self.image_mod, 
            min_detection_confidence=self.min_detection_confidence, 
            min_tracking_confidence=self.min_tracking_confidence, 
            model_complexity=self.model_complexity,
            smooth_landmarks=True,  # Helps with tracking
            enable_segmentation=False  # Disable if not needed
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.image_mod, 
            min_detection_confidence=self.min_detection_confidence, 
            min_tracking_confidence=self.min_tracking_confidence, 
            max_num_hands=2, 
            model_complexity=self.model_complexity
        )
        self.mp_hs_drawing = mp.solutions.drawing_utils

        self.width = None
        self.height = None
        self.fmt = pyvirtualcam.PixelFormat.BGR
        
        logging.info("BlazePoseCam initialized")



    
    def open(self) -> None:
        """Open camera - called by RecordingManager"""
        if self.is_open:
            logging.warning("Camera already open")
            return
        
        self._stop_flag = False
        self.cam = Camera()
        if not self.cam:
            raise IOError("Cannot open webcam")
        self.cam.open()
        self.width = int(self.cam.get_with())
        self.height = int(self.cam.get_height())
        self.is_open = True
        logging.info(f"Camera opened: {self.width}x{self.height}")
    
    def close(self) -> None:
        """Close camera - called by RecordingManager"""
        if not self.is_open:
            return
        
        self._stop_flag = True
        if hasattr(self, '_processing_thread'):
            self._processing_thread.join(timeout=2)
        
        if self.cam:
            self.cam.close()
            self.cam = None
        
        cv2.destroyAllWindows()
        self.is_open = False
        logging.info("Camera closed")
        
    def process_frames(self) -> Generator[Tuple[np.ndarray, float], None, None]:
        """
        Generator compatible with RecordingManager.
        Yields: (frame_data, timestamp)
        where frame_data is a dict containing landmarks and processed image
        """
        if not self.is_open or not self.cam:
            raise RuntimeError("Camera not opened. Call open() first.")
        
        frame_id_counter = 0
        
        for image, ts in self.cam.frames():
            if image is None:
                continue
            
            if self._stop_flag:
                break
                
            frame_id_counter += 1

            
            frame = self._preprocess_frame(image)
            if frame is None:
                logging.warning("Preprocessing returned None frame")
                continue
            # Process with MediaPipe
            frame.flags.writeable = False
            pose_results = self.pose.process(frame)
            hands_results = self.hands.process(frame)
            
            # Collect landmark data
            pose_landmarks = self._extract_pose_landmarks(pose_results)
            hands_landmarks = self._extract_hands_landmarks(hands_results)
            
            # Draw landmarks on image
            processed_frame = self._draw_image(frame.copy(), pose_results, hands_results)
            
            # Package all data together
            frame_data = FrameData(
                frame_id=frame_id_counter,
                timestamp=ts,
                raw_frame=cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),  
                processed_frame=processed_frame,
                pose_landmarks=pose_landmarks,
                hands_landmarks=hands_landmarks,
                width=self.width,
                height=self.height
            )
            
            yield frame_data

    def _preprocess_frame(self, image: np.ndarray) -> np.ndarray:
        try:
            frame = cv2.flip(image, 1)
            if frame.dtype != np.uint8:
                if np.issubdtype(frame.dtype, np.floating):
                    m = frame.max()
                    if m <= 1.0 + 1e-6:
                        frame = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
                    else:
                        frame = np.clip(frame, 0, 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
                if frame.ndim == 2 or frame.shape[2] == 1:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            return frame
        except Exception as e:
            logging.error(f"Error preprocessing frame: {e}")
            return image
    
    def _extract_pose_landmarks(self, results) -> Optional[List[PoseLandmark]]:
        """Extract pose landmarks from results"""
        if not results.pose_landmarks:
            return None
        
        landmarks = []
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            landmarks.append(PoseLandmark(
                name=self.mp_pose.PoseLandmark(idx).name,
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=lm.visibility
            ))
        return landmarks
    def _extract_hands_landmarks(self, results) -> Optional[List[List[HandLandmark]]]:
        """Extract hand landmarks from results"""
        hands_landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                single_hand_lms = []
                for idx, lm in enumerate(hand_landmarks.landmark):
                    single_hand_lms.append(HandLandmark(
                        name=self.mp_hands.HandLandmark(idx).name,
                        x=lm.x,
                        y=lm.y,
                        z=lm.z
                    ))
                hands_landmarks.append(single_hand_lms)
        return hands_landmarks if hands_landmarks else None
    
    def _draw_image(self, image: np.ndarray, results, results_hands) -> np.ndarray:
        """Draw landmarks on image"""
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS
            )
        
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                self.mp_hs_drawing.draw_landmarks(
                    image, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )

        return image
    
    
    def stop(self) -> None:
        """Stop processing"""
        self._stop_flag = True



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    cam = BlazePoseCam()
    try:
        cam.open()
        
        frame_count = 0
        for frame_data in cam.process_frames():
            frame_count += 1
            
            cv2.imshow('BlazePose', frame_data.processed_frame)

            if frame_count % 30 == 0:
                pose_detected = len(frame_data.pose_landmarks) if frame_data.pose_landmarks else 0
                hands_detected = len(frame_data.hands_landmarks) if frame_data.hands_landmarks else 0
                print(f"Frame {frame_count}: Pose landmarks: {pose_detected}, Hands: {hands_detected}")
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
                
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        cam.close()
