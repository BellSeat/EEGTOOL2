import os, av
import numpy as np
from fractions import Fraction
from typing import Iterable, Optional, Tuple


class Recorder:
    """
    Records frames from a generator into an MP4 container.

    - Handles encoder-specific pixel formats (nv12 for videotoolbox, yuv420p for libx264)
    - Sets a default bitrate, which is critical for hardware encoders.
    - Uses linear PTS based on a target FPS for a smooth playback.
    """

    def __init__(self, out_path: str, fps: int = 30, encoder: str = "libx264", bitrate: Optional[int] = None):
        self.out_path = out_path
        self.fps = int(fps)
        self.encoder = encoder
        
        # A default bitrate is crucial for many encoders (like videotoolbox)
        self.bitrate = bitrate if bitrate else 5_000_000  # 5 Mbps default
        
        self.container = None
        self.stream = None
        self._frame_idx = 0
        self._opened = False
    
    def _ensure_dir(self):
        d = os.path.dirname(self.out_path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
    
    def open(self, width: int, height: int):
        """
        Opens the container and initializes the video stream and codec.
        This is called by write_frames() on the first frame.
        """
        if self._opened:
            return
        self._ensure_dir()
        
        # Open container without fragmented mode - more stable
        self.container = av.open(
            self.out_path, 
            mode="w"
        )
        
        # Prepare codec options and pixel format based on encoder
        codec_options = {}
        
        if self.encoder == "h264_videotoolbox":
            # For videotoolbox, use NV12 which is the native format
            pix_fmt = "nv12"
            codec_options = {
                "realtime": "1",
            }
        elif self.encoder == "libx264":
            pix_fmt = "yuv420p"
            codec_options = {
                "preset": "ultrafast",  # Faster encoding
                "tune": "zerolatency",
            }
        else:
            pix_fmt = "yuv420p"  # Default
        
        # Add the video stream
        self.stream = self.container.add_stream(self.encoder, rate=self.fps)
        
        # Configure stream properties
        self.stream.width = int(width)
        self.stream.height = int(height)
        self.stream.pix_fmt = pix_fmt
        self.stream.time_base = Fraction(1, self.fps)
        
        # Set bitrate
        if self.bitrate:
            self.stream.bit_rate = int(self.bitrate)
        
        # Apply codec options through the stream's options dict
        for key, value in codec_options.items():
            self.stream.options[key] = value
        
        self._opened = True
        print(f"[recorder] opened: {self.out_path} ({width}x{height} @ {self.fps}fps, encoder={self.encoder}, format={pix_fmt})")

    def _check_frame(self, rgb: np.ndarray):
        if not isinstance(rgb, np.ndarray):
            raise TypeError(f"expect numpy ndarray, got {type(rgb)}")
        if rgb.dtype != np.uint8:
            raise TypeError(f"expect uint8 image, got {rgb.dtype}")
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"expect shape (H, W, 3), got {rgb.shape}")
            
    def write_frames(self, gen: Iterable[Tuple[np.ndarray, float]]):
        """
        Consumes frames from a generator and writes them to the video file.
        """
        for rgb, _ts in gen:
            self._check_frame(rgb)
            h, w = rgb.shape[:2]
            if not self._opened:
                # On first frame, open with correct dimensions
                self.open(width=w, height=h)

            # Create video frame from RGB data
            vf = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            
            # Reformat to the encoder's required pixel format
            vf = vf.reformat(
                format=self.stream.pix_fmt,
                width=self.stream.width,
                height=self.stream.height
            )
            
            # Set PTS for this frame
            vf.pts = self._frame_idx
            self._frame_idx += 1

            # Encode and mux
            try:
                for pkt in self.stream.encode(vf):
                    self.container.mux(pkt)
            except Exception as e:
                print(f"[recorder] error at frame {self._frame_idx-1}: {e}")
                # Try to close gracefully and re-raise
                raise
            
    def close(self):
        """
        Flushes the encoder and closes the container.
        """
        if not self._opened:
            return
            
        if self.stream:
            try:
                # Flush remaining frames from encoder
                for pkt in self.stream.encode():
                    self.container.mux(pkt)
            except Exception as e:
                print(f"[recorder] error flushing encoder: {e}")
        
        if self.container:
            try:
                self.container.close()
                print(f"[recorder] closed successfully, wrote {self._frame_idx} frames")
            except Exception as e:
                print(f"[recorder] error closing container: {e}")
        
        self.container = None
        self.stream = None
        self._opened = False


# --- Example usage ---
if __name__ == "__main__":
    import cv2
    from camera import Camera
    import platform
    
    # 1. Initialize Camera
    cam = Camera()
    
    # 2. Initialize Recorder
    # Automatically select the best encoder for the OS
    current_os = platform.system().lower()
    if "darwin" in current_os:
        # Try libx264 first as it's more stable
        encoder = "libx264"  
        print("[main] Note: Using libx264 for better stability. To use hardware encoding, change to 'h264_videotoolbox'")
    else:
        encoder = "libx264"

    print(f"[main] Using encoder: {encoder}")
    
    # Use a path in the current directory for testing
    output_path = "output.mp4"
    
    rec = Recorder(
        output_path, 
        fps=30,  # Should match camera FPS
        encoder=encoder, 
        bitrate=5_000_000  # 5 Mbps
    )
    
    print("Starting preview and recording. Press ESC in preview window to quit.")
    
    frame_count = 0
    try:
        frame_generator = cam.frames()
        
        while True:
            # Get a frame from the camera
            frame, ts = next(frame_generator)
            
            # --- Recording ---
            try:
                rec.write_frames(iter([(frame, ts)]))
                frame_count += 1
                if frame_count % 30 == 0:  # Print every second
                    print(f"[main] recorded {frame_count} frames")
            except Exception as e:
                print(f"Failed to write frame: {e}")
                break
            
            # --- Preview ---
            preview_frame = frame[:, :, ::-1].copy()  # RGB to BGR for OpenCV
            cv2.putText(preview_frame, f"{ts:.2f}s | Frame: {frame_count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Camera Preview", preview_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                print("Exiting.")
                break

    except (KeyboardInterrupt, StopIteration):
        print("\nStopping.")
    finally:
        # Clean up
        cam.close()
        rec.close()
        cv2.destroyAllWindows()
        print(f"File saved to: {os.path.abspath(output_path)}")