import av
import numpy as np
from fractions import Fraction
from typing import Optional, Tuple, Iterable
import socket
import time


class YouTubeStreamer:
    """
    Streams video frames to YouTube Live via RTMP.
    
    Uses FLV container format and H.264 encoding optimized for live streaming.
    """
    
    def __init__(self, rtmp_url: str, fps: int = 30, encoder: str = "libx264",
                 bitrate: int = 4_000_000):
        """
        Initialize YouTube streamer.
        
        Args:
            rtmp_url: YouTube RTMP URL (rtmp://a.rtmp.youtube.com/live2/YOUR_STREAM_KEY)
            fps: Frames per second (default: 30)
            encoder: Video encoder (default: "libx264")
            bitrate: Bitrate in bits per second (default: 4 Mbps)
        """
        self.url = rtmp_url
        self.fps = fps
        self.encoder = encoder
        self.bitrate = bitrate
        self.container = None
        self.video_stream = None # Renamed from self.stream
        self.audio_stream = None # NEW: Must have audio
        self.audio_frame = None  # NEW: Re-usable silent audio frame
        self.audio_frame_pts = 0 # NEW: Audio PTS counter
        
        self._frame_idx = 0
        self._opened = False
        self._last_successful_send = None
        self._connection_test_passed = False
        
    @staticmethod
    def test_rtmp_server(url: str, timeout: float = 5.0) -> dict:
        """
        Test if RTMP server is reachable before attempting to stream.
        (This method is unchanged and correct)
        """
        try:
            # Parse RTMP URL to get hostname
            if url.startswith("rtmp://"):
                url_part = url.replace("rtmp://", "").split("/")[0]
                if ":" in url_part:
                    host, port = url_part.split(":")
                    port = int(port)
                else:
                    host = url_part
                    port = 1935  # Default RTMP port
            else:
                return {"reachable": False, "message": "Invalid RTMP URL format"}
            
            # Test TCP connection to RTMP server
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            
            try:
                sock.connect((host, port))
                sock.close()
                return {
                    "reachable": True, 
                    "message": f"RTMP server {host}:{port} is reachable",
                    "host": host,
                    "port": port
                }
            except socket.timeout:
                return {
                    "reachable": False, 
                    "message": f"Connection timeout to {host}:{port}",
                    "host": host,
                    "port": port
                }
            except socket.error as e:
                return {
                    "reachable": False, 
                    "message": f"Cannot connect to {host}:{port} - {e}",
                    "host": host,
                    "port": port
                }
        except Exception as e:
            return {"reachable": False, "message": f"Error testing connection: {e}"}
    
    def open(self, width: int, height: int):
        """
        Opens the RTMP connection and initializes the video and audio streams.
        
        Args:
            width: Video width in pixels
            height: Video height in pixels
        """
        if self._opened:
            return
        
        # Test RTMP server before attempting connection
        print("[streamer] Testing RTMP server connectivity...")
        test_result = self.test_rtmp_server(self.url)
        
        if not test_result["reachable"]:
            raise ConnectionError(f"Cannot reach RTMP server: {test_result['message']}")
        
        print(f"[streamer] ✓ RTMP server reachable: {test_result['host']}:{test_result['port']}")
        print(f"[streamer] Opening RTMP connection...")
        
        # Open FLV container for RTMP streaming
        try:
            self.container = av.open(self.url, mode="w", format="flv")
        except Exception as e:
            raise ConnectionError(f"Failed to open RTMP connection: {e}")
        
        # --- VIDEO STREAM (MAJOR CHANGES) ---
        
        # Define encoder options for YouTube Live
        video_options = {
            "preset": "veryfast",      # Fast encoding
            "tune": "zerolatency",     # Minimize latency
            "profile": "baseline",     # Better compatibility
            "g": str(self.fps * 2),    # Keyframe interval (2 seconds)
            "b_frames": "0",           # No B-frames for low latency
        }

        # Add video stream
        self.video_stream = self.container.add_stream(self.encoder, rate=self.fps, options=video_options)
        
        # Set properties on the stream AND the codec context
        self.video_stream.width = width
        self.video_stream.height = height
        self.video_stream.pix_fmt = "yuv420p"

        # Get codec context and set its properties
        vcc = self.video_stream.codec_context
        vcc.width = width
        vcc.height = height
        vcc.pix_fmt = "yuv420p"
        vcc.bit_rate = self.bitrate
        vcc.time_base = Fraction(1, self.fps) # Use stream.rate for timebase
        vcc.framerate = self.fps
        
        try:
            vcc.flags |= av.codec.Flag.GLOBAL_HEADER
        except Exception:
            vcc.flags |= 0x00400000  # AV_CODEC_FLAG_GLOBAL_HEADER
            
        # Explicitly open the codec
        vcc.open()
        
        # --- AUDIO STREAM (ALL NEW) ---
        # YouTube REQUIRES an audio stream
        audio_options = {
            "ar": "44100",  # Sample rate
            "ac": "2",      # Stereo
        }
        self.audio_stream = self.container.add_stream("aac", rate=44100, options=audio_options)
        
        acc = self.audio_stream.codec_context
        acc.bit_rate = 128_000 # 128k audio bitrate
        acc.sample_rate = 44100
        acc.layout = "stereo"
        # acc.channels = 2
        acc.time_base = Fraction(1, 44100) # Timebase must match sample rate

        try:
            acc.flags |= av.codec.Flag.GLOBAL_HEADER
        except Exception:
            acc.flags |= 0x00400000
            
        acc.open()
        
       # Create a re-usable silent audio frame
        self.audio_frame = av.AudioFrame(format='flt', layout='stereo', samples=1024)
        self.audio_frame.sample_rate = 44100
        
        # --- NEW, SIMPLER CODE ---
       # Create silent data directly as bytes
        bytes_needed = self.audio_frame.samples * self.audio_frame.format.bytes * self.audio_frame.layout.nb_channels
        silent_bytes = b'\x00' * bytes_needed
        
        # Update the (single) plane with the silent bytes
        for p in self.audio_frame.planes:
            p.update(silent_bytes)
        # --- END NEW CODE ---

        
        self._opened = True
        self._connection_test_passed = True
        print("[streamer] ✓ RTMP connection established (video + audio)!")
        print("[streamer] ℹ  Note: Stream will appear on YouTube after 10-30 seconds")
        
    def send_frames(self, frames: Iterable[Tuple[np.ndarray, float]]):
        """
        Send video and silent audio frames to YouTube Live stream.
        
        Args:
            frames: Iterable of (rgb_frame, timestamp) tuples
        """
        for rgb, _ts in frames:
            # Open stream on first frame
            if not self._opened:
                h, w = rgb.shape[:2]
                self.open(w, h)
            
            # --- Send Video Frame (Same as before) ---
            try:
                vf = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                vf = vf.reformat(
                    width=self.video_stream.width,
                    height=self.video_stream.height,
                    format=self.video_stream.pix_fmt
                )
                vf.pts = self._frame_idx
                self.video_stream.codec_context.time_base = Fraction(1, self.fps)
                
                for pkt in self.video_stream.encode(vf):
                    self.container.mux(pkt)
                
                self._frame_idx += 1
                
            except Exception as e:
                print(f"[streamer] Error sending video frame {self._frame_idx}: {e}")
                self._connection_test_passed = False
                raise

            # --- Send Silent Audio Frame (NEW) ---
            try:
                self.audio_frame.pts = self.audio_frame_pts
                self.audio_frame_pts += self.audio_frame.samples
                self.audio_stream.codec_context.time_base = Fraction(1, 44100)

                for pkt in self.audio_stream.encode(self.audio_frame):
                    self.container.mux(pkt)

                self._last_successful_send = time.time()
                
            except Exception as e:
                print(f"[streamer] Error sending audio frame: {e}")
                self._connection_test_passed = False
                raise
                
    def check_connection(self) -> dict:
        """
        Check RTMP connection status.
        (Unchanged, but now tracks video_stream)
        """
        now = time.time()
        time_since_last_send = None
        if self._last_successful_send:
            time_since_last_send = now - self._last_successful_send
        
        status = {
            "connected": self._opened and self.container is not None,
            "frames_sent": self._frame_idx,
            "container_open": self.container is not None,
            "video_stream_ready": self.video_stream is not None,
            "audio_stream_ready": self.audio_stream is not None,
            "connection_test_passed": self._connection_test_passed,
            "actively_streaming": time_since_last_send is not None and time_since_last_send < 5.0,
            "last_send_seconds_ago": time_since_last_send,
            "url": self.url if self._opened else None,
            "resolution": f"{self.video_stream.width}x{self.video_stream.height}" if self.video_stream else None,
            "fps": self.fps if self._opened else None,
            "bitrate": f"{self.bitrate/1_000_000:.1f}Mbps" if self._opened else None,
        }
        return status
    
    def is_connected(self) -> bool:
        """
        Simple check if RTMP connection is active and working.
        (Unchanged, but now checks both streams)
        """
        if not (self._opened and self.container is not None and self.video_stream is not None and self.audio_stream is not None):
            return False
        
        # If we've sent frames, check if we're still actively sending
        if self._last_successful_send:
            return (time.time() - self._last_successful_send) < 5.0
        
        # If no frames sent yet, just check if connection is open
        return True
    
    def close(self):
        """
        Flush encoders and close RTMP connection.
        """
        if not self._opened:
            return
            
        print(f"[streamer] Closing stream... (sent {self._frame_idx} video frames)")
        
        # Flush video
        if self.video_stream:
            try:
                for pkt in self.video_stream.encode():
                    self.container.mux(pkt)
            except Exception as e:
                print(f"[streamer] Error flushing video encoder: {e}")

        # Flush audio
        if self.audio_stream:
            try:
                for pkt in self.audio_stream.encode():
                    self.container.mux(pkt)
            except Exception as e:
                print(f"[streamer] Error flushing audio encoder: {e}")
        
        if self.container:
            try:
                self.container.close()
            except Exception as e:
                print(f"[streamer] Error closing container: {e}")
        
        self.container = None
        self.video_stream = None
        self.audio_stream = None
        self._opened = False
        self._connection_test_passed = False
        self._last_successful_send = None
        print("[streamer] Stream closed")

# --- Example Usage ---
if __name__ == "__main__":
    import sys
    import os
    import cv2
    
    # Add parent directory to path to import camera
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from camera import Camera
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        # Check for test mode
        if arg.lower() in ['test', 'demo', 'dry-run']:
            print("=" * 60)
            print("TEST MODE - No actual streaming")
            print("=" * 60)
            print("This will preview the camera without streaming.")
            print("Use this to test your camera setup.")
            print("=" * 60)
            
            cam = Camera()
            frame_count = 0
            start_time = time.time()
            
            try:
                frame_generator = cam.frames()
                
                while True:
                    frame, ts = next(frame_generator)
                    frame_count += 1
                    
                    if frame_count % 30 == 0:
                        elapsed = time.time() - start_time
                        fps_actual = frame_count / elapsed if elapsed > 0 else 0
                        print(f"[test] {frame_count} frames | {fps_actual:.1f} fps")
                    
                    preview_frame = frame[:, :, ::-1].copy()
                    cv2.putText(preview_frame, f"TEST MODE | Frame: {frame_count}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("Camera Test - Press ESC to exit", preview_frame)
                    
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
            finally:
                cam.close()
                cv2.destroyAllWindows()
                print(f"\nTest complete: {frame_count} frames captured")
            
            sys.exit(0)
        else:
            stream_key = arg
    else:
        print("=" * 60)
        print("YouTube Live Streamer")
        print("=" * 60)
        print("\nUsage:")
        print("  python youtube_live.py YOUR_STREAM_KEY")
        print("  python youtube_live.py test              # Test mode (no streaming)")
        print("\nExamples:")
        print("  python youtube_live.py abcd-1234-efgh-5678")
        print("  python youtube_live.py test")
        print("\nGet your stream key from:")
        print("  YouTube Studio → Go Live → Stream")
        print("=" * 60)
        sys.exit(1)
    
    # Construct RTMP URL
    rtmp_url = f"rtmp://a.rtmp.youtube.com/live2/{stream_key}"
    
    print("=" * 60)
    print("YouTube Live Streamer")
    print("=" * 60)
    print(f"Stream Key: {stream_key}")
    print(f"RTMP URL: {rtmp_url}")
    print("=" * 60)
    
    # Pre-test RTMP connection
    print("\nTesting RTMP server connectivity...")
    test_result = YouTubeStreamer.test_rtmp_server(rtmp_url, timeout=5.0)
    
    if not test_result["reachable"]:
        print(f"\n❌ ERROR: {test_result['message']}")
        print("\nPossible issues:")
        print("  1. No internet connection")
        print("  2. Firewall blocking RTMP (port 1935)")
        print("  3. YouTube RTMP servers are down")
        print("  4. Invalid stream key")
        sys.exit(1)
    
    print(f"✓ RTMP server is reachable: {test_result['host']}:{test_result['port']}")
    print("\n⚠️  IMPORTANT:")
    print("  1. Make sure your YouTube Live stream is set up")
    print("  2. Go to YouTube Studio → Go Live → Stream")
    print("  3. Your stream will appear after 10-30 seconds")
    print("\nStarting stream in 3 seconds...")
    print("Press ESC in preview window to stop streaming.\n")
    
    time.sleep(3)
    
    # Initialize camera and streamer
    cam = Camera()
    streamer = YouTubeStreamer(
        rtmp_url=rtmp_url,
        fps=30,
        encoder="libx264",
        bitrate=4_000_000  # 4 Mbps - adjust based on your upload speed
    )
    
    frame_count = 0
    start_time = time.time()
    
    try:
        frame_generator = cam.frames()
        
        while True:
            # Get frame from camera
            frame, ts = next(frame_generator)
            
            # Send to YouTube
            try:
                streamer.send_frames(iter([(frame, ts)]))
                frame_count += 1
                
                # Print stats every second with connection status
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_actual = frame_count / elapsed if elapsed > 0 else 0
                    status = streamer.check_connection()
                    
                    # Show detailed status
                    conn_indicator = "✓" if status['actively_streaming'] else "⚠"
                    print(f"[stream] {conn_indicator} {frame_count} frames | "
                          f"{elapsed:.1f}s | {fps_actual:.1f} fps | "
                          f"Active: {status['actively_streaming']}")
                    
                    # Warning if connection seems stale
                    if status['last_send_seconds_ago'] and status['last_send_seconds_ago'] > 3.0:
                        print(f"[stream] ⚠  Warning: Last successful send was {status['last_send_seconds_ago']:.1f}s ago")
                
            except Exception as e:
                print(f"[stream] Error streaming frame: {e}")
                print(f"[stream] Connection status: {streamer.check_connection()}")
                import traceback
                traceback.print_exc()
                break
            
            # Preview window
            preview_frame = frame[:, :, ::-1].copy()  # RGB to BGR
            
            # Add stream status overlay
            cv2.putText(preview_frame, "LIVE ON YOUTUBE", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(preview_frame, f"Frame: {frame_count}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(preview_frame, f"Time: {time.time() - start_time:.1f}s", 
                       (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("YouTube Live Stream Preview", preview_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("\n[stream] User stopped stream")
                break
    
    except KeyboardInterrupt:
        print("\n[stream] Interrupted by user")
    except StopIteration:
        print("\n[stream] Camera stopped")
    except Exception as e:
        print(f"\n[stream] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        cam.close()
        streamer.close()
        cv2.destroyAllWindows()
        
        elapsed = time.time() - start_time
        print(f"\n{'=' * 60}")
        print(f"Stream Summary:")
        print(f"  Total frames sent: {frame_count}")
        print(f"  Duration: {elapsed:.1f}s")
        print(f"  Average FPS: {frame_count / elapsed if elapsed > 0 else 0:.1f}")
        print(f"{'=' * 60}")