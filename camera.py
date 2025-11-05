import av, time, json, platform
from pathlib import Path
from typing import Dict, Generator, Tuple, Optional

CONFIG_PATH = Path(__file__).with_name("config.json")

def _os_key() -> str:
    current_os = platform.system().lower()
    if "windows" in current_os:
        return "windows"
    elif "darwin" in current_os:
        return "macos"
    elif "linux" in current_os:
        return "linux"
    else:
        raise ValueError(f"Unsupported OS: {current_os}")
    
def load_config(path: Path = CONFIG_PATH) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
    
class Camera:
    """
    comma camera class to handle camera operations
    - load configuration based on OS
    - use pyav for decode + errno 35 handling
    - pocess (ndarray_rgb, pts_sconds) frame
    """
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or load_config()
        self.os_key = self.config[_os_key()]
        self.format = self.os_key["format"]
        self.source = self.os_key["source"]
        # reformat options to str
        self.options = {k: str(v) for k, v in self.os_key.get("options", {}).items()}
        self.container = None
        self.stream = None

    def _cold_open(self):
        candidates = [self.source]
        if _os_key == "macos":
            if self.source != "0":
                candidates.append("0")
            if self.source != "0:none":
                candidates.append("0:none")
        last = None
        for src in candidates:
            try:
                print(f"[camera] try open source={src} format={self.format} options={self.options}")
                c = av.open(src, format=self.format, options=self.options)
                s = c.streams.video[0]
                s.thread_type = "AUTO"
                self.container, self.stream = c, s
                print(f"[camera] opened source={src} successfully")
                return
            except Exception as e:
                last = e
                print(f"[camera] failed to open source={src}: {e}")
        raise last or RuntimeError("Failed to open any camera source")
    
    def open(self):
        if self.container is None:
            self._cold_open()

    def frames(self) -> Generator[Tuple["np.ndarray", float], None, None]:
        import numpy as np
        self.open()
        decoder = self.container.decode(video=0)
        while True:
            try:
                frame = next(decoder)
            except StopIteration:
                decoder = self.container.decode(video=0)
                continue
            except Exception as e:
                if getattr(e, "errno", None) == 35:
                    time.sleep(0.01)
                    continue
                raise
            ts = frame.time if frame.time is not None else time.time()
            arr = frame.to_ndarray(format="rgb24")
            yield arr, ts
    
    def close(self):
        try:
            if self.container:
                self.container.close()
                self.container = None
                self.stream = None
        except Exception:
            pass
    def get_with(self) -> int:
        self.open()
        return self.stream.width if self.stream else 0
    def get_height(self) -> int:
        self.open()
        return self.stream.height if self.stream else 0
    def get_piexl_format(self) -> str:
        self.open()
        return self.stream.pix_fmt if self.stream else ""
    
if __name__ == "__main__":
    import cv2
    cam = Camera()
    cam.open()
    print("Camera opened. Press ESC to quit.")
    try:
        for i, (frame, ts) in enumerate(cam.frames()):
            cv2.imshow("Camera Preview", frame[:, :, ::-1])  # RGB to BGR
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                print("Exiting.")
                break
            if i % 30 == 0:
                print(f"Frame {i} @ {ts:.2f}s")
    except KeyboardInterrupt:
        print("Exiting.")
    finally:
        cam.close()
        cv2.destroyAllWindows()