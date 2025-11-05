# pip install av opencv-python
import av
import cv2
import json
import platform
import sys
import time
from pathlib import Path

CONFIG_PATH = Path(__file__).with_name("config.json")

def load_config(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {Path}")
    with path.open("r", encoding = "utf-8") as f:
        return json.load(f)
    
def get_os_key() -> str:
    current_os = platform.system().lower()
    if "windows" in current_os:
        return "windows"
    elif "darwin" in current_os:
        return "macos"
    elif "linux" in current_os:
        return "linux"
    else:
        return "unknown"
    
def load_camera_config(os_cfg: dict): 
    fmt = os_cfg.get("format")
    source = os_cfg.get("source")
    options = os_cfg.get("options", {})
    
    if not fmt or not source:
        raise   ValueError("Invalid configuration: 'format' and 'source' are required.")
    return av.open(source, format=fmt, options=options)
    
def main():
    cfg = load_config(CONFIG_PATH)
    os_key = get_os_key()
    common = cfg.get("common", {})
    os_cfg = cfg.get(os_key, {})

    # check if os_cfg is empty
    if not os_cfg:
        raise ValueError(f"No configuration found for OS: {os_key}")
    
    preview = bool(common.get("preview", False))
    log_every_n = int(common.get("log_every_n", 60))

    print(f"📦 OS config = {os_key} -> {os_cfg['format']} / {os_cfg['source']}")
    if preview and cv2 is None:
        print("⚠️ Preview enabled but OpenCV is not installed. Disabling preview.")
        preview = False
    
    try: 
        container = load_camera_config(os_cfg)
    except Exception as e:
        print(f"❌ Failed to open camera source: {e}")
        print("suggestion:")
        if os_key == "windows":
            print(" - Ensure that the DirectShow device name is correct.")
        elif os_key == "macos":
            print(" - Ensure that the AVFoundation device index is correct.")
        else:
            print(" - Ensure that the V4L2 device path is correct.")
        sys.exit(1)

    print("✅ Camera source opened successfully.")
    video_stream = container.streams.video[0]
    video_stream.thread_type = "AUTO"
    decoder = container.decode(video=0)
    print(f"📷 Video stream: {video_stream.width}x{video_stream.height} @ {video_stream.average_rate} fps")

    print("✅ Camera opened. Press ESC to quit." if preview else "✅ Camera opened. Ctrl+C to quit.")
    frame_count = 0

    try:
        while True:
            try:
                frames = next(decoder)
            except OSError as e:
                if getattr(e, 'errno', None) == 35:
                    time.sleep(0.01)
                    continue
                else: 
                    raise
            except StopIteration:
                decoder = container.decode(video=0)
                continue

            
            ts = frames.time if frames.time is not None else time.time()
            arr = frames.to_ndarray(format="bgr24")
            if preview and cv2:
                cv2.imshow("Camera Preview", arr)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    print("👋 Exiting program.")
                    break
            frame_count += 1
            if frame_count % log_every_n == 0:
                print(f"🖼️ Frame {frame_count} @ {ts:.2f}s")
           
    except KeyboardInterrupt:
        print("👋 Exiting program.")
        pass
    finally:
        try: 
            container.close()
        except Exception:
            pass
        if preview and cv2:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        print("✅ Camera closed.")

if __name__ == "__main__":
    main()