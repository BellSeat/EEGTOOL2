from typing import Literal

def get_streamer(kind: Literal["youtube", "lsl"], **kwargs):
    if kind == "youtube":
        from .youtube_live import YouTubeStreamer
        return YouTubeStreamer(**kwargs)
    elif kind == "lsl":
        from .lsl import LSLStreamer
        return LSLStreamer(**kwargs)
    else:
        raise ValueError(f"unknown stream kind: {kind}")
