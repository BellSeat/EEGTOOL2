import multiprocessing as mp
import queue
import signal
import sys
from typing import Optional

import numpy as np

try:
    from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore
    QT_BACKEND = "PyQt5"
except ImportError:  # pragma: no cover - fallback branch
    try:
        from PySide6 import QtCore, QtGui, QtWidgets  # type: ignore
        QT_BACKEND = "PySide6"
    except ImportError:  # pragma: no cover - Qt unavailable
        QtCore = None  # type: ignore
        QtGui = None  # type: ignore
        QtWidgets = None  # type: ignore
        QT_BACKEND = ""


HAS_QT = QtWidgets is not None


def _numpy_to_qimage(frame: np.ndarray) -> QtGui.QImage:
    height, width, channels = frame.shape
    bytes_per_line = channels * width
    image = QtGui.QImage(
        frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888
    )
    return image.copy()


def _preview_loop(frame_queue: mp.Queue, window_title: str) -> None:
    if not HAS_QT:
        return

    signal.signal(signal.SIGINT, signal.SIG_IGN)

    app = QtWidgets.QApplication(sys.argv or ["qt_preview"])
    label = QtWidgets.QLabel()
    label.setWindowTitle(window_title)
    label.setAlignment(QtCore.Qt.AlignCenter)
    label.resize(640, 480)
    label.show()

    timer = QtCore.QTimer()

    def pump_queue() -> None:
        try:
            while True:
                frame = frame_queue.get_nowait()
                if frame is None:
                    timer.stop()
                    app.quit()
                    return
                image = _numpy_to_qimage(frame)
                pixmap = QtGui.QPixmap.fromImage(image)
                label.setPixmap(pixmap)
                label.resize(pixmap.size())
        except queue.Empty:
            pass

    timer.timeout.connect(pump_queue)  # type: ignore[arg-type]
    timer.start(30)
    exec_fn = getattr(app, "exec", None) or getattr(app, "exec_", None)
    if exec_fn is not None:
        exec_fn()


class QtPreview:
    def __init__(
        self,
        window_title: str = "Camera Preview",
        max_queue_size: int = 4,
    ) -> None:
        self.window_title = window_title
        self._ctx = mp.get_context("spawn")
        self._queue: Optional[mp.Queue] = None
        self._process: Optional[mp.Process] = None
        self._max_queue_size = max_queue_size
        self._enabled = HAS_QT

    @property
    def is_available(self) -> bool:
        return self._enabled

    def start(self) -> bool:
        if not self._enabled:
            return False
        if self._process and self._process.is_alive():
            return True
        self._queue = self._ctx.Queue(self._max_queue_size)
        self._process = self._ctx.Process(
            target=_preview_loop, args=(self._queue, self.window_title)
        )
        self._process.daemon = True
        self._process.start()
        return True

    def show(self, frame: np.ndarray) -> None:
        if not self._enabled or self._queue is None or frame is None:
            return
        try:
            rgb_frame = frame[:, :, ::-1].copy()
            try:
                self._queue.put_nowait(rgb_frame)
            except queue.Full:
                try:
                    _ = self._queue.get_nowait()
                except queue.Empty:
                    pass
                self._queue.put_nowait(rgb_frame)
        except Exception:
            pass

    def stop(self) -> None:
        if not self._enabled:
            return
        if self._queue is not None:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
        if self._process is not None:
            self._process.join(timeout=2)
            if self._process.is_alive():
                self._process.terminate()
        self._queue = None
        self._process = None
