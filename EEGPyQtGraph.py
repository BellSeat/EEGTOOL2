import sys, time, threading, queue
import numpy as np
from PySide6 import QtWidgets, QtCore
import pyqtgraph as pg

N_CH = 16
FS = 250
WIN_SEC = 5
N_SAMPLES = FS * WIN_SEC

class EEGWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG - PyQtGraph Realtime (Time/POLAR toggle: P)")
        self.resize(1400, 900)

        # UI: grid 4x4
        central = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(central)
        self.setCentralWidget(central)

        pg.setConfigOptions(antialias=True)
        self.plots, self.curves = [], []

        # data buffers
        self.t = np.linspace(0, WIN_SEC, N_SAMPLES, endpoint=False)
        self.buf = np.zeros((N_CH, N_SAMPLES), dtype=np.float32)
        self.polar_mode = False

        # make subplots
        for i in range(N_CH):
            pw = pg.PlotWidget()
            pw.setLabel("bottom", "Time", units="s")
            pw.setLabel("left", "Amp")
            pw.setYRange(-3, 3)
            pw.setXRange(self.t[0], self.t[-1])
            pw.showGrid(x=True, y=True, alpha=0.25)
            pw.setAspectLocked(False)  # Allow different scaling for time series
            curve = pw.plot(pen=pg.mkPen(width=1))
            r, c = divmod(i, 4)
            grid.addWidget(pw, r, c)
            self.plots.append(pw)
            self.curves.append(curve)

        # timer for GUI update
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_once)
        self.timer.start(int(1000/60))  # target ~60 FPS

        # data queue
        self.q = queue.Queue(maxsize=32)

    def keyPressEvent(self, ev):
        """Handle key press events."""
        if ev.key() == QtCore.Qt.Key_P:
            self.polar_mode = not self.polar_mode
            if self.polar_mode:
                for pw in self.plots:
                    pw.setLabel("bottom", "X (r*cos(θ))")
                    pw.setLabel("left", "Y (r*sin(θ))")
                    pw.setAspectLocked(True)  # Lock aspect for circular polar plot
                    pw.setXRange(-1.2, 1.2)
                    pw.setYRange(-1.2, 1.2)
            else:
                for pw in self.plots:
                    pw.setLabel("bottom", "Time", units="s")
                    pw.setLabel("left", "Amp")
                    pw.setAspectLocked(False)
                    pw.setXRange(self.t[0], self.t[-1])
                    pw.setYRange(-3, 3)

    def _to_polar(self, t, y, r_pad=0.05, spiral=True):
        """Convert time series to polar coordinates (theta, r).
        
        Returns Cartesian coordinates (x, y) = (r*cos(theta), r*sin(theta))
        for proper polar plotting.
        """
        # Map time to angle [0, 2π)
        T = t[-1] - t[0] if t.size > 1 else 1.0
        theta = 2 * np.pi * (t - t[0]) / T
        theta = np.mod(theta, 2 * np.pi)
        
        # Normalize amplitude to radius [r_pad, 1]
        y0 = y - np.nanmean(y)
        mn, mx = np.nanmin(y0), np.nanmax(y0)
        span = (mx - mn) if (mx > mn) else 1.0
        r = (y0 - mn) / span
        r = r_pad + (1.0 - r_pad) * r
        
        if spiral:
            r = r + 0.25 * (t - t[0]) / T
        
        # Convert to Cartesian for proper polar display
        x = r * np.cos(theta)
        y_cart = r * np.sin(theta)
        
        return x, y_cart

    def update_once(self):
        """Update plots with new data from queue."""
        # Drain queue; keep newest batch only
        got = False
        chx = None
        while True:
            try:
                chx = self.q.get_nowait()   # shape (N_CH, batch)
                got = True
            except queue.Empty:
                break
        if not got:
            return

        # Roll buffer and append newest
        batch = chx.shape[1]
        self.buf = np.roll(self.buf, -batch, axis=1)
        self.buf[:, -batch:] = chx

        # Plot
        if not self.polar_mode:
            # Time series
            for i, curve in enumerate(self.curves):
                curve.setData(self.t, self.buf[i])
        else:
            # Polar: convert to Cartesian (x, y) for circular display
            for i, curve in enumerate(self.curves):
                x, y = self._to_polar(self.t, self.buf[i])
                curve.setData(x, y)

def data_thread(q: queue.Queue, stop: threading.Event):
    """Simulate EEG data generation."""
    dt = 1.0 / FS
    freq = np.linspace(6, 18, N_CH)  # Different frequency per channel
    phase = np.zeros(N_CH)
    batch = 25  # Samples per frame
    rng = np.random.default_rng(123)

    while not stop.is_set():
        t = np.arange(batch) * dt
        # Simulate: sine + noise
        sig = np.sin(2 * np.pi * freq[:, None] * t + phase[:, None])
        sig += 0.25 * rng.standard_normal((N_CH, batch))
        phase += 2 * np.pi * freq * dt * batch
        phase %= 2 * np.pi
        try:
            q.put(sig.astype(np.float32), timeout=0.1)
        except queue.Full:
            pass
        time.sleep(batch / FS * 0.5)  # Control production rate

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = EEGWindow()
    win.show()

    stop = threading.Event()
    th = threading.Thread(target=data_thread, args=(win.q, stop), daemon=True)
    th.start()

    try:
        sys.exit(app.exec())
    finally:
        stop.set()
        th.join(timeout=1)