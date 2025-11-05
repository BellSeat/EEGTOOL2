import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional

class EEGMatplotDisplay:
    """Real-time EEG multi-channel visualization with configurable projections."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.channels = self.config.get("channels", list(range(16)))
        self.sampling_rate = self.config.get("sampling_rate", 250)
        self.window_size = self.config.get("window_size", 5)
        
        self.fig = None
        self.axes = []
        self.displayed = False
        self.current_projection = "rectilinear"
        self.projection_list = ["rectilinear", "polar"]
        self.eeg_data = None
        self.time_stamps = None
    
    def _calculate_polar_coordinates(self, x: np.ndarray, y: np.ndarray, r_pad: float = 0.5,spiral: bool = False) -> tuple:
        """Convert cartesian to polar coordinates."""
        t = np.asarray(x)
        y = np.asarray(y)

        T = float(t[-1] - t[0]) if t.size > 1 else 1.0
        theta = 2 * np.pi * (t - t[0]) / T
        theta = np.mod(theta, 2 * np.pi)

        y0 = y - np.nanmean(y)
        y_min, y_max = np.nanmin(y0), np.nanmax(y0)
        span = (y_max - y_min) if (y_max > y_min) else 1.0
        r = (y0 - y_min) / span
        r = r_pad + (1 - r_pad) * r
        if spiral:
            r += (theta / (2 * np.pi)) * r_pad
        return r, theta
    
    def _plot_channel(self, ax, channel_data: np.ndarray, time_stamps: np.ndarray, 
                     channel_idx: int, projection: str):
        """Plot a single channel on given axes."""
        ax.cla()
        
        if projection == "polar":
            r, theta = self._calculate_polar_coordinates(time_stamps, channel_data)
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.set_ylim(0,max(1.0, np.nanmax(r)))
            ax.plot(theta, r, marker='.')
            # set ax to point instead of line   

            ax.set_title(f'CH{channel_idx + 1} Polar', fontsize=10)
        else:
            ax.plot(time_stamps, channel_data)
            ax.set_title(f'CH{channel_idx + 1}', fontsize=10)
            ax.set_xlabel('Time (s)', fontsize=8)
            ax.set_ylabel('Amplitude', fontsize=8)
            ax.tick_params(labelsize=8)
    
    def prepare_all_channel_plots(self, eeg_data: np.ndarray, time_stamps: np.ndarray, 
                                  plt_projection: str = "rectilinear"):
        """Initialize all channel plots in a single figure."""
        if plt_projection not in self.projection_list:
            raise ValueError(f"Projection '{plt_projection}' not supported. "
                           f"Choose from: {self.projection_list}")
        
        

        num_channels = eeg_data.shape[0]
        
        # Close previous figure if exists
        if self.fig is not None:
            plt.close(self.fig)
        
        # Calculate grid dimensions (4x4 for 16 channels)
        n_rows = int(np.ceil(np.sqrt(num_channels)))
        n_cols = int(np.ceil(num_channels / n_rows))
        
        # Create single figure with all subplots
        self.fig = plt.figure(figsize=(4 * n_cols, 3 * n_rows), 
                             num='EEG Multi-Channel Display')
        self.axes = []
        
        for i in range(num_channels):
            ax = self.fig.add_subplot(n_rows, n_cols, i + 1, projection=plt_projection)
            self._plot_channel(ax, eeg_data[i, :], time_stamps, i, plt_projection)
            self.axes.append(ax)
        
        self.current_projection = plt_projection
        self.eeg_data = eeg_data
        self.time_stamps = time_stamps
        plt.tight_layout()
        
        return True
    
    def turn_on_plots(self):
        """Enable interactive plot mode."""
        if self.fig is None or not self.axes:
            raise RuntimeError("No plots prepared. Call prepare_all_channel_plots first.")
        plt.ion()
        self.fig.show()
        self.displayed = True
        return True
    
    def turn_off_plots(self):
        """Disable interactive plot mode."""
        if self.fig is None or not self.axes:
            print("Warning: No plots to turn off.")
            return False
        plt.ioff()
        self.displayed = False
        return True
    
    def close_plots(self):
        """Close all plots and clean up resources."""
        if self.fig is None:
            print("Warning: No plots to close.")
            return False
        plt.close(self.fig)
        self.fig = None
        self.axes = []
        self.displayed = False
        return True
    
    def update_living_figure_projection(self, next_projection: str = 'polar'):
        """Switch projection type for all channels."""
        if self.fig is None or not self.axes:
            raise RuntimeError("No plots to update. Call prepare_all_channel_plots first.")
        
        if next_projection not in self.projection_list:
            raise ValueError(f"Projection '{next_projection}' not supported. "
                           f"Choose from: {self.projection_list}")
        
        if self.current_projection == next_projection:
            print("Info: Already using requested projection. No update needed.")
            return True

        if self.eeg_data is None or self.time_stamps is None:
            raise RuntimeError("No data available. Update data first.")
        
        # Recreate figure with new projection
        was_displayed = self.displayed
        if was_displayed:
            self.turn_off_plots()
        
        self.prepare_all_channel_plots(self.eeg_data, self.time_stamps, next_projection)
        
        if was_displayed:
            self.turn_on_plots()
        
        return True
    
    def drawing_plots_with_updating_data(self, eeg_data: np.ndarray, time_stamps: np.ndarray):
        """Update all channel plots with new data."""
        if self.fig is None or not self.axes:
            raise RuntimeError("No plots to update. Call prepare_all_channel_plots first.")
        
        # Save data for projection switching
        self.eeg_data = eeg_data
        self.time_stamps = time_stamps
        
        try:
            num_channels = min(eeg_data.shape[0], len(self.axes))
            
            for i in range(num_channels):
                self._plot_channel(self.axes[i], eeg_data[i, :], time_stamps, 
                                 i, self.current_projection)
            
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.01)
            
        except Exception as e:
            print(f"Error updating plots: {e}")
            return False
        
        return True


# Example usage
if __name__ == "__main__":
    import EEG_direct_connect as eegdc
    import time

    continue_test = True  # Set to True to run the second test case
    # Test 1 simulate real-time EEG data acquisition and plotting
    eegdata = np.random.randn(16, 512)  # 16 channels, 512 samples
    print(f"EEG data max value: {np.max(eegdata)}, min value: {np.min(eegdata)}")
    timestamps = np.linspace(0, 2, 512)  # 2 seconds of data
    
    device = eegdc.EEGDevice()
    eegPlt = EEGMatplotDisplay()
    # case 1:
    eegPlt.prepare_all_channel_plots(eegdata, timestamps, plt_projection="rectilinear")
    eegPlt.turn_on_plots()
    for i in range(20):
        eegdata = np.random.randn(16, 512)  # New random data
        eegPlt.drawing_plots_with_updating_data(eegdata, timestamps)
    #  switch projection
    eegPlt.update_living_figure_projection(next_projection="polar")
    for i in range(20):
        eegdata = np.random.randn(16, 512)
        eegPlt.drawing_plots_with_updating_data(eegdata, timestamps)
        
    eegPlt.close_plots()

    if not continue_test:
        # end for now
        exit(0)


    # case 2
    try:
        # Connect to device and start streaming
        device.wifi_connect()
        device.start_stream()
        time.sleep(2)  # Wait for data accumulation
        
        # Get initial data and prepare plots
        eeg_data, time_stamps = device.get_data(512)
        eegPlt.prepare_all_channel_plots(eeg_data, time_stamps, plt_projection="rectilinear")
        eegPlt.turn_on_plots()
        
        # Update plots in real-time
        for i in range(20):
            eeg_data, time_stamps = device.get_data(512)
            starting_time = time_stamps[0]
            
            eegPlt.drawing_plots_with_updating_data(eeg_data, time_stamps)
            
            # Switch to polar projection halfway through
            if i == 10:
                eegPlt.update_living_figure_projection(next_projection="polar")
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error during EEG data acquisition and plotting: {e}")
    finally:
        eegPlt.close_plots()
        if 'device' in locals():
            device.stop_stream()
            device.close()
    
    