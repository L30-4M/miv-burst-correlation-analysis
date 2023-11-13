__all__ = ["BurstsFilter", "CrossCorrelograms", "CoincidenceMatrix", "MeanCrossCorrelogram"]

from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union

import os
import pathlib
from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import neo

from miv.core.operator import OperatorMixin
from miv.core.wrapper import wrap_cacher
from miv.core.datatype import Spikestamps
from miv.visualization.event import plot_spiketrain_raster
from miv.statistics.spiketrain_statistics import firing_rates

from miv.statistics.burst import burst_array

# TODO: add aditional check for burst rate to be inbetween 3 and 25 spikes/s (maybe?)
# TODO: Minor refactoring so code can work with arbitrary bins etc. 
#       (Probably make C_XY return the bin number, then other code can use it.) 
#       (Alternatively make CI matrix and Mean matrix just use array dimensions.)
# TODO: Rename all functions and file names to be more clear. Much here is temporary.
# TODO: Implement visualizations from pages 44, 47
# TODO: Decide if BurstsFilter should return multiple things or just one. Likely best to keep singular until *kwargs is implemented.
@dataclass
class BurstsFilter(OperatorMixin):
    """
    Filter out bursts from spike trains

    Example usage code::
        spike_detection: Operator = ThresholdCutoff(cutoff=5.0, dead_time=0.003)
        burst_filter: Operator = BurstsFilter()
        
        data >> spike_detection >> burst_filter
        Pipeline(burst_filter).run()

    Parameters
    ----------
    tag : str, optional
        Operator tag.
    min_isi : float, optional
        Minimum inter-spike interval for burst detection.
    min_len : int, optional
        Minimum burst length in number of spikes for burst detection.

    Attributes
    ----------
    burst_lens : list of int, optional
        List to store the burst lengths for each channel.
    burst_durations : list of float, optional
        List to store the burst durations for each channel.

    Returns
    ----------
    spiketrain, burst lengths, burst durations
    """

    tag: str = "bursts filter"

    min_isi: float = 0.1
    min_len: int = 10

    def __post_init__(self):
        super().__init__()

    @wrap_cacher(cache_tag="burst_filter")
    def __call__(self, spiketrains: Spikestamps):
        import copy
        self.allspikestamps = copy.deepcopy(spiketrains)
        burst_lens = []
        burst_durations = []
        chnls = spiketrains.number_of_channels
        for i in range(chnls):
            # TODO: REPLACE WITH Q = burst_array(spiketrains[i], self.min_isi, self.min_len)
            # BELOW IS SUPER TEMP WHILE BURST ARRAY IS NOT INCLUDED
            Q = burst_array(spiketrains[i], self.min_isi, self.min_len)
            # ABOVE IS SUPER TEMP WHILE BURST ARRAY IS NOT INCLUDED
            spike = np.array(spiketrains[i])
            if np.sum(Q) != 0:
                spiketrains.data[i] = np.concatenate(
                    [spike[start: end + 1] for start, end in Q]
                )
                start_time = spike[Q[:, 0]]
                end_time = spike[Q[:, 1]]

                burst_lens.append(Q[:, 1] - Q[:, 0] + 1)
                burst_durations.append(end_time - start_time)
            else:
                spiketrains.data[i] = []

        return spiketrains, np.asarray(burst_lens, dtype=object), np.asarray(burst_durations, dtype=object)

    def plot_bursttrain(
        self,
        outputs,
        show: bool = False,
        save_path: Optional[pathlib.Path] = None,
    ):
        spikestamps, burst_lens, burst_durations = outputs

        t0 = spikestamps.get_first_spikestamp()
        tf = spikestamps.get_last_spikestamp()

        # Create a single plot with adjusted x-axis limits
        fig, ax = plot_spiketrain_raster(spikestamps, t0, tf)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("Channel")
        if save_path is not None:
            plt.savefig(os.path.join(save_path, "burst_raster.png"))

        if show:
            plt.show()
        
        return ax
    
    def plot_overlay(
        self,
        outputs,
        show: bool = False,
        save_path: Optional[pathlib.Path] = None,
    ) -> plt.Axes:
        """
        Plot spike train in raster
        """
        spikestamps = self.allspikestamps
        burststamps, burst_lens, burst_durations = outputs

        t0 = spikestamps.get_first_spikestamp()
        tf = spikestamps.get_last_spikestamp()

        # TODO: REFACTOR. Make single plot, and change xlim
        term = 10
        n_terms = int(np.ceil((tf - t0) / term))
        if n_terms == 0:
            # TODO: Warning message
            return None
        for idx in range(n_terms):
            t_start = idx * term + t0
            t_stop = min((idx + 1) * term + t0, tf)
            fig, ax = plt.subplots(figsize=(16, 6))
            spikes = spikestamps.get_view(t_start, t_stop)
            bursts = burststamps.get_view(t_start, t_stop)
            # Plot spikes as black dots
            ax.eventplot(spikes)
            ax.eventplot(bursts, colors="r")
            ax.set_xlim(t_start, t_stop)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Channel")
            ax.set_title(f"Raster plot (from {t_start} to {t_stop})")
            ax.set_xlim(t_start, t_stop)
            if save_path is not None:
                plt.savefig(os.path.join(save_path, f"overlay_raster_{idx:03d}.png"))
            if not show:
                plt.close("all")
        if show:
            plt.show()
            plt.close("all")
        return ax
    
    def plot_firing_rate_histogram(self, outputs, show=False, save_path=None):
        """Plot firing rate histogram"""
        
        spikestamps, burst_lens, burst_durations = outputs
        rates = firing_rates(spikestamps)["rates"]
        hist, bins = np.histogram(rates, bins=20)
        logbins = np.logspace(
            np.log10(max(bins[0], 1e-3)), np.log10(bins[-1]), len(bins)
        )
        fig = plt.figure()
        ax = plt.gca()
        ax.hist(rates, bins=logbins)
        ax.axvline(
            np.mean(rates),
            color="r",
            linestyle="dashed",
            linewidth=1,
            label=f"Mean {np.mean(rates):.2f} Hz",
        )
        ax.set_xscale("log")
        xlim = ax.get_xlim()
        ax.set_xlabel("Firing rate (Hz) (log-scale)")
        ax.set_ylabel("Count")
        ax.set_xlim([min(xlim[0], 1e-1), max(1e2, xlim[1])])
        ax.legend()
        if save_path is not None:
            fig.savefig(os.path.join(f"{save_path}", "firing_rate_histogram.png"))
        if show:
            plt.show()

        return ax
    
    # Visualization (probably?) based on what is shown on page 45
    def plot_length_vs_duration(
        self,
        outputs,
        show: bool = False,
        save_path: Optional[pathlib.Path] = None,
    ) -> plt.Axes:
        
        spikestamps, burst_lens, burst_durations = outputs
        #burst_lens = self.burst_lens
        #burst_durations = self.burst_durations
        fig, ax = plt.subplots()
        for i in range(len(burst_lens)):
            ax.scatter(burst_lens[i], burst_durations[i])
        ax.set_xlabel('Burst length')
        ax.set_ylabel('Burst duration')
        
        if save_path is not None:
            plt.savefig(os.path.join(save_path, "burstlen_vs_burstdur.png"))
        
        if show:
            plt.show()
        
        return ax

    def plot_burst_length_histogram(
        self,
        outputs,
        show: bool = False,
        save_path: Optional[pathlib.Path] = None,
    ) -> plt.Axes:
        
        spikestamps, burst_lens, burst_durations = outputs
        fig, ax = plt.subplots()
        Histo = np.concatenate(burst_lens, axis=None)
        # Another magic number that should TODO: be given logic
        plt.hist(Histo, bins=20, edgecolor='black')
        ax.set_xlabel("Burst length")
        ax.set_ylabel("Count")
        if save_path is not None:
           plt.savefig(os.path.join(save_path, "burst_length_histogram.png"))
        
        if show:
            plt.show()
    
        return ax

    def plot_burst_duration_histogram(
        self,
        outputs,
        show: bool = False,
        save_path: Optional[pathlib.Path] = None,
    ) -> plt.Axes:   
        
        spikestamps, burst_lens, burst_durations = outputs
        dur = np.concatenate(burst_durations, axis=None)
        hist, bins = np.histogram(dur, bins=20)
        logbins = np.logspace(
            np.log10(max(bins[0], 1e-3)), np.log10(bins[-1]), len(bins)
        )
        fig = plt.figure()
        ax = plt.gca()
        ax.hist(dur, bins=logbins, edgecolor='black')
        ax.axvline(
            np.mean(dur),
            color="r",
            linestyle="dashed",
            linewidth=1,
            label=f"Mean {np.mean(dur):.2f} s",
        )
        ax.set_xscale("log")
        xlim = ax.get_xlim()
        ax.set_xlabel("Burst Durations (s) (log-scale)")
        ax.set_ylabel("Count")
        # TODO: set_xlim should be changed possibly?
        ax.set_xlim([min(xlim[0], 1e-1), max(1e2, xlim[1])])
        ax.legend()
        if save_path is not None:
            fig.savefig(os.path.join(f"{save_path}", "burst_duration_histogram.png"))
        if show:
            plt.show()

        return ax

    def plot_burst_rate_histogram(
        self,
        outputs,
        show: bool = False,
        save_path: Optional[pathlib.Path] = None,
    ) -> plt.Axes:
          
        spikestamps, burst_lens, burst_durations = outputs
        temp = burst_lens/burst_durations
        rates = np.concatenate(temp, axis=None)
        hist, bins = np.histogram(rates, bins=20)
        logbins = np.logspace(
            np.log10(max(bins[0], 1e-3)), np.log10(bins[-1]), len(bins)
        )
        fig = plt.figure()
        ax = plt.gca()
        ax.hist(rates, bins=logbins, edgecolor='black')
        ax.axvline(
            np.mean(rates),
            color="r",
            linestyle="dashed",
            linewidth=1,
            label=f"Mean {np.mean(rates):.2f} Hz",
        )
        ax.set_xscale("log")
        xlim = ax.get_xlim()
        ax.set_xlabel("Burst Firing rate (Hz) (log-scale)")
        ax.set_ylabel("Count")
        ax.set_xlim([min(xlim[0], 1e-1), max(1e2, xlim[1])])
        ax.legend()
        if save_path is not None:
            fig.savefig(os.path.join(f"{save_path}", "burst_rate_histogram.png"))
        if show:
            plt.show()

        return ax
    
    def plot_burst_rate_histogram_alt(
        self,
        outputs,
        show: bool = False,
        save_path: Optional[pathlib.Path] = None,
    ) -> plt.Axes:
        
        spikestamps, burst_lens, burst_durations = outputs
        fig, ax = plt.subplots()
        rates = burst_lens/burst_durations
        Histo = np.concatenate(rates, axis=None)
        plt.hist(Histo, bins=20, edgecolor='black')
        ax.set_xlabel("Burst Firing rate (Hz)")
        ax.set_ylabel("Count")
        # Another magic number that should TODO: be given logic
        if save_path is not None:
           plt.savefig(os.path.join(save_path, "burst_rate_histogram_alt.png"))
        
        if show:
            plt.show()
    
        return ax

@dataclass
class CrossCorrelograms(OperatorMixin):
    """
    Calculate cross-correlograms between two spike trains

    Example usage code::
        spike_detection: Operator = ThresholdCutoff(cutoff=5.0, dead_time=0.003)
        burst_filter: Operator = BurstsFilter()
        cross_correlogram: Operator = CrossCorrelograms()
        data >> spike_detection >> burst_filter >> cross_correlogram
        spike_detection >> cross_correlogram
        Pipeline(cross_correlogram).run()

    Parameters
    ----------
    tag : str
        Operator tag
    keep_autocorr : bool
        Whether to keep autocorrelation or not
    deltaT : float
        Time window for cross-correlograms
    binsize : float
        Bin size for binning the spikes in the cross-correlograms
    
    Attributes
    ----------
    binN : int
        Number of bins in the correlation array
    """
    tag: str = "C XY matrix"

    keep_autocorr: bool = False
    deltaT: float = 0.15
    binsize: float = 0.01
    binN: float = round(2*(deltaT/binsize))

    def __post_init__(self):
        super().__init__()

    # TODO: If inputs are different than it should not use the same cache?
    @wrap_cacher(cache_tag="C_XY_matrix")
    def __call__(self, Xspikestamps, Yspikestamps):
        if isinstance(Xspikestamps, tuple):
            Xspikestamps = Xspikestamps[0]
        spiketimesMega = Xspikestamps.neo() # Extracts the spike times from Xspikestamps

        # Initializing constants 
        
        Xch = Xspikestamps.number_of_channels
        Ych = Yspikestamps.number_of_channels
        

        # Initializing empty return array
        C_XY = [[[0 for _ in range(self.binN)] for _ in range(Ych)] for _ in range(Xch)]

        # Main logic behind calculating cross correlogram
        for X in range(Xch) :

            Ys_binned = np.array([[0 for _ in range(Ych)] for _ in range(self.binN)])
            spikeQ = 0 # Counter for spikes

            for time in spiketimesMega[X].magnitude :
                # Binning the spike times in Yspikestamps within a time window and counting the number of spikes
                Ys_binned += Yspikestamps.binning(
                    bin_size=self.binsize, 
                    t_start=time-self.deltaT, 
                    t_end=time+self.deltaT, 
                    return_count=True
                ).data[:self.binN]
                spikeQ += 1

            if(spikeQ == 0) : spikeQ = 1 # Avoiding divide by zero error.
            Ys_norm = np.rot90(Ys_binned/(spikeQ * self.binsize), -1) # Normalizing the binned data
            if(not self.keep_autocorr) : Ys_norm[X] = [0 for _ in range(self.binN)] # Throwing out autocorrelation
            C_XY[X] = Ys_norm # Updating the C_XY matrix with the normalized data
        return C_XY

    # TODO: Implement visualization à la page 47.
    # Naive approach is display all sublots. Too many!
    # Probably best approach is to let user select which subplots to display. 
    # Alternatively only plot most "interesting" (?) channels. How to quantify?
    def plot_CXY(
        self,
        C_XY,
        Chx: int = 0,
        Chy: int = 0,
        show: bool = False,
        save_path: Optional[pathlib.Path] = None,
    ) : 
        fig, ax = plt.subplots()
        plt.plot(C_XY[Chx][Chy])
        ax.set_xlabel('Times (ms)')
        ax.set_ylabel('Spikes/s')

        if save_path is not None:
            plt.savefig(os.path.join(save_path, f"cross_correlogram{Chx}{Chy}.png"))
        
        if show:
            plt.show()

        return ax

@dataclass
class MeanCrossCorrelogram(OperatorMixin):
    """
    Returns the mean from a given cross-correlograms. 
    Assumes autocorrelation is not included.

    Example usage code::
        spike_detection: Operator = ThresholdCutoff(cutoff=5.0, dead_time=0.003)
        burst_filter: Operator = BurstsFilter()
        cross_correlogram: Operator = CrossCorrelograms()
        mean_cross_correlogram: Operator = MeanCrossCorrelogram()
        
        
        data >> spike_detection >> burst_filter >> cross_correlogram
        spike_detection >> cross_correlogram >> mean_cross_correlogram
        Pipeline(mean_cross_correlogram).run()

    Parameters
    ----------
    tag : str
        Operator tag
    """

    tag: str = "C X matrix"

    def __post_init__(self):
        super().__init__()

    # TODO: Decide if this function should rely on assumption that autocorrelation is not included
    def __call__(self, C_XY):
        return np.sum(C_XY, axis=1) / (len(C_XY)-1)
    
    # Visualization (probably?) based on what is shown on page 46
    def plot_CX3D(self,
        C_X,
        show: bool = False,
        save_path: Optional[pathlib.Path] = None,
    ) :
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        n_channels, n_indices = C_X.shape
        # Make logic based on dimensions of array/bins sizes.
        time_range = np.arange(-145, 150, 10)

        channel_indices, time_indices = np.meshgrid(time_range, np.arange(n_channels))
        # Magic numbers that should TODO: be given logic
        # Needed to be done so that interpolation was heavily prefered along one axis. 
        # Unsure how or why they work atm. When I get around to fixing I will figure out.
        ax.plot_surface(time_indices, channel_indices, C_X, rcount = 400, ccount = 100, cmap='viridis')
        ax.set_xlabel('Channel')
        ax.set_ylabel('Time (ms)')
        ax.set_zlabel('Spikes/s')
        if save_path is not None:
            plt.savefig(os.path.join(save_path, "c_x_mean_3d.png"))
        
        if show:
            plt.show()
    
        return ax
    
    # Personal visualization metric
    def plot_CXHeatmap(self,
        C_X,
        show: bool = False,
        save_path: Optional[pathlib.Path] = None,
    ) :
        fig, ax = plt.subplots()
        plt.imshow(C_X, cmap='hot', interpolation='none')
        colorbar = plt.colorbar()
        colorbar.set_label('Spikes/s')
        ax.set_xlabel('Times (ms)')
        ax.set_ylabel('Channel')
        num_channels, num_time_points = C_X.shape
        xticks_positions = np.linspace(0, num_time_points - 1, 5)
        xticks_labels = np.linspace(-150, 150, 5).astype(int)
        ax.set_xticks(xticks_positions)
        ax.set_xticklabels(xticks_labels)

        if save_path is not None:
            plt.savefig(os.path.join(save_path, "c_x_mean_heatmap.png"))
        
        if show:
            plt.show()

        return ax

@dataclass
class CoincidenceMatrix(OperatorMixin):
    """
    Calculate coincidence index matrix from the cross-correlograms.
    (Gives you a matrix that tells you how correlated two channels/neurons/sensors/regions are)
    For more information see equations 2 and 3 on page 52 of [paper link]

    Example usage code::
        spike_detection: Operator = ThresholdCutoff(cutoff=5.0, dead_time=0.003)
        burst_filter: Operator = BurstsFilter()
        cross_correlogram: Operator = CrossCorrelograms()
        coincidence_matrix: Operator = CoincidenceMatrix()
        
        
        data >> spike_detection >> burst_filter >> cross_correlogram
        spike_detection >> cross_correlogram >> coincidence_matrix
        Pipeline(coincidence_matrix).run()

    Parameters
    ----------
    tag : str
        Operator tag
    deltaT : float
        Time window for cross-correlograms
    binsize : float
        Bin size for binning the spikes in the cross-correlograms
    
    Attributes
    ----------
    binN : int
        Number of bins in the correlation array
    """

    tag: str = "coincidence matrix"
    deltaT: float = 0.15
    binsize: float = 0.01
    binN: float = round(2*(deltaT/binsize))

    def __post_init__(self):
        super().__init__()

    @wrap_cacher(cache_tag="CI_XY_matrix")
    def __call__(self, C_XY):
        # Janky way of finding center bin(s). Low priority refactor.
        binN = self.binN
        k = 1
        mL = int(np.floor((binN-1)/2))
        mR = int(np.ceil((binN-1)/2))
        if(mL == mR) : k = 0.5

        C_XY = np.asarray(C_XY)
        CHAMOUNT = len(C_XY)
        C_XY_sum = np.sum(C_XY, axis=2)

        CI_XY = np.divide(
            k*(C_XY[:, :, mL] + C_XY[:, :, mR]),
            C_XY_sum,
            out=np.zeros((CHAMOUNT, CHAMOUNT)),
            where=C_XY_sum != 0,
        )
        return CI_XY

    def plot_ci_heatmap(
        self,
        CI_XY,
        show: bool = False,
        save_path: Optional[pathlib.Path] = None,
    ):
        fig, ax = plt.subplots()
        plt.imshow(CI_XY, cmap='hot', interpolation='nearest')
        plt.colorbar()
        if save_path is not None:
           plt.savefig(os.path.join(save_path, f"ci_heatmap.png"))
        ax.set_xlabel('Channel')
        ax.set_ylabel('Channel')
        if show:
            plt.show()
    
        return ax

    def plot_ci_scatter(
        self,
        CI_XY,
        ax = None,
        show: bool = False,
        save_path: Optional[pathlib.Path] = None,
    ):
        if(ax == None) : fig, ax = plt.subplots()
        for i, sublist in enumerate(CI_XY):
            x_values = [i+1] * len(sublist)
            y_values = sublist
            ax.scatter(x_values, y_values)
        ax.set_yscale('log')
        ax.set_xlabel('Channel')
        ax.set_ylabel('Correlation Index')
        #max_val = np.max(CI_XY)
        #plt.yticks(list(plt.yticks()[0]) + [max_val])
        if save_path is not None:
           plt.savefig(os.path.join(save_path, f"ci_scatter.png"))
        
        if show:
            plt.show()
    
        return ax

    def plot_ci_histogram(
        self,
        CI_XY,
        ax = None,
        show: bool = False,
        save_path: Optional[pathlib.Path] = None,
    ):
        if(ax == None) : fig, ax = plt.subplots()
        Histo = np.array(CI_XY).flatten()
        Histo = Histo[Histo != 0]
        ax.hist(Histo, bins=500, edgecolor='black')
        # Another magic number that should TODO: be given logic
        ax.set_xlabel('Correlation Index')
        ax.set_ylabel('Count')
        if save_path is not None:
           plt.savefig(os.path.join(save_path, "ci_histogram.png"))
        
        if show:
            plt.show()
    
        return ax
    
    def plot_ci_combined(
        self,
        CI_XY,
        show: bool = False,
        save_path: Optional[pathlib.Path] = None,
    ):
        fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 3]})

        # Plot CI Scatter
        self.plot_ci_scatter(CI_XY, ax2)  # Don't show the subplots here
        ax2.set_yscale('log')
        
        # Plot CI Histogram
        self.plot_ci_histogram(CI_XY, ax1)
        counts = [patch.get_height() for patch in ax1.patches]
        bins = [patch.get_x() for patch in ax1.patches]

        # Clear the second axes
        ax1.clear()

        # Plot the histogram data on the second axes, swapping x and y
        ax1.set_yscale('log')
        ax1.set_xlabel('counts')
        ax1.plot(counts, bins)
        ax2.set_ylabel('')
        ax1.set_ylabel('Correlation Index')
        if save_path is not None:
            plt.savefig(os.path.join(save_path, "ci_experiment.png"))
        
        if show:
            plt.show()


