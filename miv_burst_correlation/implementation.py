__all__ = ["BurstsFilter", "CrossCorrelograms", "CorrelationMatrix", "MeanCrossCorrelogram"]

from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union

import os
import pathlib
from dataclasses import dataclass

import neo
import matplotlib.pyplot as plt
import numpy as np
import quantities as pq
from tqdm import tqdm

from miv.core.operator import Operator, OperatorMixin
from miv.core.pipeline import Pipeline
from miv.core.wrapper import wrap_cacher
from miv.core.datatype import Spikestamps
from miv.visualization.event import plot_spiketrain_raster
from miv.statistics.spiketrain_statistics import interspike_intervals, firing_rates
from miv.statistics.burst import burst_array

# TODO: add aditional check for burst rate to be inbetween 3 and 25 spikes/s (maybe?)
# TODO: Minor refactoring so code can work with arbitrary bins etc.
# TODO: Rename all functions and file names to be more clear. Much here is temporary.
# TODO: Implement visualizations from pages 44, 47
# TODO: Should add histogram visualization of burst lens and durations.

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
    """

    tag: str = "bursts filter"

    min_isi: float = 0.1
    min_len: int = 10

    burst_lens: List[int] = None
    burst_durations: List[float] = None

    def __post_init__(self):
        super().__init__()

    @wrap_cacher(cache_tag="burst_filter")
    def __call__(self, spiketrains: Spikestamps):
        erroneouscopy = spiketrains  # TODO: is this needs to be deep-copy?

        burst_lens = []
        burst_durations = []
        for i in range(spiketrains.number_of_channels):
            Q = burst_array(spiketrains[i], self.min_isi, self.min_len)

            spike = np.array(spiketrains[i])
            if np.sum(Q) != 0:
                erroneouscopy.data[i] = np.concatenate(
                    [spike[start: end + 1] for start, end in Q]
                )
                start_time = spike[Q[:, 0]]
                end_time = spike[Q[:, 1]]

                burst_lens.append(Q[:, 1] - Q[:, 0] + 1)
                burst_durations.append(end_time - start_time)
            else:
                erroneouscopy.data[i] = []

        return erroneouscopy, np.asarray(burst_lens), np.asarray(burst_durations)

    def plot_bursttrain(
        self,
        outputs,
        show: bool = False,
        save_path: Optional[pathlib.Path] = None,
    ):
        burst_spikestamps, burst_lens, burst_durations = outputs

        t0 = burst_spikestamps.get_first_spikestamp()
        tf = burst_spikestamps.get_last_spikestamp()

        # Create a single plot with adjusted x-axis limits
        fig, ax = plot_spiketrain_raster(burst_spikestamps, t0, tf)
        if save_path is not None:
            plt.savefig(os.path.join(save_path, "burst_raster.png"))

        if show:
            plt.show()
            plt.close("all")
        
        return ax
    
    
    def plot_firing_rate_histogram(self, spikestamps, show=False, save_path=None):
        """Plot firing rate histogram"""
        threshold = 3

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
        ax.axvline(
            threshold,
            color="g",
            linestyle="dashed",
            linewidth=1,
            label="Quality Threshold",
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
        spikestamps,
        show: bool = False,
        save_path: Optional[pathlib.Path] = None,
    ) -> plt.Axes:
        
        fig, ax = plt.subplots()
        for i in range(len(self.burst_lens)):
            ax.scatter(self.burst_lens[i], self.burst_durations[i])
        ax.set_xlabel('Burst length')
        ax.set_ylabel('Burst duration')
        
        if save_path is not None:
            plt.savefig(os.path.join(save_path, "burstlen_vs_burstdur.png"))
        
        if show:
            plt.show()
            plt.close("all")
        
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
            if(self.keep_autocorr) : Ys_norm[X] = [0 for _ in range(self.binN)] # Throwing out autocorrelation
            C_XY[X] = Ys_norm # Updating the C_XY matrix with the normalized data
        return C_XY

    # TODO: Implement visualization à la page 47.
    # Naive approach is display all sublots. Too many!
    # Probably best approach is to let user select which subplots to display. 
    # Alternatively only plot most "interesting" (?) channels. How to quantify?
    def plot_CXY(
        self,
        C_XY,
        show: bool = False,
        save_path: Optional[pathlib.Path] = None,
    ) : return None

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
            plt.close("all")
    
        return ax
    
    # Personal visualization metric
    def plot_CXHeatmap(self,
        C_X,
        show: bool = False,
        save_path: Optional[pathlib.Path] = None,
    ) :
        fig, ax = plt.subplots()
        plt.imshow(C_X, cmap='hot', interpolation='nearest')
        plt.colorbar()
        if save_path is not None:
            plt.savefig(os.path.join(save_path, "c_x_mean_heatmap.png"))
        
        if show:
            plt.show()
            plt.close("all")
    
        return ax

@dataclass
class CorrelationMatrix(OperatorMixin):
    """
    Calculate correlation index matrix from the cross-correlograms.
    (Gives you a matrix that tells you how correlated two channels/neurons/sensors/regions are)
    For more information see equations 2 and 3 on page 52 of [paper link]

    Example usage code::
        spike_detection: Operator = ThresholdCutoff(cutoff=5.0, dead_time=0.003)
        burst_filter: Operator = BurstsFilter()
        cross_correlogram: Operator = CrossCorrelograms()
        correlation_matrix: Operator = CorrelationMatrix()
        
        
        data >> spike_detection >> burst_filter >> cross_correlogram
        spike_detection >> cross_correlogram >> correlation_matrix
        Pipeline(correlation_matrix).run()

    Parameters
    ----------
    tag : str
        Operator tag
    """

    tag: str = "CI XY matrix"

    def __post_init__(self):
        super().__init__()

    @wrap_cacher(cache_tag="CI_XY_matrix")
    def __call__(self, C_XY, binN: int):
        # Janky way of finding center bin(s). Low priority refactor.
        k = 1
        mL = np.floor((binN-1)/2)
        mR = np.ceil((binN-1)/2)
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
        
        if show:
            plt.show()
            plt.close("all")
    
        return ax

    def plot_ci_scatter(
        self,
        CI_XY,
        show: bool = False,
        save_path: Optional[pathlib.Path] = None,
    ):
        fig, ax = plt.subplots()
        for i, sublist in enumerate(CI_XY):
            x_values = [i+1] * len(sublist)
            y_values = sublist
            plt.scatter(x_values, y_values)
    
        if save_path is not None:
           plt.savefig(os.path.join(save_path, f"ci_scatter.png"))
        
        if show:
            plt.show()
            plt.close("all")
    
        return ax

    def plot_ci_histogram(
        self,
        CI_XY,
        show: bool = False,
        save_path: Optional[pathlib.Path] = None,
    ):
        fig, ax = plt.subplots()
        Histo = np.array(CI_XY).flatten()
        Histo = Histo[Histo != 0]
        plt.hist(Histo, bins=500, edgecolor='black')
        # Another magic number that should TODO: be given logic
        if save_path is not None:
           plt.savefig(os.path.join(save_path, "ci_histogram.png"))
        
        if show:
            plt.show()
            plt.close("all")
    
        return ax


if __name__ == "__main__":
    from miv.core.operator import DataLoader
    from miv.signal.spike import ThresholdCutoff
    from miv.io.openephys import DataManager

    # Example usage:
    path = "D:/Globus"
    dataset: DataManager = DataManager(data_collection_path=path)
    for i in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
        data: DataLoader = dataset[i]
        spike_detection: Operator = ThresholdCutoff(cutoff=5.0, dead_time=0.003)
        CXY_matrix: Operator = CrossCorrelograms()
        CI_XY_matrix: Operator = CorrelationMatrix()
        burst_filter: Operator = BurstsFilter()

        data >> spike_detection >> burst_filter
        pipeline = Pipeline(burst_filter)
        pipeline.run(working_directory="results/experiment" + str(i + 1))
