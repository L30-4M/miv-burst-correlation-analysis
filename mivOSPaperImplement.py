import os
import neo
import csv
import pathlib
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union
from miv.core.wrapper import wrap_cacher
from miv.core.operator import OperatorMixin
from miv.statistics.spiketrain_statistics import interspike_intervals
from miv.visualization.event import plot_spiketrain_raster
from miv.statistics.spiketrain_statistics import firing_rates
# This can be "easily" multiprocessed as there is no dependence on results between channels. We are technically just running the burst function a bunch of times.
# I actually think most for loops in this whole file of code are areas in which the code can be sharded into many instances.

# TODO: add aditional check for burst rate to be inbetween 3 and 25 spikes/s (possibly?)
# TODO: Minor refactoring so code can work with arbitrary bins etc.
# TODO: Rename all functions and file names to be more clear. Much here is temporary.
# TODO: Implement visualizations from pages 44, 47
# TODO: Should add histogram visualization of burst lens and durations.
@dataclass
class BurstsFilter(OperatorMixin):
    tag: str = "bursts filter"
    min_isi: float = 0.1
    min_len: int = 10
    # Temporary
    burst_lens = []
    burst_durations = []

    def __post_init__(self):
        super().__init__()
        # Temporary
        self.burst_lens = []
        self.burst_durations = []

    @wrap_cacher(cache_tag="burstfilter")
    def __call__(self, spiketrains):
        # Code taken from burst function
        # I should probably include comments that where in the burst function.
        # If they are not here then refer to burst function
        min_isi = self.min_isi
        min_len = self.min_len
        erroneouscopy = spiketrains
        Chnls = spiketrains.number_of_channels
        
        for i in range(Chnls) :
            spike_interval = interspike_intervals(spiketrains[i])
            assert spike_interval.all() > 0, "Inter Spike Interval cannot be zero"
            burst_spike = (spike_interval <= min_isi).astype(np.bool_)
            delta = np.logical_xor(burst_spike[:-1], burst_spike[1:])
            interval = np.where(delta)[0]
            if len(interval) % 2:
                interval = np.append(interval, len(delta))
            interval += 1
            interval = interval.reshape([-1, 2])
            mask = np.diff(interval) >= min_len
            interval = interval[mask.ravel(), :]
            Q = np.array(interval)
            spike = np.array(spiketrains[i])
            if np.sum(Q) != 0 :
                erroneouscopy.data[i] = np.concatenate([spike[start:end+1] for start,end in Q])
                start_time = spike[Q[:, 0]]
                end_time = spike[Q[:, 1]]
                self.burst_lens.append(Q[:, 1] - Q[:, 0] + 1)
                self.burst_durations.append(end_time - start_time)
            else :
                erroneouscopy.data[i] = []
        
        return erroneouscopy
    
    def plot_bursttrain(
        self,
        spikestamps,
        show: bool = False,
        save_path: Optional[pathlib.Path] = None,
    ) -> plt.Axes:
        t0 = spikestamps.get_first_spikestamp()
        tf = spikestamps.get_last_spikestamp()

        # TODO: REFACTOR. Make single plot, and change xlim
        term = 60
        n_terms = int(np.ceil((tf - t0) / term))
        if n_terms == 0:
            # TODO: Warning message
            return None
        for idx in range(n_terms):
            fig, ax = plot_spiketrain_raster(
                spikestamps, idx * term + t0, min((idx + 1) * term + t0, tf)
            )
            if save_path is not None:
                plt.savefig(os.path.join(save_path, f"burst_raster_{idx:03d}.png"))
            if not show:
                plt.close("all")
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
            with open(
                os.path.join(f"{save_path}", "firing_rate_histogram.csv"), "w"
            ) as f:
                writer = csv.writer(f)
                writer.writerow(["channel", "firing_rate_hz"])
                data = list(enumerate(rates))
                data.sort(reverse=True, key=lambda x: x[1])
                for ch, rate in data:
                    writer.writerow([ch, rate])
        if show:
            plt.show()

        return ax
    
    # Visualization (probably?) based on what is shown on page 45
    def plot_lengthVduration(
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
            np.save(os.path.join(save_path, "burstlengths"), np.asarray(self.burst_lens))
            np.save(os.path.join(save_path, "burstdurations"), np.asarray(self.burst_durations))
            plt.savefig(os.path.join(save_path, "burststats"))
        
        if show:
            plt.show()
            plt.close("all")


@dataclass
class CrossCorrelograms(OperatorMixin):
    tag: str = "C XY matrix"

    keep_autocorr: bool = False
    deltaT: float = 0.15
    binsize: float = 0.01

    def __post_init__(self):
        super().__init__()

    # TODO: If inputs are different than it should not use the same cache.
    @wrap_cacher(cache_tag="CXYmatrix")
    def __call__(self, Xspikestamps, Yspikestamps):
        spiketimesMega = Xspikestamps.neo() # Extracts the spike times from Xspikestamps

        # Initializing constants
        binN = round(2*(self.deltaT/self.binsize))
        Xch = Xspikestamps.number_of_channels
        Ych = Yspikestamps.number_of_channels
        binsize = self.binsize
        deltaT = self.deltaT

        # Initializing empty return array
        C_XY = [[[0 for _ in range(binN)] for _ in range(Ych)] for _ in range(Xch)]

        # Main logic behind calculating cross correlogram
        for X in range(Xch) :

            Ys_binned = np.array([[0 for _ in range(Ych)] for _ in range(binN)])
            spikeQ = 0 # Counter for spikes

            for time in spiketimesMega[X].magnitude :
                # Binning the spike times in Yspikestamps within a time window and counting the number of spikes
                Ys_binned += Yspikestamps.binning(bin_size=binsize, t_start=time-deltaT, t_end=time+deltaT, return_count=True).data[:binN]
                spikeQ += 1
            
            
            if(spikeQ == 0) : spikeQ = 1 # Avoiding divide by zero error.
            Ys_norm = np.rot90(Ys_binned/(spikeQ * binsize), -1) # Normalizing the binned data
            if(self.keep_autocorr) : Ys_norm[X] = [0 for _ in range(binN)] # Throwing out autocorrelation
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
    tag: str = "C X matrix"

    def __post_init__(self):
        super().__init__()

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
            plt.savefig(os.path.join(save_path, "C_XMean"))
        
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
            plt.savefig(os.path.join(save_path, "C_XMeanHeatMap"))
        
        if show:
            plt.show()
            plt.close("all")
    
        return ax

@dataclass
class CorrelationIndex(OperatorMixin):
    tag: str = "CI XY matrix"

    def __post_init__(self):
        super().__init__()

    def __call__(self, C_XY):
        CHAMOUNT = len(C_XY)
        CI_XY = [[0 for _ in range(CHAMOUNT)] for _ in range(CHAMOUNT)] # Initializing a 2D matrix with zeros
        for X in range(CHAMOUNT) :
            for Y in range(CHAMOUNT) :
                if(np.sum(C_XY[X][Y]) == 0) : CI = 0
                else : CI = (C_XY[X][Y][14] + C_XY[X][Y][15]) / np.sum(C_XY[X][Y]) # Grabbing the two bins left and right of zero and dividing it by the number of spikes
                #                                                                    For more information see equation 2 on page 52 of [paper link]
                CI_XY[X][Y] = CI 
        return CI_XY
    
    # Visualization (probably?) based on what is shown on page 48
    def plot_CIscatter(
        self,
        CI_XY,
        show: bool = False,
        save_path: Optional[pathlib.Path] = None,
    ) :
        fig, ax = plt.subplots()
        for i, sublist in enumerate(CI_XY):
            x_values = [i+1] * len(sublist)
            y_values = sublist
            plt.scatter(x_values, y_values)
    
        if save_path is not None:
           plt.savefig(os.path.join(save_path, f"CIScatter"))
        
        if show:
            plt.show()
            plt.close("all")
    
        return ax
    
    # Personal visualization metric
    def plot_CIheatmap(
        self,
        CI_XY,
        show: bool = False,
        save_path: Optional[pathlib.Path] = None,
    ) :
        fig, ax = plt.subplots()
        plt.imshow(CI_XY, cmap='hot', interpolation='nearest')
        plt.colorbar()
        if save_path is not None:
           plt.savefig(os.path.join(save_path, f"CIHeatmap"))
        
        if show:
            plt.show()
            plt.close("all")
    
        return ax
    
    # Visualization (probably?) based on what is shown on page 48
    def plot_CIhistogram(
        self,
        CI_XY,
        show: bool = False,
        save_path: Optional[pathlib.Path] = None,
    ) :
        fig, ax = plt.subplots()
        Histo = np.array(CI_XY).flatten()
        Histo = Histo[Histo != 0]
        plt.hist(Histo, bins=500, edgecolor='black')
        # Another magic number that should TODO: be given logic
        if save_path is not None:
           plt.savefig(os.path.join(save_path, f"CIHisto"))
        
        if show:
            plt.show()
            plt.close("all")
    
        return ax
    
    def plot_CIseperatedhistograms(
        self,
        CI_XY,
        show: bool = False,
        save_path: Optional[pathlib.Path] = None,
    ) :
        
        for i, sublist in enumerate(CI_XY):
            if(np.sum(sublist) == 0): continue
            fig, ax = plt.subplots()
            plt.hist(sublist, bins=40, edgecolor='black')
            if save_path is not None:
                plt.savefig(os.path.join(save_path, f"Channel{i+1}Histogram"))
            if not show:
                plt.close("all")
        if show:
            plt.show()
            plt.close("all")

        return ax
        
