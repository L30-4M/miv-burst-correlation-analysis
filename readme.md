# (MiV) Burst Correlation Analysis

In this repository, we aim to implement the correlation index matrix algorithm proposed by Chippalone et. al. (2006): [Spike train correlations: A comparison of performance and complexity](https://www.sciencedirect.com/science/article/pii/S0006899306008018).

## Pipeline

- Implementing what was presented in the paper follows the following pipeline:

data >> spike detection >> Burst Filter >> Cross Correlogram

spike detection >> Cross Correlogram >> Coincidence Index

## Installation

To install the package, run the following command in your terminal:

```
pip install miv-burst-correlation
```

If user wants to install from source, they can clone the repository and run the following command in the root directory:

```
cd miv-burst-correlation
pip install .
```

## Implemented Modules

Currently three Operators are added:
- BurstFilter
- CrossCorrelograms
- CorrelationIndex

These three Operators do as followed:
- Takes a signal of Spikestamps and throws out any Spikes that are not part of a burst.
- Takes two signals consisting of Spikestamps (Xtrain and Ytrain respectively) and creates a cross correlogram matrix.
- Takes a cross correlogram matrix and creates a Correlation index matrix

> The analysis pipeline is developed as a plugin for the [MiV-OS](https://github.com/GazzolaLab/MiV-OS).

### Burst Detection

To detect bursts, the spike train is analyzed using a shifting time window. Two thresholds are fixed:

- MaxISI: the maximum inter-spike interval (set at 100 ms) for spikes within a burst.
- MinSpikes: the minimum number of consecutive spikes belonging to a burst (set at 10 spikes).

The spike sequence is examined, and spikes are considered part of a burst if the time delay between two adjacent pulses is less than MaxISI, and the total number of spikes accounted for is more than MinSpikes.

#### Cross-Correlogram Calculation

- We take a Spike time and create a range from time-150 ms to time+150 ms, these are then divided into 10 ms increments (so like [time-150, time-140, ..., time + 150]) we then go through each channel in the Y train and we make a binned aray of spikes in that time frame. In the case of multiple Spikes in a single Xtrain these arrays are summed into a single array. So at the end of this proccess what we will have is a 3D matrix with the following dimensions 60 x 60 x 30, where the two 60's correspond to how many channels are in the signal, and 30 corresponds to the dimension of the array.
- Given two spike trains (X and Y) recorded from two electrodes of the multi-electrode array (MEA), the code counts the number of spikes in the Y train within a time frame of ± T (T = 150 ms) around each spike in the X train.
- Bins of size Δτ = 10 ms are used to divide the time frame into intervals.
- Each resulting cross-correlogram Cxy(τ) consists of an array of 30 elements, representing the counts of spikes at different time lags (τ).
- Each element of Cxy(τ) is normalized by dividing it by the number of spikes in the X train and the size of the bin.


#### Coincidence Index Calculation

- coincidence index, CI, which represents the ratio of the integral of Cxy(0) over a time period of Δτ/2 to the integral of the total area under Cx(τ) over a time period of T, where T is the time frame (150 ms).
- The code computes the cross-correlogram coefficient, Cxy(0), by summing the values of Cxy(τ) over a time period of Δτ/2 centered at zero, where Δτ is the bin size (10 ms).
Cxy(0) represents the center of the bin, so in an array of size 30, the indecies that represent that is 15 and 16

## Side Notes

> The `spike_detection` operator outputs a `Spikestamp` object, not a `neo.Spiketrain` or `np.array` as the documentation states.

> `Spikestamp.select()` needs to be given as an array, not as an int, as it can select multiple channels.
