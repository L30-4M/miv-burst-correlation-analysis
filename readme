# README

This README file provides an overview of the code implementation for the Paper: [Spike train correlations: A comparison of performance and complexity](https://www.sciencedirect.com/science/article/pii/S0006899306008018)

## Algorithms to Implement

- Spike Train Correlation Analysis
- Burst Detection [already implemented?]

## Current Code

The current code performs Spike Train Correlation Analysis on neural spike data.

### Burst Detection

To detect bursts, the spike train is analyzed using a shifting time window. Two thresholds are fixed:
- MaxISI: the maximum inter-spike interval (set at 100 ms) for spikes within a burst.
- MinSpikes: the minimum number of consecutive spikes belonging to a burst (set at 10 spikes).

The spike sequence is examined, and spikes are considered part of a burst if the time delay between two adjacent pulses is less than MaxISI, and the total number of spikes accounted for is more than MinSpikes.

### Correlation Analysis

To study timing interactions within the neuronal networks, spike train correlation analysis is applied. The cross-correlation function measures the frequency at which one cell fires relative to the firing of a spike in another cell. The cross-correlogram represents the probability of observing a spike in one train given that there is a spike in another train.

#### Cross-Correlogram Calculation

- Given two spike trains (X and Y) recorded from two electrodes of the multi-electrode array (MEA), count the number of spikes in the Y train within a time frame of ± T (T = 150 ms) around each spike in the X train.
- Use bins of size Δτ = 10 ms to divide the time frame into intervals.
- Each resulting cross-correlogram Cxy(τ) consisted of an array of 30 elements, representing the counts of spikes at different time lags (τ).
- Normalize each element of Cxy(τ) by dividing it by the number of spikes in the X train and the size of the bin.

#### Mean Correlogram Calculation

- For each electrode X, calculate the mean correlogram, Cx, which represents the average correlation between electrode X and all other electrodes (Y ≠ X).
- Iterate over all electrodes (X) and calculate Cxy(τ) for each pair (X, Y). Sum the values of Cxy(τ) for Y ≠ X.
- Divide the sum by (N-1), where N is the total number of electrodes.

#### Cross-Correlogram Coefficient Calculation

- Compute the cross-correlogram coefficient, Cxy(0), by summing the values of Cxy(τ) over a time period of Δτ/2 centered at zero, where Δτ is the bin size (10 ms).

#### Coincidence Index Calculation

- Calculate the coincidence index, CI, which represents the ratio of the integral of Cxy(0) over a time period of Δτ/2 to the integral of the total area under Cx(τ) over a time period of T, where T is the time frame (150 ms).


# NONSENSE NOTES
Calculate SD of biological noise than set cutoff to be 7 times that.
Indicates a temporary value
Threshold ~4.4 ± 5.0 μV
maxISI set at 100 ms
minSpikes, set at 10 spikes

Pipeline Process:
- Detect Spikes
- Detect burst in Spikes
- Correlation analysis on Spiketrains throughout channels

Note: spike_detection outputs a Spikestamp object, not a neo.Spiketrain or np.array as the documentation states.
Note2: Spikestamp.select() needs to be given as an array not as an int. This is because it can select multiple channels.

