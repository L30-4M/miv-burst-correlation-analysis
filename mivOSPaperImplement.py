import numpy as np
import quantities as pq
import elephant.spike_train_correlation as stc
import matplotlib.pyplot as plt
from neo.core import SpikeTrain
from quantities import s
#-------------FIXHACK--------------
import os
os.environ["OMP_NUM_THREADS"] = "1"
#-------------FIXHACK--------------
from miv.core.datatype import Signal, Spikestamps
from miv.core.operator import Operator, DataLoader
from miv.core.pipeline import Pipeline
from miv.signal.filter import ButterBandpass, MedianFilter
from miv.signal.spike import ThresholdCutoff, ExtractWaveforms
from miv.io.openephys import Data, DataManager
from miv.statistics.burst import burst

from miv.datasets.openephys_sample import load_data

# Prepare data
dataset = load_data(progbar_disable=True)
data = dataset[0]


# Calculate SD of biological noise than set cutoff to be 7 times that.
# 0 Indicates a temporary value
# Threshold ~4.4 ± 5.0 μV
# maxISI set at 100 ms 
# minSpikes, set at 10 spikes

# Spike trains X and Y were recorded from two electrodes of a multi-electrode array (MEA).
# The number of spikes in spike train Y within a time frame of ± T (T = 150 ms) around each spike in spike train X was counted.
# Bins of size Δτ = 10 ms were used for counting the spikes.
# Each resulting cross-correlogram Cxy(τ) consisted of an array of 30 elements.
# To obtain the correct Cxy(τ), a normalization procedure was performed.
# Normalization involved dividing each element of the array by the number of spikes in spike train X and by the size of the bin.
# Then by steps in paper(page 52) we obtain CI (Coincidence Index)

# Pipeline Process:
# Detect Spikes
# Detect burst in Spikes
# Correlation analysis on Bursts throughout channels
#storage = []
#for i in range(60) :
#    storage.append(Spikestamps(googas[i]))
# temp.number_of_channels

# Note: spike_detection outputs a Spikestamp object, not a neo.Spiketrain or np.array as the documentation states.
# Note2: Spikestamp.select() needs to be given as an array not as an int. This is because it can select multiple channels.

#    for j in range(i+1, 60) :
#        for time in storage[i].neo() :
#           storage[j].binning(bin_size=0.01, t_start=time-0.15, t_end=time+0.15, return_count=False)

# a new matrix is made. it will be kind of like:
# Channels[1][2]
# Channels[1][3]
# ...
# Channels[1][N]

# Channels[2][3]
# Channels[2][4]
# ...
# Channels[2][N]

# Total number of entrees is now (N-1)(N-2)/2

spike_detection: Operator = ThresholdCutoff(cutoff=4.0, dead_time=0.002)
data >> spike_detection
temp = spike_detection.output

# THIS IS ROUGH OUTLINE FOR HOW C_XY IS CONSTRUCTED
C_XY = [[] for _ in range(60)] 
for i in range(58,59):
    X = temp.select([i])
    t_start = X.get_first_spikestamp()
    t_end = X.get_last_spikestamp()
    temp2 = X.get_view(t_start,t_end)
    temp3 = temp2.neo()
    number_of_spikes = len(temp[i])
    for timestamp in temp3[i] :
        time = timestamp.magnitude.item()
        temp4 = temp.binning(bin_size=0.01, t_start=time-0.15, t_end=time+0.15, return_count=True)[0]
        temp4_normalized = temp4 / number_of_spikes
        temp4_normalized /= 0.01
        C_XY[i].append(temp4_normalized)
# Might need to rotate this array
print(len(C_XY))
# THIS IS ROUGH OUTLINE FOR HOW C_X IS CONTRUCTED
# C_XY is a 2D array
C_X = []
for X in C_XY :
    try :
        print(len(X[0]))
        size = len(X) - 1
        mean = np.sum(X)
        mean = mean/size
        C_X.append(mean)
    except :
        pass

# CI
# We look at the first bin and sum the C_XY value in that first bin through all the channels.
# Notation is a bit confusing since it implies that we are summing though a range of time.

CXYZERO = np.sum(C_XY[0])
OTHERSCLAR = np.sum(C_X)
CI = CXYZERO/OTHERSCLAR
print(CI)