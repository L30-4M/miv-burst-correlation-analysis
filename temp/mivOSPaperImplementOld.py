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
# Indicates a temporary value
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

# Note: spike_detection outputs a Spikestamp object, not a neo.Spiketrain or np.array as the documentation states.
# Note2: Spikestamp.select() needs to be given as an array not as an int. This is because it can select multiple channels.

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
spikestamps = spike_detection.output

# THIS IS ROUGH OUTLINE FOR HOW C_XY IS CONSTRUCTED
#t_start = spikestamps.get_first_spikestamp()
#t_end = spikestamps.get_last_spikestamp()
#for j, timestamp in enumerate(temp3[i]):

C_XY_mega = []

for i in range(60):
    X = spikestamps.select([i])
    X_neo = X.neo()
    number_of_spikes = len(X_neo[i])
    # Dimensions: Dim1: XTrain Spike #, Dim2: Channel #, Dim3(/Dim2.5): Bin nummber
    # Dim1 has arbitrary length N, Dim2 has set length of 60, Dim3 has set length of 30 (although atm its 31 and only edges are at length 30)
    C_XY = []
    for timestamp in X_neo[i]:
        time = timestamp.magnitude.item()
        Y = X.binning(bin_size=0.01, t_start=time-0.15, t_end=time+0.15, return_count=True).data
        Y_normalized = Y/number_of_spikes
        Y_normalized /= 0.01
        C_XY.append(Y_normalized)
    C_XY_mega.append(C_XY)

#THIS IS ROUGH OUTLINE FOR HOW C_X IS CONTRUCTED
#C_XY is N x 60 x 30
C_X_mega = []
for C_XY in C_XY_mega :
    C_X = []
    for X in C_XY :
        temp = [0 for _ in range(30)]
        for binned in X :
            temp += binned[:30]
        temp /= (len(X)-1) # Might exclude -1 for now if I get any divide by zero errors
        C_X.append(temp)
    C_X_mega.append(C_X)    

# CI
# We look at the first bin and sum the C_XY value in that first bin through all the channels.
# Notation is a bit confusing since it implies that we are summing though a range of time.
CI_mega = [[0 for _ in range(60)] for _ in range(60)]

for X in range(60) :
    for Y in range(60) :
        #CXZERO = 
        OTHERSCLAR = np.sum(C_X_mega[X][Y])




#OTHERSCLAR = np.sum(C_X)
#CI = CXYZERO/OTHERSCLAR
#print(CI)