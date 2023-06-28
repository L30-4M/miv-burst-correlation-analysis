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

def array_to_dict(dict, array, X) :
    for Y, value in enumerate(array) :
        dict[(X,Y)] = dict[(X,Y)] + value[:30]

C_XY = {}
CI_XY = [[0.0 for _ in range(60)] for _ in range(60)]
for i in range(60):
    for j in range(60):
        C_XY[(i,j)] = [0 for _ in range(30)]


for i in range(60):
    X = spikestamps.select([i])
    X_neo = X.neo()
    number_of_spikes = len(X_neo[i])
    for timestamp in X_neo[i]:
        time = timestamp.magnitude.item()
        Ys = X.binning(bin_size=0.01, t_start=time-0.15, t_end=time+0.15, return_count=True).data
        Ys_normalized = Ys/(number_of_spikes*0.01)
        array_to_dict(C_XY, Ys_normalized, i)

C_X = [[] for _ in range(60)]
for X in range(60) : 
    C_X[X] = C_XY[(X,0)]  
    for Y in range(1,60) :
        C_X[X] += C_XY[(X,Y)]
    #Add 1/N-1 step
#Bug here for the X=0 case

# CI
# We look at the first bin and sum the C_XY value in that first bin through all the channels.
# Notation is a bit confusing since it implies that we are summing though a range of time.


for X in range(40) :
    teA = np.sum(C_X[X])
    for Y in range(40) :
        te = C_XY[(X,Y)]
        CI = te[0]/np.sum(teA)
        CI_XY[X][Y] = CI
        
import matplotlib.pyplot as plt
import numpy as np

CI_XY = np.zeros((40, 40))  # Initialize CI_XY array

for X in range(40):
    teA = np.sum(C_X[X])
    for Y in range(40):
        te = C_XY[(X, Y)]
        CI = te[0] / np.sum(teA)
        CI_XY[X][Y] = CI
