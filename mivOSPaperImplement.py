import numpy as np
import math
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

spike_detection: Operator = ThresholdCutoff(cutoff=4.0, dead_time=0.002)
data >> spike_detection
spikestamps = spike_detection.output
CHAMOUNT = spikestamps.number_of_channels
delta = 0.15
binsize = 0.01
bins = 30


#t_start = spikestamps.get_first_spikestamp()
#t_end = spikestamps.get_last_spikestamp()

CI_XY = [[0 for _ in range(CHAMOUNT)] for _ in range(CHAMOUNT)]
C_XY = [[[0 for _ in range(30)] for _ in range(CHAMOUNT)] for _ in range(CHAMOUNT)]


for X in range(CHAMOUNT):
    Xtrain = spikestamps.select([X])
    X_neo = Xtrain.neo()
    number_of_spikes = len(X_neo[X])
    for timestamp in X_neo[X]:
        
        time = timestamp.magnitude.item()
        Ys = spikestamps.binning(bin_size=binsize, t_start=time-delta, t_end=time+delta, return_count=True).data
        Ys_normalized = Ys/(number_of_spikes*binsize)
        ddd = np.rot90(Ys_normalized, -1)
        for Y in range(CHAMOUNT) :
            C_XY[X][Y] += ddd[Y][:bins]

C_X = [[] for _ in range(CHAMOUNT)]
tempZ = [0 for _ in range(bins)]
for X in range(CHAMOUNT) : 
    C_X[X] = tempZ
    for Y in range(CHAMOUNT) :
        if(X != Y) : C_X[X] += C_XY[X][Y] 
    C_X[X] = np.array(C_X[X]) / (CHAMOUNT-1)

for X in range(CHAMOUNT) :
    teA = C_X[X]
    for Y in range(CHAMOUNT) :
        te = C_XY[X][Y]
        CI = te[0]/np.sum(teA)
        CI_XY[X][Y] = CI

for i, sublist in enumerate(CI_XY):
    x_values = [i+1] * len(sublist)
    y_values = sublist
    plt.scatter(x_values, y_values)

plt.show()
Histo = np.array(CI_XY).flatten()

plt.hist(Histo, bins=128, edgecolor='black')
plt.show()
