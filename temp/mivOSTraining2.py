import numpy as np
import quantities as pq
import matplotlib.pyplot as plt
#-------------FIXHACK--------------
import os
os.environ["OMP_NUM_THREADS"] = "1"
#-------------FIXHACK--------------
from miv.core.datatype import Signal
from miv.core.operator import Operator, DataLoader
from miv.core.pipeline import Pipeline
from miv.io.openephys import Data, DataManager
from miv.signal.filter import ButterBandpass, MedianFilter
from miv.signal.spike import ThresholdCutoff, ExtractWaveforms

from miv.datasets.openephys_sample import load_data

# Prepare data
dataset: DataManager = load_data(progbar_disable=True)
data: DataLoader = dataset[0]

bandpass_filter: Operator = ButterBandpass(lowcut=300, highcut=3000, order=4, tag="bandpass")
data >> bandpass_filter

def callback_statistics(self, signal:Signal):
    """Print the statistics of the filtered signal"""
    for s in signal:
        for channel in range(5):
            print(f"{channel=} | mean={s[channel].mean():.2f} | std={s[channel].std():.2f} | median={np.median(s[channel]):.2f}")
        yield s

def callback_median_histogram(self, signals:Signal):
    """Plot the histogram of the median of each channel"""
    medians = []
    for signal in signals:
        for channel in range(signal.number_of_channels):
            medians.append(np.median(signal.data[channel]))
        plt.hist(medians, bins=20)
        plt.title("Histogram of the median of each channel")
        plt.xlabel("Median (mV)")
        plt.ylabel("Count")
        yield signal

bandpass_filter << callback_statistics << callback_median_histogram

pipeline = Pipeline(bandpass_filter)
pipeline.run()