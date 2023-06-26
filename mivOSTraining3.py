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

# Create operator modules:
bandpass_filter: Operator = ButterBandpass(lowcut=300, highcut=3000, order=4, tag="bandpass")
spike_detection: Operator = ThresholdCutoff(cutoff=4.0, dead_time=0.002, tag="spikes")

from miv.signal.spike import ExtractWaveforms

extract_waveforms: Operator = ExtractWaveforms(channels=[11, 26, 37, 50], plot_n_spikes=150)

data >> bandpass_filter >> spike_detection
bandpass_filter >> extract_waveforms
spike_detection >> extract_waveforms

data.visualize(show=True)

pipeline = Pipeline(extract_waveforms)
print(pipeline.summarize())

pipeline.run()

extract_waveforms.plot(show=True)

cutouts = extract_waveforms.output
for ch, cutout in cutouts.items():
    print(f"Channel {ch}: {cutout.data.shape}")