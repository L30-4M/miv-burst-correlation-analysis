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

# Download the sample data
path:str = load_data(progbar_disable=True).data_collection_path
print(path)

# Create data modules:
dataset: DataManager = DataManager(data_collection_path=path)
data: DataLoader = dataset[0]

# Create operator modules:
bandpass_filter: Operator = ButterBandpass(lowcut=300, highcut=3000, order=4)
lfp_filter: Operator = ButterBandpass(highcut=3000, order=2, btype='lowpass')
spike_detection: Operator = ThresholdCutoff(cutoff=4.0, dead_time=0.002)

# Build analysis pipeline
data >> bandpass_filter >> spike_detection
data >> lfp_filter

print(data.summarize())  # Print the summary of data flow

pipeline = Pipeline(spike_detection)  # Create a pipeline object to get `spike_detection` output
pipeline.run(working_directory="results/")  # Save outcome into "results" directory

filtered_signal = next(bandpass_filter.output)  # Next is used to retrieve the first fragment of the output
print(filtered_signal.shape)

time = filtered_signal.timestamps
elctrode = filtered_signal.data[:, 11]

plt.figure(figsize=(10, 5))
plt.plot(time, elctrode)
plt.xlabel("Time (s)")
plt.ylabel("Voltage (mV)")
plt.show()

median_filter = MedianFilter(threshold=100 * pq.mV)
data >> median_filter >> bandpass_filter
pipeline = Pipeline(bandpass_filter)

print(pipeline.summarize())

# Plot
spike_detection.plot(show=True)