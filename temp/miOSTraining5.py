import numpy as np
import quantities as pq
import matplotlib.pyplot as plt

from miv.core.operator import Operator, DataLoader
from miv.core.pipeline import Pipeline
from miv.io.openephys import Data, DataManager
from miv.signal.filter import ButterBandpass, MedianFilter
from miv.signal.spike import ThresholdCutoff

from miv.datasets.openephys_sample import load_data
if __name__ == '__main__':
    # Prepare data
    dataset: DataManager = load_data(progbar_disable=True)
    data: DataLoader = dataset[0]

    # Create operator modules:
    bandpass_filter: Operator = ButterBandpass(lowcut=300, highcut=3000, order=4, tag="bandpass")
    spike_detection: Operator = ThresholdCutoff(cutoff=4.0, dead_time=0.002, tag="spikes")

    data >> bandpass_filter >> spike_detection

    from miv.statistics.connectivity import DirectedConnectivity
    connectivity_analysis = DirectedConnectivity(mea="64_intanRHD", skip_surrogate=False, progress_bar=False, channels=[11, 26, 37, 50])

    spike_detection >> connectivity_analysis
    print(data.summarize())  # Print the summary of data flow

    pipeline = Pipeline(connectivity_analysis)
    print(pipeline.summarize())
    pipeline.run(working_directory="results")


    from miv.statistics.connectivity import plot_eigenvector_centrality
    connectivity_analysis << plot_eigenvector_centrality
    connectivity_analysis.plot(show=True)