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
dataset = load_data(progbar_disable=True)
data = dataset[0]

bandpass_filter = ButterBandpass(lowcut=400, highcut=1500, order=4)
spike_detection = ThresholdCutoff()
extract_waveforms = ExtractWaveforms(channels=[11, 26, 37, 50], plot_n_spikes=None)

data >> bandpass_filter >> spike_detection
bandpass_filter >> extract_waveforms
spike_detection >> extract_waveforms

data.visualize(show=True)

from dataclasses import dataclass
from miv.core.operator import OperatorMixin

@dataclass
class CutoutShape(OperatorMixin):
    tag = "cutout shape"

    def __post_init__(self):
        super().__init__()

    def __call__(self, cutouts):
        shapes = []
        for channel, cutout in cutouts.items():
            shapes.append(cutout.shape)
        return shapes

    def after_run_print(self, output):
        print(output)
        return output

cutout_shape = CutoutShape()
extract_waveforms >> cutout_shape

data.visualize(show=True)

pipeline = Pipeline(cutout_shape)
pipeline.run()

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

@dataclass
class PCAClustering(OperatorMixin):
    n_clusters: int = 3
    n_pca_components: int = 2
    tag = "pca clustering"

    plot_n_spikes: int = 100

    def __post_init__(self):
        super().__init__()

    def __call__(self, cutouts):
        labeled_cutout = {}
        features = {}
        labels_each_channel = {}
        for ch, cutout in cutouts.items():
            # Standardize
            scaler = StandardScaler()
            scaled_cutouts = scaler.fit_transform(cutout.data.T)

            # PCA
            pca = PCA()
            pca.fit(scaled_cutouts)
            pca.n_components = self.n_pca_components
            transformed = pca.fit_transform(scaled_cutouts)

            # GMM Clustering
            gmm = GaussianMixture(n_components=self.n_clusters, n_init=10)
            labels = gmm.fit_predict(transformed)

            cutout_for_each_labels = []
            for i in range(self.n_clusters):
                idx = labels == i
                cutout_for_each_labels.append(cutout.data[:,idx])

            labeled_cutout[ch] = cutout_for_each_labels
            features[ch] = transformed
            labels_each_channel[ch] = labels

        return dict(labeled_cutout=labeled_cutout, features=features, labels=labels_each_channel)

    def plot_pca_clustered_spikes_all_channels(self, output, show=False, save_path=None):
        labeled_cutout = output["labeled_cutout"]
        features = output["features"]
        labels = output["labels"]

        for ch, cutout in labeled_cutout.items():
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            for i in range(self.n_clusters):
                idx = labels[ch] == i
                axes[0].plot(features[ch][idx, 0], features[ch][idx, 1], ".", label=f"group {i}")


            #time = signal.timestamps * pq.s
            #time = time.rescale(pq.ms).magnitude
            for label in range(self.n_clusters):
                color = plt.rcParams["axes.prop_cycle"].by_key()["color"][label]
                for i in range(min(self.plot_n_spikes, cutout[label].shape[1])):
                    axes[1].plot(
                        #time,
                        cutout[label][:, i],
                        alpha=0.3,
                        linewidth=1,
                        color=color
                    )
        axes[0].set_title("Cluster assignments by a GMM")
        axes[0].set_xlabel("Principal Component 1")
        axes[0].set_ylabel("Principal Component 2")
        axes[0].legend()
        axes[0].axis("tight")

        axes[1].set_xlabel("Time (ms)")
        axes[1].set_ylabel("Voltage (mV)")
        axes[1].set_title(f"Spike Cutouts")

        # custom legend
        custom_lines = [plt.Line2D([0], [0], color=plt.rcParams["axes.prop_cycle"].by_key()["color"][i], lw=4,) \
                            for i in range(self.n_clusters)]
        axes[1].legend(custom_lines, [f"component {i}" for i in range(self.n_clusters)])

cluster = PCAClustering()
extract_waveforms >> cluster

data.visualize(show=True, seed=150)

pipeline = Pipeline(cluster)
print(pipeline.summarize())
pipeline.run()

cluster.plot(show=True)