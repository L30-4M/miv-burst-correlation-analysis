import os
import neo
import csv
import pytest
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from miv.core.pipeline import Pipeline
from miv.core.datatype import Spikestamps
from miv.io.openephys import DataManager
from miv.signal.spike import ThresholdCutoff
from miv.core.operator import Operator, DataLoader
from mivOSPaperImplement import CrossCorrelograms, CorrelationIndex, MeanCrossCorrelogram, BurstsFilter

spiketrains = Spikestamps()
spiketrains.append(neo.SpikeTrain(
    times=[0.1, 1.2, 1.3, 1.4, 1.5, 1.6, 4, 5, 5.1, 5.2, 8, 9.5],
    units="sec",
    t_stop=10,
))
spiketrains.append(neo.SpikeTrain(
    times=[0.1, 1.2, 1.3, 1.4, 5, 6],
    units="sec",
    t_stop=10,
))

# Test set for BurstsFilter class
def test_bursts_filter_call():
    # Initialize the spiketrain as below

    bursts_filter = BurstsFilter(min_isi=0.1, min_len=3)
    result = bursts_filter(spiketrains)

    # Check that the result is a neo.SpikeTrain object
    assert isinstance(result, Spikestamps)

    # Check that the number of channels in the result matches the input
    assert result.number_of_channels == 2

    # Additional checks if required


# Test set for CrossCorrelograms class
def test_cross_correlograms_call():

    Xspikestamps = spiketrains
    Yspikestamps = spiketrains
    cross_corr = CrossCorrelograms()
    result = cross_corr(Xspikestamps, Yspikestamps)

    # Check the dimensions of the result
    assert len(result) == len(Xspikestamps)
    assert len(result[0]) == len(Yspikestamps)
    assert len(result[0][0]) == len(result[0][1])  # Assuming it's a square matrix

    # Additional checks if required


# Test set for MeanCrossCorrelogram class
def test_mean_cross_correlogram_call():

    Xspikestamps = spiketrains
    Yspikestamps = spiketrains
    cross_corr = CrossCorrelograms()
    C_XY = cross_corr(Xspikestamps, Yspikestamps)  # Replace with your cross-correlogram data

    mean_corr = MeanCrossCorrelogram()
    result = mean_corr(C_XY)

    # Check the dimensions of the result
    assert len(result) == len(C_XY)

    # Additional checks if required


# Test set for CorrelationIndex class
def test_correlation_index_call():

    spiketrains = Spikestamps()
    Xspikestamps = spiketrains
    Yspikestamps = spiketrains
    cross_corr = CrossCorrelograms()
    C_XY = cross_corr(Xspikestamps, Yspikestamps) 

    correlation_index = CorrelationIndex()
    result = correlation_index(C_XY)

    # Check the dimensions of the result
    assert len(result) == len(C_XY)
    assert len(result[0]) == len(C_XY)

    # Additional checks if required


# Test case 2: Empty inputs
def test_empty_inputs():
    # Create empty Neo SpikeTrain objects
    Xspikestamps = Spikestamps()
    Yspikestamps = Spikestamps()
    
    # Call the function and get the result
    test = CrossCorrelograms()
    result = test(Xspikestamps, Yspikestamps)
    #assert len(result) == 0
    # Perform your assertion on the result

# Test case 3: One spike in each channel
def test_one_spike_per_channel():
    # Create Neo SpikeTrain objects with one spike in each channel
    Xspikestamps = Spikestamps([[0], [1], [2]])  # First Spikestamps object with one spike in each channel and 3 channels
    Yspikestamps = Spikestamps([[0], [1], [2]])  # Second Spikestamps object with one spike in each channel and 3 channels
    
    
    # Call the function and get the result
    test = CrossCorrelograms()
    result = test(Xspikestamps, Yspikestamps)
    
    # Perform your assertion on the result
    #assert len(result) == 0
    # Perform additional assertions based on the expected output

def test_example_usage1():
    #Example usage:
    path = "D:/Globus"
    dataset: DataManager = DataManager(data_collection_path=path)
    data : DataLoader = dataset[0]
    spike_detection: Operator = ThresholdCutoff(cutoff=5.0, dead_time=0.003)
    C_XY_matrix: Operator = CrossCorrelograms()
    burst_filter: Operator = BurstsFilter()
    
    data >> spike_detection >> burst_filter >> C_XY_matrix
    burst_filter >> C_XY_matrix
    pipeline = Pipeline(C_XY_matrix) 
    pipeline.run()
    #assert len(C_XY_matrix.output) != 0


def test_example_usage2():
    #Example usage:
    path = "D:/Globus"
    dataset: DataManager = DataManager(data_collection_path=path)
    for i in [] :
        data : DataLoader = dataset[i]
        spike_detection: Operator = ThresholdCutoff(cutoff=5.0, dead_time=0.003)
        cross_correlogram: Operator = CrossCorrelograms()
        correlation_matrix: Operator = CorrelationIndex() 
        test_mean_cross_correlogram: Operator = MeanCrossCorrelogram()
        burst_filter: Operator = BurstsFilter()
        
        data >> spike_detection >> burst_filter >> cross_correlogram
        spike_detection >> cross_correlogram
        cross_correlogram >> CI_XY_matrix
        pipeline = Pipeline(CI_XY_matrix) 
        pipeline.run(working_directory="results/experiment" + str(i+1))

        