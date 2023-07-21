import pytest

from miv.core.pipeline import Pipeline
from miv.core.datatype import Spikestamps



# Test set for BurstsFilter class
def test_bursts_filter_call():
    from miv_burst_correlation import BurstsFilter
    # Initialize the spiketrain as below
    spiketrains = Spikestamps([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    spiketrains.append([[0.1, 1.2, 1.3, 1.4, 1.5, 1.6, 4, 5, 5.1, 5.2, 8, 9.5]])
    spiketrains.append([[0.1, 1.2, 1.3, 1.4, 5, 6]])
    bursts_filter = BurstsFilter(min_isi=0.1, min_len=3)
    result = bursts_filter(spiketrains)

    # Check that the result is a Spikestamp object
    assert isinstance(result, Spikestamps)

    # Check that the number of channels in the result matches the input
    assert result.number_of_channels == 3

    # Additional checks if required