import pytest

from miv.core.pipeline import Pipeline
from miv.core.datatype import Spikestamps


def test_pipeline(tmp_path):
    """
    data >> spike detection >> Burst Filter >> Cross Correlogram
            spike detection >>              >> Cross Correlogram
                                               Cross Correlogram >> Coincidence Index
    """
    from miv_burst_correlation import BurstsFilter, CrossCorrelograms, CoincidenceMatrix, MeanCrossCorrelogram

    test_burststamps = Spikestamps([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])

    burst_filter = BurstsFilter()
    cross_correlograms = CrossCorrelograms()
    correlation_index = CoincidenceMatrix()
    mean_cross_correlogram = MeanCrossCorrelogram()

    test_burststamps >> burst_filter >> cross_correlograms
    burst_filter >> cross_correlograms >> correlation_index
    cross_correlograms >> mean_cross_correlogram

    Pipeline(correlation_index).run(tmp_path)

