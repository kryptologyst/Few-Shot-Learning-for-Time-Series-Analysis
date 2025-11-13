"""
Few-Shot Time Series Learning Package

A comprehensive Python package for few-shot learning in time series analysis.
"""

__version__ = "1.0.0"
__author__ = "Time Series Analysis Team"
__email__ = "contact@example.com"

from .timeseries_analyzer import (
    TimeSeriesAnalyzer,
    TimeSeriesDataGenerator,
    EmbeddingNetwork,
    PrototypicalNetwork,
    AnomalyDetector
)

__all__ = [
    "TimeSeriesAnalyzer",
    "TimeSeriesDataGenerator", 
    "EmbeddingNetwork",
    "PrototypicalNetwork",
    "AnomalyDetector"
]
