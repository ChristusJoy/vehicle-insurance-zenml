# Initializes the pipeline package.
"""Expose pipeline entry points for training and prediction."""

from .training_pipeline import training_pipeline

__all__ = ["training_pipeline"]
