"""mRNA-stability regression head (UTR3-only pooling).

NOTE (refactor): the original stability_regression.py called
`self.forward(batch['rna_input_ids'], batch)` against a `forward(batch)`
signature — a latent bug fixed silently here.
"""
from jsm.diffusion.prediction_base import (
    BasePredictionDiffusion,
    PredictionHead,
    ShallowEnsembleHead,
)

__all__ = ['StabilityDiffusionRegressor', 'PredictionHead', 'ShallowEnsembleHead']


class StabilityDiffusionRegressor(BasePredictionDiffusion):
    LABEL_KEY = 'stability_label'
    TASK_TYPE = 'regression'
    POOLING = 'utr3_only'
