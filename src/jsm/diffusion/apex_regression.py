"""APEX (8-cell-type expression) regression head on the MDLM backbone.

All shared logic lives in jsm.diffusion.prediction_base.BasePredictionDiffusion.
"""
from jsm.diffusion.prediction_base import (
    BasePredictionDiffusion,
    PredictionHead,
    ShallowEnsembleHead,
)

__all__ = ['APEXDiffusionRegressor', 'PredictionHead', 'ShallowEnsembleHead']


class APEXDiffusionRegressor(BasePredictionDiffusion):
    LABEL_KEY = 'apex_label'
    TASK_TYPE = 'regression'
    POOLING = 'concat_utr5_utr3'
