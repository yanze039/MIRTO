"""Translation-efficiency (78 cell/tissue conditions) regression head."""
from jsm.diffusion.prediction_base import (
    BasePredictionDiffusion,
    PredictionHead,
    ShallowEnsembleHead,
)

__all__ = ['TEDiffusionRegressor', 'PredictionHead', 'ShallowEnsembleHead']


class TEDiffusionRegressor(BasePredictionDiffusion):
    LABEL_KEY = 'te_label'
    TASK_TYPE = 'regression'
    POOLING = 'concat_utr5_cds_utr3'
