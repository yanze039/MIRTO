"""Protein-localization multilabel classification head."""
from jsm.diffusion.prediction_base import (
    BasePredictionDiffusion,
    PredictionHead,
    ShallowEnsembleHead,
)

__all__ = [
    'ProteinLocalizationClassifier', 'PredictionHead', 'ShallowEnsembleHead']


class ProteinLocalizationClassifier(BasePredictionDiffusion):
    LABEL_KEY = 'proteinloc_label'
    TASK_TYPE = 'multilabel'
    POOLING = 'concat_utr5_utr3'
