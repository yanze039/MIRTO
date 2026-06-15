"""RNA-localization multilabel classification head.

NOTE (refactor): the original rna_loc_classification.py called
`self.forward(batch['rna_input_ids'], batch)` against a 2-arg `forward(x, batch)`
where `x` was unused; both the leading arg and the dual-arg signature are
removed here in favor of `forward(batch)`.
"""
from jsm.diffusion.prediction_base import (
    BasePredictionDiffusion,
    PredictionHead,
    ShallowEnsembleHead,
)

__all__ = [
    'JointSequenceDiffusionClassifier', 'PredictionHead', 'ShallowEnsembleHead']


class JointSequenceDiffusionClassifier(BasePredictionDiffusion):
    LABEL_KEY = 'rnaloc_label'
    TASK_TYPE = 'multilabel'
    POOLING = 'mean_utr5_cds_utr3'
