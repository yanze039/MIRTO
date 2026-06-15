"""mRNA stability (half-life) regression data module."""
from jsm.data.prediction_dataset import (
    PredictionJointSequenceBatchConverter,
    PredictionJointSequenceDataModule,
    PredictionJointSequenceDataset,
)

__all__ = [
    'StabilityJointSequenceDataset',
    'StabilitySpeciesSpecificJointSequenceBatchConverter',
    'StabilitySpeciesSpecificJointSequenceDataModule',
]


_LABEL_COLUMNS = ['normalized_half_life']


class StabilityJointSequenceDataset(PredictionJointSequenceDataset):
    LABEL_COLUMNS = _LABEL_COLUMNS
    INDEX_COL = 'best_refseq_mrna_id'


class StabilitySpeciesSpecificJointSequenceBatchConverter(
        PredictionJointSequenceBatchConverter):
    LABEL_KEY = 'stability_label'
    LABEL_DIM = 1
    # Original code used int64 even for this continuous target; downstream
    # casts via `.float()`. Override to 'float32' if you want clean dtypes.
    LABEL_DTYPE = 'int64'


class StabilitySpeciesSpecificJointSequenceDataModule(
        PredictionJointSequenceDataModule):
    LABEL_COLUMNS = _LABEL_COLUMNS
    LABEL_KEY = 'stability_label'
    LABEL_DIM = 1
    INDEX_COL = 'best_refseq_mrna_id'
    DATASET_CLS = StabilityJointSequenceDataset
    BATCH_CONVERTER_CLS = StabilitySpeciesSpecificJointSequenceBatchConverter
