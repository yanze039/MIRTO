"""Protein-localization multilabel-classification data module."""
from jsm.data.prediction_dataset import (
    PredictionJointSequenceBatchConverter,
    PredictionJointSequenceDataModule,
    PredictionJointSequenceDataset,
)

__all__ = [
    'ProteinLocJointSequenceDataset',
    'ProteinLocSpeciesSpecificJointSequenceBatchConverter',
    'ProteinLocSpeciesSpecificJointSequenceDataModule',
]


_LABEL_COLUMNS = ['Cytoplasm', 'Nucleus', 'Cell membrane']


class ProteinLocJointSequenceDataset(PredictionJointSequenceDataset):
    LABEL_COLUMNS = _LABEL_COLUMNS
    INDEX_COL = 'ensg_id'


class ProteinLocSpeciesSpecificJointSequenceBatchConverter(
        PredictionJointSequenceBatchConverter):
    LABEL_KEY = 'proteinloc_label'
    LABEL_DIM = 3
    LABEL_DTYPE = 'int64'


class ProteinLocSpeciesSpecificJointSequenceDataModule(
        PredictionJointSequenceDataModule):
    LABEL_COLUMNS = _LABEL_COLUMNS
    LABEL_KEY = 'proteinloc_label'
    LABEL_DIM = 3
    INDEX_COL = 'ensg_id'
    DATASET_CLS = ProteinLocJointSequenceDataset
    BATCH_CONVERTER_CLS = ProteinLocSpeciesSpecificJointSequenceBatchConverter
