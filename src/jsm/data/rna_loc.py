"""RNA-localization multilabel-classification data module."""
from jsm.data.prediction_dataset import (
    PredictionJointSequenceBatchConverter,
    PredictionJointSequenceDataModule,
    PredictionJointSequenceDataset,
)

__all__ = [
    'RNALocJointSequenceDataset',
    'RNALocSpeciesSpecificJointSequenceBatchConverter',
    'RNALocSpeciesSpecificJointSequenceDataModule',
]


_LABEL_COLUMNS = [
    'Cytoplasm', 'Nucleus', 'Extracellular', 'Cell membrane',
    'Mitochondrion', 'Endoplasmic reticulum', 'membraneless organelle',
]


class RNALocJointSequenceDataset(PredictionJointSequenceDataset):
    LABEL_COLUMNS = _LABEL_COLUMNS
    INDEX_COL = 'ensg_id'
    SORT_LABEL_COLUMNS = True  # original rna_loc.py sorted the columns


class RNALocSpeciesSpecificJointSequenceBatchConverter(
        PredictionJointSequenceBatchConverter):
    LABEL_KEY = 'rnaloc_label'
    LABEL_DIM = 7
    LABEL_DTYPE = 'int64'


class RNALocSpeciesSpecificJointSequenceDataModule(
        PredictionJointSequenceDataModule):
    LABEL_COLUMNS = _LABEL_COLUMNS
    LABEL_KEY = 'rnaloc_label'
    LABEL_DIM = 7
    INDEX_COL = 'ensg_id'
    SORT_LABEL_COLUMNS = True
    DATASET_CLS = RNALocJointSequenceDataset
    BATCH_CONVERTER_CLS = RNALocSpeciesSpecificJointSequenceBatchConverter
