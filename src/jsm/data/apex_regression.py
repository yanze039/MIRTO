"""APEX (8-cell-type localization) data module.

All logic lives in jsm.data.prediction_dataset.PredictionJointSequenceDataModule.
Per-task knobs are class attrs so YAMLs only need to specify paths.
"""
from jsm.data.prediction_dataset import (
    PredictionJointSequenceBatchConverter,
    PredictionJointSequenceDataModule,
    PredictionJointSequenceDataset,
)

__all__ = [
    'APEXJointSequenceDataset',
    'APEXSpeciesSpecificJointSequenceBatchConverter',
    'APEXSpeciesSpecificJointSequenceDataModule',
]


_LABEL_COLUMNS = [
    'Nucleus', 'Nucleolus', 'Lamina', 'Nuclear_Pore',
    'Cytosol', 'ERM', 'OMM', 'ER_Lumen',
]


class APEXJointSequenceDataset(PredictionJointSequenceDataset):
    LABEL_COLUMNS = _LABEL_COLUMNS
    INDEX_COL = 'best_refseq_mrna_id'


class APEXSpeciesSpecificJointSequenceBatchConverter(
        PredictionJointSequenceBatchConverter):
    LABEL_KEY = 'apex_label'
    LABEL_DIM = 8
    LABEL_DTYPE = 'int64'


class APEXSpeciesSpecificJointSequenceDataModule(
        PredictionJointSequenceDataModule):
    LABEL_COLUMNS = _LABEL_COLUMNS
    LABEL_KEY = 'apex_label'
    LABEL_DIM = 8
    INDEX_COL = 'best_refseq_mrna_id'
    DATASET_CLS = APEXJointSequenceDataset
    BATCH_CONVERTER_CLS = APEXSpeciesSpecificJointSequenceBatchConverter
