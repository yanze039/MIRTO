"""Translation-efficiency (78 cell-type) regression data module."""
from jsm.data.prediction_dataset import (
    PredictionJointSequenceBatchConverter,
    PredictionJointSequenceDataModule,
    PredictionJointSequenceDataset,
)

__all__ = [
    'TEJointSequenceDataset',
    'TESpeciesSpecificJointSequenceBatchConverter',
    'TESpeciesSpecificJointSequenceDataModule',
]


_LABEL_COLUMNS = [
    '108T', '12T', 'A2780', 'A549', 'BJ', 'BRx-142', 'C643', 'CRL-1634',
    'Calu-3', 'Cybrid Cells', 'H1-hESC', 'H1933', 'H9-hESC', 'HAP-1',
    'HCC tumor', 'HCC_adjancent_normal', 'HCT116', 'HEK293', 'HEK293T',
    'HMECs', 'HSB2', 'HSPCs', 'HeLa', 'HeLa S3', 'HepG2', 'Huh-7.5', 'Huh7',
    'K562', 'Kidney normal tissue', 'LCL', 'LuCaP-PDX', 'MCF10A',
    'MCF10A-ER-Src', 'MCF7', 'MD55A3', 'MDA-MB-231', 'MM1.S', 'MOLM-13',
    'Molt-3', 'Mutu', 'OSCC', 'PANC1', 'PATU-8902', 'PC3', 'PC9',
    'Primary CD4+ T-cells', 'Primary human bronchial epithelial cells',
    'RD-CCL-136', 'RPE-1', 'SH-SY5Y', 'SUM159PT', 'SW480TetOnAPC', 'T47D',
    'THP-1', 'U-251', 'U-343', 'U2392', 'U2OS', 'Vero 6', 'WI38', 'WM902B',
    'WTC-11', 'ZR75-1', 'cardiac fibroblasts', 'ccRCC', 'early neurons',
    'fibroblast', 'hESC', 'human brain tumor',
    'iPSC-differentiated dopamine neurons', 'megakaryocytes',
    'muscle tissue', 'neuronal precursor cells', 'neurons',
    'normal brain tissue', 'normal prostate', 'primary macrophages',
    'skeletal muscle',
]


class TEJointSequenceDataset(PredictionJointSequenceDataset):
    LABEL_COLUMNS = _LABEL_COLUMNS
    INDEX_COL = 'best_refseq_mrna_id'


class TESpeciesSpecificJointSequenceBatchConverter(
        PredictionJointSequenceBatchConverter):
    LABEL_KEY = 'te_label'
    LABEL_DIM = 78
    LABEL_DTYPE = 'int64'


class TESpeciesSpecificJointSequenceDataModule(
        PredictionJointSequenceDataModule):
    LABEL_COLUMNS = _LABEL_COLUMNS
    LABEL_KEY = 'te_label'
    LABEL_DIM = 78
    INDEX_COL = 'best_refseq_mrna_id'
    DATASET_CLS = TEJointSequenceDataset
    BATCH_CONVERTER_CLS = TESpeciesSpecificJointSequenceBatchConverter
