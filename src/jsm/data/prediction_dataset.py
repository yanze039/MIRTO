"""Parametric prediction-task data module.

Collapses the 5 near-identical files
  apex_regression.py, te_regression.py, stability_regression.py,
  protein_loc.py, rna_loc.py
into a single Dataset / BatchConverter / DataModule trio configured by the
per-task knobs: which CSV/parquet columns hold the labels, which column is
the row index, the label-dict key emitted into the batch, the label dim, and
the dtype.

YAMLs in `src/diffusion_configs/data/*_prediction.yaml` can target this
class directly; per-task Python shim files in this directory subclass it with
class-attr defaults so existing import paths keep working.

Per-task spec (the 5 leaves):

  apex        : label_columns = ['Nucleus','Nucleolus','Lamina','Nuclear_Pore',
                                 'Cytosol','ERM','OMM','ER_Lumen']
                label_key = 'apex_label',  index_col = 'best_refseq_mrna_id'
                label_dim = 8
  te          : 78 cell-type columns,
                label_key = 'te_label',    index_col = 'best_refseq_mrna_id'
                label_dim = 78
  stability   : ['normalized_half_life'],
                label_key = 'stability_label', index_col = 'best_refseq_mrna_id'
                label_dim = 1   (regression target — float in the data)
  protein_loc : ['Cytoplasm','Nucleus','Cell membrane'],
                label_key = 'proteinloc_label', index_col = 'ensg_id'
                label_dim = 3
  rna_loc     : 7 sorted columns,
                label_key = 'rnaloc_label', index_col = 'ensg_id'
                label_dim = 7

Refactor notes:
  * The original 5 files duplicated `tokenize_inputs`/`prepare_inputs_for_model`/
    `prepare_protein_inputs_for_model` at the bottom of each. The canonical
    versions live in jsm.data.species_specific; the duplicates are dropped.
  * Original code hardcoded `species = 'Homo sapiens'` per file. Kept as the
    default of the `hardcoded_species` knob (override via DataModule kwarg).
  * Label dtype was `int64` in all 5 original files even for stability (which
    is a continuous regression target). The downstream regressor casts with
    `.float()` so the bug was silent. Default kept as `int64` for compatibility;
    set `label_dtype='float32'` to fix.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Sequence, Union

import lightning.pytorch as pl
import msgpack
import pandas as pd
import torch
import torch.distributed as dist
import zstandard as zstd
from torch.utils.data import Dataset

from .utils import (
    Alphabet,
    NoncanonicalNucleotideLabels,
    esm_tokenize,
    modality_map,
    nucleotide_label_combination_rules,
    truncate,
)


# --------------------------------------------------------------------------- #
# Shared utilities (previously duplicated across all 5 leaf files).
# --------------------------------------------------------------------------- #
def is_main_process() -> bool:
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def broadcast_error_flag(error_flag: int) -> int:
    error_tensor = torch.tensor([error_flag], dtype=torch.int64)
    dist.broadcast(error_tensor, src=0)
    return error_tensor.item()


def read_compressed_msgpack(file_path):
    with open(file_path, 'rb') as fp:
        decompressor = zstd.ZstdDecompressor()
        return msgpack.unpackb(decompressor.decompress(fp.read()), raw=False)


def save_compressed_msgpack(file_path, obj):
    packed = msgpack.packb(obj, use_bin_type=True)
    compressed = zstd.ZstdCompressor(level=3).compress(packed)
    with open(file_path, 'wb') as fp:
        fp.write(compressed)


def _handle_special_nucleotides(
    sequence,
    max_length=2046,
    random_truncate=True,
    inverse_truncate=False,
    replace_T=True,
):
    if max_length < 0:
        pass
    elif len(sequence) > max_length:
        sequence = truncate(
            sequence, max_length,
            random_truncate=random_truncate,
            inverse_truncate=inverse_truncate)
    sequence = sequence.strip().upper()
    new_sequence = ''
    for char in sequence:
        if char in NoncanonicalNucleotideLabels:
            new_sequence += random.choice(
                nucleotide_label_combination_rules[char])
        else:
            new_sequence += char
    if replace_T:
        new_sequence = new_sequence.replace('T', 'U')
    else:
        new_sequence = new_sequence.replace('U', 'T')
    return new_sequence


def generate_random_sequence(sequence_length: int) -> str:
    return ''.join(random.choice(['A', 'U', 'C', 'G'])
                   for _ in range(sequence_length))


_LABEL_DTYPE_MAP = {
    'int64': torch.int64,
    'int32': torch.int32,
    'float32': torch.float32,
    'float64': torch.float64,
}


def _resolve_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype in _LABEL_DTYPE_MAP:
        return _LABEL_DTYPE_MAP[dtype]
    raise ValueError(
        f'label_dtype must be a torch.dtype or one of '
        f'{sorted(_LABEL_DTYPE_MAP)}; got {dtype!r}')


# --------------------------------------------------------------------------- #
# Dataset.
# --------------------------------------------------------------------------- #
class PredictionJointSequenceDataset(Dataset):
    """Single CSV/parquet table → per-row tuple of (RNA + protein + label)."""

    # Subclass-overridable defaults.
    LABEL_COLUMNS: Optional[List[str]] = None
    INDEX_COL: str = 'best_refseq_mrna_id'
    SORT_LABEL_COLUMNS: bool = False

    def __init__(
        self,
        data_file_path,
        metadata_path,
        label_columns: Optional[Sequence[str]] = None,
        *,
        index_col: Optional[str] = None,
        sort_label_columns: Optional[bool] = None,
        mode: str = 'train',
        hardcoded_species: str = 'Homo sapiens',
        max_transcript_length: int = 8000,
    ):
        assert mode in ('train', 'val', 'test'), \
            "mode must be 'train', 'val' or 'test'"
        self.data_file_path = Path(str(data_file_path))
        if self.data_file_path.suffix == '.csv':
            self.data = pd.read_csv(self.data_file_path)
        elif self.data_file_path.suffix == '.parquet':
            self.data = pd.read_parquet(self.data_file_path)
        else:
            raise ValueError(
                f'data_file_path must end in .csv or .parquet; '
                f'got {self.data_file_path}')
        self.data = self.data[self.data['split'] == mode]

        # Length-based filter (matches the original 5 leaves).
        self.data['transcript_length'] = (
            self.data['protein_sequence'].str.len()
            + self.data['utr5_sequence'].str.len()
            + self.data['cds_sequence'].str.len() // 3
            + self.data['utr3_sequence'].str.len()
            + 10
        )
        self.data = self.data[
            self.data['transcript_length'] < max_transcript_length]

        self.mode = mode
        self.protein_max_length = 2048
        self.rna_max_length = max_transcript_length
        self.hardcoded_species = hardcoded_species

        # Metadata is loaded for parity with the originals (the index column
        # lookup partition('.')[0] strips refseq version suffixes).
        self.metadata_path = metadata_path
        self.metadata = read_compressed_msgpack(self.metadata_path)
        self.metadata = {k.partition('.')[0]: v
                         for k, v in self.metadata.items()}

        cols = (list(label_columns) if label_columns is not None
                else (list(self.LABEL_COLUMNS) if self.LABEL_COLUMNS
                      else None))
        if cols is None:
            raise ValueError(
                'label_columns must be provided either via kwarg or via the '
                'subclass LABEL_COLUMNS class attribute.')
        should_sort = (sort_label_columns if sort_label_columns is not None
                       else self.SORT_LABEL_COLUMNS)
        self.loc_columns = sorted(cols) if should_sort else cols

        idx_col = index_col if index_col is not None else self.INDEX_COL

        # Only keep the columns we actually need; preserve idx_col + sequence
        # cols + label cols.
        keep_cols = list(dict.fromkeys(
            ['utr5_sequence', 'cds_sequence', idx_col,
             'utr3_sequence', 'protein_sequence']
            + self.loc_columns))
        self.data = self.data[keep_cols]
        self.data.set_index(idx_col, inplace=True)
        self.keys = self.data.index.unique().tolist()
        # Oversample to len(self.data) (preserves the original behavior).
        unique_keys = list(set(self.keys))
        if unique_keys:
            self.keys = unique_keys * (len(self.data) // len(unique_keys))
        random.shuffle(self.keys)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        rid = self.keys[idx]
        row = self.data.loc[rid]
        loc_vector = row[self.loc_columns].values.astype(float)
        return (
            rid,
            row['protein_sequence'],
            row['utr5_sequence'],
            row['cds_sequence'],
            row['utr3_sequence'],
            loc_vector,
            self.hardcoded_species,
        )

    def register_sequence_length(self, sequence_length: Optional[dict] = None):
        self.sequences_length = sequence_length


# --------------------------------------------------------------------------- #
# BatchConverter.
# --------------------------------------------------------------------------- #
class PredictionJointSequenceBatchConverter(object):
    """Per-batch collator producing the dict consumed by BasePredictionDiffusion."""

    # Subclass overrides.
    LABEL_KEY: str = ''
    LABEL_DIM: int = 0
    LABEL_DTYPE: Union[str, torch.dtype] = 'int64'

    def __init__(
        self,
        global_alphabet,
        utr_alphabet,
        codon_alphabet,
        protein_tokenizer,
        *,
        label_key: Optional[str] = None,
        label_dim: Optional[int] = None,
        label_dtype: Optional[Union[str, torch.dtype]] = None,
        max_length: Optional[int] = None,
        random_validation: Optional[str] = None,
        species_list: Sequence[str] = ('Homo sapiens',),
    ):
        self.global_tokenizer = global_alphabet
        self.utr_5_tokenizer = utr_alphabet
        self.codon_tokenizer = codon_alphabet
        self.utr_3_tokenizer = utr_alphabet
        self.protein_tokenizer = protein_tokenizer
        self.modality_map = modality_map
        self.number_of_modalities = len(self.modality_map)
        self.max_length = max_length
        self.random_validation = random_validation
        self.species_list = list(species_list)
        self.label_key = label_key or self.LABEL_KEY
        self.label_dim = label_dim if label_dim is not None else self.LABEL_DIM
        self.label_dtype = _resolve_dtype(label_dtype or self.LABEL_DTYPE)
        if not self.label_key:
            raise ValueError(
                'label_key must be set (kwarg or LABEL_KEY class attr).')
        if not self.label_dim:
            raise ValueError(
                'label_dim must be > 0 (kwarg or LABEL_DIM class attr).')

    def __call__(self, raw_batch: Sequence):
        batch_size = len(raw_batch)
        max_len = max(
            len(utr5) + len(cds) // 3 + len(utr3)
            for _, _, utr5, cds, utr3, _, _ in raw_batch)
        seq_total = max_len + 2 + 2 * 3  # 2 global + 2*3 modality boundaries

        tokens = torch.empty((batch_size, seq_total), dtype=torch.int64)
        tokens.fill_(self.global_tokenizer.padding_idx)
        modality_type_tensors = torch.empty(
            (batch_size, seq_total), dtype=torch.int64)
        modality_type_tensors.fill_(self.modality_map['padding'])
        translation_rna_mask = torch.zeros_like(modality_type_tensors)
        special_token_mask = torch.zeros_like(modality_type_tensors)
        species_tensors = torch.empty((batch_size,), dtype=torch.int64)
        label_tensors = torch.empty(
            (batch_size, self.label_dim), dtype=self.label_dtype)

        all_protein_sequence = []

        for batch_idx, (_rid, protein_seq, utr5_seq, cds_seq, utr3_seq,
                        loc_vector, species) in enumerate(raw_batch):

            label_tensors[batch_idx] = torch.as_tensor(
                loc_vector, dtype=self.label_dtype)

            cds_seq = _handle_special_nucleotides(
                cds_seq, max_length=-1, replace_T=False)
            utr5_seq = _handle_special_nucleotides(
                utr5_seq, max_length=-1, replace_T=True)
            utr3_seq = _handle_special_nucleotides(
                utr3_seq, max_length=-1, replace_T=True)

            if self.random_validation == 'utr5':
                utr5_seq = generate_random_sequence(len(utr5_seq))
            elif self.random_validation == 'utr3':
                utr3_seq = generate_random_sequence(len(utr3_seq))
            elif self.random_validation is not None:
                raise ValueError(
                    f'Unknown random validation type: {self.random_validation}')

            # <cls>
            tokens[batch_idx, 0] = self.global_tokenizer.cls_idx
            special_token_mask[batch_idx, 0] = 1
            modality_type_tensors[batch_idx, 0] = (
                self.modality_map['global_special_tokens'])
            all_protein_sequence.append(protein_seq)

            # 5' UTR
            utr5_tensor = torch.tensor(
                [self.utr_5_tokenizer.get_idx(c) for c in utr5_seq],
                dtype=torch.int64)
            tokens[batch_idx, 1] = self.global_tokenizer.get_idx('<utr_5_bos>')
            special_token_mask[batch_idx, 1] = 1
            modality_type_tensors[batch_idx, 1] = self.modality_map['utr_5']
            tokens[batch_idx, 2:2 + len(utr5_tensor)] = utr5_tensor
            modality_type_tensors[
                batch_idx, 2:2 + len(utr5_tensor)] = self.modality_map['utr_5']
            tokens[batch_idx, 2 + len(utr5_tensor)] = (
                self.global_tokenizer.get_idx('<utr_5_eos>'))
            special_token_mask[batch_idx, 2 + len(utr5_tensor)] = 1
            modality_type_tensors[
                batch_idx, 2 + len(utr5_tensor)] = self.modality_map['utr_5']

            # CDS
            codon_list = [cds_seq[i:i + 3]
                          for i in range(0, len(cds_seq), 3)]
            assert len(codon_list) == len(protein_seq) + 1, (
                f'irregular cds sequences that do not match protein: '
                f'{len(codon_list)}, {len(protein_seq)}')
            codon_tensor = torch.tensor(
                [self.codon_tokenizer.get_idx(c) for c in codon_list],
                dtype=torch.int64)
            offset = 3 + len(utr5_tensor)
            tokens[batch_idx, offset] = (
                self.global_tokenizer.get_idx('<cds_bos>'))
            special_token_mask[batch_idx, offset] = 1
            modality_type_tensors[batch_idx, offset] = self.modality_map['cds']
            tokens[batch_idx, offset + 1:offset + 1 + len(codon_tensor)] = (
                codon_tensor)
            modality_type_tensors[
                batch_idx, offset + 1:offset + 1 + len(codon_tensor)] = (
                    self.modality_map['cds'])
            cds_end = offset + 1 + len(codon_tensor)
            tokens[batch_idx, cds_end] = (
                self.global_tokenizer.get_idx('<cds_eos>'))
            special_token_mask[batch_idx, cds_end] = 1
            modality_type_tensors[batch_idx, cds_end] = self.modality_map['cds']
            translation_rna_mask[
                batch_idx, offset + 1:cds_end - 1] = 1

            # 3' UTR
            utr3_tensor = torch.tensor(
                [self.utr_3_tokenizer.get_idx(c) for c in utr3_seq],
                dtype=torch.int64)
            tokens[batch_idx, cds_end + 1] = (
                self.global_tokenizer.get_idx('<utr_3_bos>'))
            special_token_mask[batch_idx, cds_end + 1] = 1
            modality_type_tensors[batch_idx, cds_end + 1] = (
                self.modality_map['utr_3'])
            tokens[
                batch_idx,
                cds_end + 2:cds_end + 2 + len(utr3_tensor)] = utr3_tensor
            modality_type_tensors[
                batch_idx,
                cds_end + 2:cds_end + 2 + len(utr3_tensor)] = (
                    self.modality_map['utr_3'])
            utr3_eos_idx = cds_end + 2 + len(utr3_tensor)
            tokens[batch_idx, utr3_eos_idx] = (
                self.global_tokenizer.get_idx('<utr_3_eos>'))
            special_token_mask[batch_idx, utr3_eos_idx] = 1
            modality_type_tensors[batch_idx, utr3_eos_idx] = (
                self.modality_map['utr_3'])

            # global <eos>
            global_eos_idx = utr3_eos_idx + 1
            tokens[batch_idx, global_eos_idx] = self.global_tokenizer.eos_idx
            modality_type_tensors[batch_idx, global_eos_idx] = (
                self.modality_map['global_special_tokens'])
            special_token_mask[batch_idx, global_eos_idx] = 1

            species_tensors[batch_idx] = self.species_list.index(species)

        protein_input_ids = esm_tokenize(
            all_protein_sequence, self.protein_tokenizer).to(torch.int64)
        translation_protein_mask = torch.zeros_like(protein_input_ids)
        protein_eos_idx = self.protein_tokenizer.vocab['<eos>']
        for batch_idx in range(protein_input_ids.shape[0]):
            first_pos = (
                protein_input_ids[batch_idx] == protein_eos_idx
            ).nonzero(as_tuple=True)[0][0].item()
            translation_protein_mask[batch_idx, 1:first_pos] = 1

        if self.max_length is not None:
            pad_w = (self.max_length
                     - protein_input_ids.shape[1] - max_len - 8)
            padding_matrix = torch.zeros((batch_size, pad_w), dtype=torch.int64)
            translation_rna_mask = torch.cat(
                [translation_rna_mask, padding_matrix], dim=1)
            pad_pad = padding_matrix.clone()
            pad_pad.fill_(self.global_tokenizer.padding_idx)
            tokens = torch.cat([tokens, pad_pad], dim=1)
            pad_mod = padding_matrix.clone()
            pad_mod.fill_(self.modality_map['padding'])
            modality_type_tensors = torch.cat(
                [modality_type_tensors, pad_mod], dim=1)

        species_tensors = species_tensors.reshape(-1, 1)

        rna_padding_mask = tokens.ne(
            self.global_tokenizer.padding_idx).to(torch.long)
        protein_padding_mask = protein_input_ids.ne(
            self.protein_tokenizer.vocab['<pad>']).to(torch.long)
        species_padding_mask = torch.ones((batch_size, 1), dtype=torch.long)
        L = (rna_padding_mask.shape[1] + protein_padding_mask.shape[1]
             + species_tensors.shape[1])
        joint_masking = torch.cat(
            [species_padding_mask, protein_padding_mask, rna_padding_mask],
            dim=1)
        arange_tensor = torch.arange(L).unsqueeze(0).expand(batch_size, L)
        product = joint_masking * (arange_tensor + 1)
        product[product == 0] = L + 1
        row_wise_col_perms = torch.argsort(
            product, dim=1, descending=False, stable=True)
        inverse_indices = torch.empty_like(row_wise_col_perms)
        inverse_indices.scatter_(1, row_wise_col_perms, arange_tensor)
        attention_mask = torch.gather(
            joint_masking, dim=1, index=row_wise_col_perms).to(torch.int64)
        seqlens = attention_mask.sum(dim=1)
        seq_idx = torch.cat(
            [torch.full((s,), i, dtype=torch.int32)
             for i, s in enumerate(seqlens)],
            dim=0).unsqueeze(0)

        utr5_mask = (
            modality_type_tensors == self.modality_map['utr_5']).to(torch.long)
        utr3_mask = (
            modality_type_tensors == self.modality_map['utr_3']).to(torch.long)
        cds_mask = (
            modality_type_tensors == self.modality_map['cds']).to(torch.long)
        modality_mask = utr5_mask + utr3_mask + cds_mask

        return {
            'protein_sequence': all_protein_sequence,
            'rna_input_ids': tokens,
            'protein_input_ids': protein_input_ids,
            'modality_type_ids': modality_type_tensors,
            'translation_rna_mask': translation_rna_mask,
            'translation_protein_mask': translation_protein_mask,
            'rna_padding_mask': rna_padding_mask,
            'protein_padding_mask': protein_padding_mask,
            'row_wise_col_perms': row_wise_col_perms,
            'inverse_row_wise_col_perms': inverse_indices,
            'attention_mask': attention_mask,
            'joint_mask': joint_masking,
            'seq_idx': seq_idx,
            'species_ids': species_tensors,
            'utr5_mask': utr5_mask,
            'utr3_mask': utr3_mask,
            'cds_mask': cds_mask,
            'modality_mask': modality_mask,
            'special_token_mask': special_token_mask,
            self.label_key: label_tensors,
        }


# --------------------------------------------------------------------------- #
# DataModule.
# --------------------------------------------------------------------------- #
class PredictionJointSequenceDataModule(pl.LightningDataModule):
    """Parametric LightningDataModule for one of the 5 prediction tasks.

    Per-task class attrs (subclasses override) OR constructor kwargs:
      LABEL_COLUMNS / label_columns
      LABEL_KEY     / label_key
      LABEL_DIM     / label_dim
      INDEX_COL     / index_col
      LABEL_DTYPE   / label_dtype          ('int64', 'float32', ...)
      SORT_LABEL_COLUMNS / sort_label_columns
      HARDCODED_SPECIES  / hardcoded_species
    """

    LABEL_COLUMNS: Optional[List[str]] = None
    LABEL_KEY: str = ''
    LABEL_DIM: int = 0
    INDEX_COL: str = 'best_refseq_mrna_id'
    LABEL_DTYPE: Union[str, torch.dtype] = 'int64'
    SORT_LABEL_COLUMNS: bool = False
    HARDCODED_SPECIES: str = 'Homo sapiens'

    DATASET_CLS = PredictionJointSequenceDataset
    BATCH_CONVERTER_CLS = PredictionJointSequenceBatchConverter

    def __init__(
        self,
        data_path: str,
        metadata_path: str,
        *,
        label_columns: Optional[Sequence[str]] = None,
        label_key: Optional[str] = None,
        label_dim: Optional[int] = None,
        index_col: Optional[str] = None,
        label_dtype: Optional[Union[str, torch.dtype]] = None,
        sort_label_columns: Optional[bool] = None,
        hardcoded_species: Optional[str] = None,
        training_key_list_file: Optional[str] = None,
        validation_key_list_file: Optional[str] = None,
        test_key_list_file: Optional[str] = None,
        num_workers: int = 16,
        batch_size: Optional[int] = None,
        tokens_per_batch: Optional[int] = None,
        sequence_length_file: Optional[str] = None,
        work_dir: Union[str, Path] = Path().cwd(),
        overwrite: bool = False,
        protein_tokenizer=None,
        random_validation: Optional[str] = None,
        max_transcript_length: int = 8000,
    ):
        super().__init__()
        self.data_path = data_path
        self.metadata_path = metadata_path

        self.label_columns = list(
            label_columns if label_columns is not None
            else (self.LABEL_COLUMNS or []))
        if not self.label_columns:
            raise ValueError(
                'label_columns required (kwarg or LABEL_COLUMNS class attr)')
        self.label_key = label_key or self.LABEL_KEY
        if not self.label_key:
            raise ValueError(
                'label_key required (kwarg or LABEL_KEY class attr)')
        self.label_dim = (
            label_dim if label_dim is not None
            else (self.LABEL_DIM or len(self.label_columns)))
        self.index_col = index_col if index_col is not None else self.INDEX_COL
        self.label_dtype = label_dtype if label_dtype is not None else self.LABEL_DTYPE
        self.sort_label_columns = (
            sort_label_columns if sort_label_columns is not None
            else self.SORT_LABEL_COLUMNS)
        self.hardcoded_species = (
            hardcoded_species if hardcoded_species is not None
            else self.HARDCODED_SPECIES)
        self.max_transcript_length = max_transcript_length

        self.training_key_list_file = training_key_list_file
        self.validation_key_list_file = validation_key_list_file
        self.test_key_list_file = test_key_list_file
        if batch_size is None and tokens_per_batch is None:
            raise ValueError(
                'either batch_size or tokens_per_batch must be specified')
        if tokens_per_batch is not None:
            if sequence_length_file is None:
                raise ValueError(
                    'sequence_length_file must be specified when '
                    'tokens_per_batch is specified')
            if batch_size is not None:
                raise ValueError(
                    'batch_size and tokens_per_batch are mutually exclusive')
        self.batch_size = batch_size
        self.toks_per_batch = tokens_per_batch

        self.global_tokenizer = Alphabet.initialize_for_global(offset=0)
        self.utr_offset = len(self.global_tokenizer.all_toks)
        self.utr_alphabet = Alphabet.initialize_for_utr(offset=self.utr_offset)
        self.codon_offset = (
            self.utr_offset + len(self.utr_alphabet.all_toks))
        self.codon_alphabet = Alphabet.initialize_for_codon(
            offset=self.codon_offset)
        self.protein_tokenizer = protein_tokenizer
        self.rna_vocab_size = (
            len(self.global_tokenizer) + len(self.utr_alphabet)
            + len(self.codon_alphabet) + len(self.utr_alphabet))
        self.protein_vocab_size = len(self.protein_tokenizer)
        self.num_workers = num_workers
        self.sequence_length_file = sequence_length_file
        self.work_dir = Path(work_dir)
        self.overwrite = overwrite
        self.training_dataset = None
        self.validation_dataset = None
        self.test_dataset = None

        metadata = read_compressed_msgpack(self.metadata_path)
        self.species_list = list(set(metadata.values()))
        self.species_list.append('Unknown')
        self.species_list.sort()

        self.batch_converter = self.BATCH_CONVERTER_CLS(
            global_alphabet=self.global_tokenizer,
            utr_alphabet=self.utr_alphabet,
            codon_alphabet=self.codon_alphabet,
            protein_tokenizer=self.protein_tokenizer,
            label_key=self.label_key,
            label_dim=self.label_dim,
            label_dtype=self.label_dtype,
            random_validation=random_validation,
            species_list=self.species_list,
        )

    def _build_dataset(self, mode: str):
        return self.DATASET_CLS(
            self.data_path,
            self.metadata_path,
            label_columns=self.label_columns,
            index_col=self.index_col,
            sort_label_columns=self.sort_label_columns,
            mode=mode,
            hardcoded_species=self.hardcoded_species,
            max_transcript_length=self.max_transcript_length,
        )

    def setup(self, stage: Optional[str] = None):
        if stage in ('fit', 'validate', None):
            self.training_dataset = self._build_dataset('train')
            self.validation_dataset = self._build_dataset('val')
        if stage == 'test':
            self.test_dataset = self._build_dataset('test')

    def get_dataloader(self, dataset):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.batch_converter,
            pin_memory=True,
            drop_last=True,
        )

    def train_dataloader(self):
        return self.get_dataloader(self.training_dataset)

    def val_dataloader(self):
        return self.get_dataloader(self.validation_dataset)
