import os
import lmdb
import random
from typing import Optional, Union, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
import lightning.pytorch as pl
from pathlib import Path
from .utils import (Alphabet, 
                    nucleotide_label_combination_rules, 
                    NoncanonicalNucleotideLabels,
                    DistributedSequenceBucketBatchSampler,
                    truncate,
                    modality_map,
                    DataLoaderWrapper,
                    esm_tokenize
                    )
import msgpack
import zstandard as zstd


def is_main_process():
    if dist.is_initialized():
        return dist.get_rank() == 0
    else:
        return True

def read_file_to_list(file_path):
    mylist = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            if line.strip() != "":
                mylist.append(line.strip())
    return mylist


def save_list_to_file(mylist, file_path):
    with open(file_path, 'w') as f:
        f.write("\n".join(mylist))


def broadcast_error_flag(error_flag):
    # Broadcast 0 or 1
    error_tensor = torch.tensor([error_flag], dtype=torch.int64)
    dist.broadcast(error_tensor, src=0)
    return error_tensor.item()


def read_compressed_msgpack(file_path):
    with open(file_path, "rb") as fp:
        decompressor = zstd.ZstdDecompressor()
        return msgpack.unpackb(decompressor.decompress(fp.read()), raw=False)
    
def save_compressed_msgpack(file_path, obj):
    packed = msgpack.packb(obj, use_bin_type=True)
    compressed = zstd.ZstdCompressor(level=3).compress(packed)
    with open(file_path, "wb") as fp:
        fp.write(compressed)
    print(f"Saved compressed msgpack to {file_path}")

class SpeciesSpecificJointSequenceLMDBSequenceDataset(Dataset):
    def __init__(
            self, 
            lmdb_path,
            metadata_path,
            keys=None,
        ):
        """
        Initialize the LMDB dataset for direct retrieval.
        :param lmdb_path: Path to the LMDB file
        :param keys: List of known keys (sequence IDs)
        """
        self.lmdb_path = str(lmdb_path)
        self.env = lmdb.open(self.lmdb_path, readonly=True, 
                             lock=False, readahead=True, 
                             meminit=False, subdir=False,
                             map_size=200 * 1024 * 1024 * 1024)
        # If keys are not provided, retrieve all keys from the LMDB file
        if keys is None:
            with self.env.begin() as txn:
                self.keys = [key.decode("ascii") for key, _ in txn.cursor()]
        else:
            self.keys = keys  # Use the provided list of keys
        self.txn = self.env.begin(write=False)
        self.decompressor = zstd.ZstdDecompressor()
        self.protein_max_length = 2048
        self.rna_max_length = 8000
        
        self.metadata_path = metadata_path
        self.metadata = read_compressed_msgpack(self.metadata_path)

    def __len__(self):
        """Returns total number of stored sequences."""
        return len(self.keys)

    def __getitem__(self, idx):
        """Retrieves a sequence by its key."""
        
        seq_id = self.keys[idx]  # Get the sequence ID
        if not hasattr(self, "txn"):
            self.txn = self.env.begin(write=False)

        value = self.txn.get(seq_id.encode("ascii"))  # Retrieve sequence
        if value is None:
            raise KeyError(f"Sequence ID {seq_id} not found in LMDB")

        # value = pickle.loads(value)  # Deserialize the sequence
        value = msgpack.unpackb(self.decompressor.decompress(value), raw=False)  # raw=False for str instead of bytes
        # replace T with U
        protein_sequence = value[0]
        utr5_sequence = value[1]
        cds_sequence = value[2]
        utr3_sequence = value[3]
        species = self.metadata.get(seq_id, "Unknown")
        return seq_id, protein_sequence, utr5_sequence, cds_sequence, utr3_sequence, species
    
    def close(self):
        if hasattr(self, "txn"):
            self.txn.abort()
            del self.txn
        if hasattr(self, "env"):
            self.env.close()
    
    def register_sequence_length(self, 
                            sequence_length: Optional[dict] = None
                            ) -> None:
        self.sequences_length = sequence_length
    

    def get_batch_indices(self, 
                          toks_per_batch, 
                          extra_toks_per_seq=8, 
                          sort=True,
                          shuffle_batch=False,
                          consumed_key_indices=None
                          ):
        sizes = []
        for i, kk in enumerate(self.keys):
            sizes.append(
                (
                    min(self.sequences_length[kk][0], self.protein_max_length),
                    min(self.sequences_length[kk][1]+self.sequences_length[kk][2]//3+self.sequences_length[kk][3], self.rna_max_length),
                    i
                )  # Ensure we do not exceed max_length
            )
        if len(sizes) == 0:
            raise ValueError("No sequences left after skipping keys, please check your consumed_key_indices")
        if sort:
            sizes.sort(reverse=False, key=lambda x: (x[1]+x[0], x[1], x[0]))
        batches = []
        buf = []
        max_len = 0

        def _flush_current_buf():
            nonlocal max_len, buf
            if len(buf) == 0:
                return
            batches.append(buf)
            buf = []
            max_len = 0

        for sz_protein, sz_rna, i in sizes:
            sz = sz_protein + sz_rna
            sz += extra_toks_per_seq
            if max(sz, max_len) * (len(buf) + 1) > toks_per_batch:
                _flush_current_buf()
            max_len = max(max_len, sz)
            buf.append(i)

        _flush_current_buf()
        if shuffle_batch:
            random.shuffle(batches)
        return batches
    
    def get_batch_indices_by_count(self, 
                          batch_size, 
                          extra_toks_per_seq=8, 
                          sort=True,
                          shuffle_batch=False,
                          consumed_key_indices=None
                          ):
        sizes = []
        # (min(self.sequences_length[kk], self.max_length), i) for i, kk in enumerate(self.keys)
        for i, kk in enumerate(self.keys):
            sizes.append(
                (
                    min(self.sequences_length[kk][0], self.protein_max_length),
                    min(self.sequences_length[kk][1]+self.sequences_length[kk][2]//3+self.sequences_length[kk][3], self.rna_max_length),
                    i
                )  # Ensure we do not exceed max_length
            )
        if len(sizes) == 0:
            raise ValueError("No sequences left after skipping keys, please check your consumed_key_indices")
        if sort:
            sizes.sort(reverse=False, key=lambda x: (x[1], x[0]))
        batches = []
        buf = []
        count = 0

        def _flush_current_buf():
            nonlocal buf, count
            if len(buf) == 0:
                return
            batches.append(buf)
            buf = []
            count = 0

        for sz_protein, sz_rna, i in sizes:
            if count >= batch_size:
                _flush_current_buf()
            buf.append(i)
            count += 1
        
        if len(buf) > 0 and len(buf) < batch_size:
            for kk in range(batch_size-len(buf)):
                buf.append(random.randint(0, len(sizes)))
        _flush_current_buf()
        if shuffle_batch:
            random.shuffle(batches)
        return batches


def _handle_special_nucleotides(
                            sequence, 
                            max_length=2046,
                            random_truncate=True,
                            inverse_truncate=False,
                            replace_T=True,
                            ):
    if max_length < 0:
        sequence = sequence
    elif len(sequence) > max_length:
        sequence = truncate(sequence, max_length, random_truncate=random_truncate, inverse_truncate=inverse_truncate)
    new_sequence = ""
    sequence = sequence.strip().upper()
    for char in sequence:
        if char in NoncanonicalNucleotideLabels:
            new_sequence += random.choice(nucleotide_label_combination_rules[char])
        else:
            new_sequence += char
    if replace_T:
        new_sequence = new_sequence.replace("T", "U")
    else:
        new_sequence = new_sequence.replace("U", "T")
    return new_sequence


def generate_random_sequence(sequence_length):
    nas = ""
    for na_i in range(sequence_length):
        nas += (random.choice(["A", "U", "C", "G"]))
    return nas


# from typing import Tuple, Sequence, Union
class SpeciesSpecificJointSequenceBatchConverter(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(
            self, 
            global_alphabet,
            utr_alphabet, 
            codon_alphabet, 
            protein_tokenizer,
            max_length=None,
            random_validation=None,
            species_list=['Homo sapiens']
        ):
        self.global_tokenizer = global_alphabet
        self.utr_5_tokenizer = utr_alphabet
        self.codon_tokenizer = codon_alphabet
        self.utr_3_tokenizer = utr_alphabet
        self.protein_tokenizer = protein_tokenizer
        self.modality_map = modality_map
        self.number_of_modalities = len(self.modality_map.keys())
        self.max_length = max_length
        self.random_validation = random_validation
        self.species_list = species_list
        

    def __call__(self, 
                 raw_batch: Sequence,
                 ):
        # if self.nested_tensor:
            # return self.convert_nested_tensor_batch(raw_batch)
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)
        max_len = max((len(utr5_sequence) + len(cds_sequence)//3 + len(utr3_sequence)) \
                    for _, _, utr5_sequence, cds_sequence, utr3_sequence, _ in raw_batch)
        # seq_id, protein_sequence, utr5_sequence, cds_sequence, utr3_sequence
        tokens = torch.empty(
            (
                batch_size,
                max_len
                + 2  # global special tokens <cls> and <eos>
                + 2 * 3  # <cls> and <eos> for each modality, serving as boundary
            ),
            dtype=torch.int64,
        )
        modality_type_tensors = torch.empty(
            (
                batch_size,
                max_len
                + 2  # global special tokens <cls> and <eos>
                + 2 * 3  # <cls> and <eos> for each modality, serving as boundary
            ),
            dtype=torch.int64,
        )
        species_tensors = torch.empty((batch_size,), dtype=torch.int64)
        tokens.fill_(self.global_tokenizer.padding_idx)
        modality_type_tensors.fill_(self.modality_map["padding"])
        translation_rna_mask = torch.zeros_like(modality_type_tensors)
        labels = []
        
        all_protein_sequence = []
        
        for batch_idx, (label, protein_sequence, utr5_sequence, cds_sequence, utr3_sequence, species ) in enumerate(raw_batch):
            cds_sequence = _handle_special_nucleotides(cds_sequence, max_length=-1, replace_T=False)
            utr5_sequence = _handle_special_nucleotides(utr5_sequence, max_length=-1, replace_T=True)
            utr3_sequence = _handle_special_nucleotides(utr3_sequence, max_length=-1, replace_T=True)
            
            if self.random_validation == "utr5":
                utr5_sequence = generate_random_sequence(len(utr5_sequence))
            elif self.random_validation == "utr3":
                utr3_sequence = generate_random_sequence(len(utr3_sequence))
            elif self.random_validation is None:
                pass
            else:
                raise ValueError(f"Unknown random validation type: {self.random_validation}")

            labels.append(label)
            tokens[batch_idx, 0] = self.global_tokenizer.cls_idx
            modality_type_tensors[batch_idx, 0] = self.modality_map["global_special_tokens"]
            all_protein_sequence.append(protein_sequence)
            
            # >>> 5' UTR Section <<<
            utr_5_tensor = torch.tensor(
                [self.utr_5_tokenizer.get_idx(utr5_sequence[i]) for i in range(0, len(utr5_sequence))], dtype=torch.int64
            )
            tokens[
                batch_idx,
                1,
            ] = self.global_tokenizer.get_idx("<utr_5_bos>")
            modality_type_tensors[
                batch_idx,
                1
            ] = self.modality_map["utr_5"]
            tokens[
                batch_idx,
                2 : 2 + len(utr_5_tensor),
            ] = utr_5_tensor
            modality_type_tensors[
                batch_idx,
                2 : 2 + len(utr_5_tensor),
            ] = self.modality_map["utr_5"]
            tokens[
                batch_idx,
                2 + len(utr_5_tensor)
            ] = self.global_tokenizer.get_idx("<utr_5_eos>")
            modality_type_tensors[
                batch_idx,
                2 + len(utr_5_tensor)
            ] = self.modality_map["utr_5"]
            
            # >>> Codon Section <<<
            codon_list = [cds_sequence[i:i+3] for i in range(0, len(cds_sequence), 3)]
            assert len(codon_list) == len(protein_sequence) + 1, \
                f"irregular cds sequences that don't match protein, {len(codon_list)}, {len(protein_sequence)}"
            codon_tensor   = torch.tensor(
                [self.codon_tokenizer.get_idx(codon_list[i]) for i in range(0, len(codon_list))], dtype=torch.int64
            ) 
            tokens[
                batch_idx,
                3 + len(utr_5_tensor),
            ] = self.global_tokenizer.get_idx("<cds_bos>")
            modality_type_tensors[
                batch_idx,
                3 + len(utr_5_tensor),
            ] = self.modality_map["cds"]
            tokens[
                batch_idx,
                4 + len(utr_5_tensor) : 4 + len(utr_5_tensor) + len(codon_tensor),
            ] = codon_tensor
            modality_type_tensors[
                batch_idx,
                4 + len(utr_5_tensor) : 4 + len(utr_5_tensor) + len(codon_tensor),
            ] = self.modality_map["cds"]
            tokens[
                batch_idx,
                4 + len(utr_5_tensor) + len(codon_tensor)
            ] = self.global_tokenizer.get_idx("<cds_eos>")
            modality_type_tensors[
                batch_idx,
                4 + len(utr_5_tensor) + len(codon_tensor)
            ] = self.modality_map["cds"]
            translation_rna_mask[
                batch_idx,
                4 + len(utr_5_tensor) : 4 + len(utr_5_tensor) + len(codon_tensor) - 1 ,
            ] = 1

            # >>> 3' UTR Section <<<
            utr_3_tensor = torch.tensor(
                [self.utr_3_tokenizer.get_idx(utr3_sequence[i]) for i in range(0, len(utr3_sequence))], dtype=torch.int64
            ) 
            tokens[
                batch_idx,
                4 + len(utr_5_tensor) + len(codon_tensor) +1,
            ] = self.global_tokenizer.get_idx("<utr_3_bos>")
            modality_type_tensors[
                batch_idx,
                4 + len(utr_5_tensor) + len(codon_tensor) +1,
            ] = self.modality_map["utr_3"]
            tokens[
                batch_idx,
                4 + len(utr_5_tensor) + len(codon_tensor) +2 : 4 + len(utr_5_tensor) + len(codon_tensor) +2 + len(utr_3_tensor),
            ] = utr_3_tensor
            modality_type_tensors[
                batch_idx,
                4 + len(utr_5_tensor) + len(codon_tensor) +2 : 4 + len(utr_5_tensor) + len(codon_tensor) +2 + len(utr_3_tensor),
            ] = self.modality_map["utr_3"]
            tokens[
                batch_idx,
                6 + len(utr_5_tensor) + len(codon_tensor) + len(utr_3_tensor)
            ] = self.global_tokenizer.get_idx("<utr_3_eos>")
            modality_type_tensors[
                batch_idx,
                6 + len(utr_5_tensor) + len(codon_tensor) + len(utr_3_tensor)
            ] = self.modality_map["utr_3"]
            
            # global eos
            tokens[batch_idx, 7 + len(utr_5_tensor) + len(codon_tensor) + len(utr_3_tensor)] = self.global_tokenizer.eos_idx
            modality_type_tensors[batch_idx, 7 + len(utr_5_tensor) + len(codon_tensor) + len(utr_3_tensor)] = self.modality_map["global_special_tokens"]
            
            species_idx = self.species_list.index(species)
            species_tensors[batch_idx] = species_idx
            
        protein_input_ids = esm_tokenize(
                all_protein_sequence, self.protein_tokenizer
            ).to(torch.int64)
        translation_protein_mask = torch.zeros_like(protein_input_ids)
        protein_eos_idx = self.protein_tokenizer.vocab["<eos>"]
        for batch_idx in range(protein_input_ids.shape[0]):
            first_pos = (protein_input_ids[batch_idx] == protein_eos_idx).nonzero(as_tuple=True)[0][0].item()
            translation_protein_mask[batch_idx, 1:first_pos] = 1
        
        if self.max_length is not None:
            padding_matrix = torch.zeros((batch_size, self.max_length-protein_input_ids.shape[1]-max_len-8), dtype=torch.int64)
            translation_rna_mask = torch.cat([translation_rna_mask, padding_matrix], dim=1)
            padding_matrix.fill_(self.global_tokenizer.padding_idx)
            tokens = torch.cat([tokens, padding_matrix], dim=1)
            padding_matrix.fill_(self.modality_map["padding"])
            modality_type_tensors = torch.cat([modality_type_tensors, padding_matrix], dim=1)
        
        species_tensors = species_tensors.reshape(-1, 1)
        
        rna_padding_mask = tokens.ne(self.global_tokenizer.padding_idx).to(torch.long)
        protein_padding_mask = protein_input_ids.ne(self.protein_tokenizer.vocab["<pad>"]).to(torch.long)
        species_padding_mask = torch.ones((batch_size, 1), dtype=torch.long)
        L = rna_padding_mask.shape[1] + protein_padding_mask.shape[1] + species_tensors.shape[1]
        joint_masking = torch.cat([species_padding_mask, protein_padding_mask, rna_padding_mask], dim=1)        
        arange_tensor = torch.arange(L).unsqueeze(0).expand(batch_size, L)  
        product = joint_masking * (arange_tensor + 1)
        product[product == 0] = L + 1
        row_wise_col_perms = torch.argsort(product, dim=1, descending=False, stable=True)
        inverse_indices = torch.empty_like(row_wise_col_perms)
        inverse_indices.scatter_(1, row_wise_col_perms, arange_tensor)
        attention_mask = torch.gather(joint_masking, dim=1, index=row_wise_col_perms).to(torch.int64)
        # indices, cu_seqlens, _ = get_unpad_data(attention_mask)
        seqlens = attention_mask.sum(dim=1)
        seq_idx = torch.cat([torch.full((s,), i, dtype=torch.int32) for i, s in enumerate(seqlens)], dim=0).unsqueeze(0)
        
        utr5_mask = (modality_type_tensors == self.modality_map["utr_5"]).to(torch.long)
        utr3_mask = (modality_type_tensors == self.modality_map["utr_3"]).to(torch.long)
        cds_mask = (modality_type_tensors == self.modality_map["cds"]).to(torch.long)
        modality_mask = utr5_mask + utr3_mask + cds_mask
        # print statistics
        
        return {
            "labels": labels,
            "protein_sequence": all_protein_sequence,
            "rna_input_ids": tokens,
            "protein_input_ids": protein_input_ids,
            "modality_type_ids": modality_type_tensors,
            "translation_rna_mask": translation_rna_mask,
            "translation_protein_mask": translation_protein_mask,
            "rna_padding_mask": rna_padding_mask,
            "protein_padding_mask": protein_padding_mask,
            "row_wise_col_perms": row_wise_col_perms,
            "inverse_row_wise_col_perms": inverse_indices,
            "attention_mask": attention_mask,
            "joint_mask": joint_masking,
            "seq_idx": seq_idx,
            "species_ids": species_tensors,
            "utr5_mask": utr5_mask,
            "utr3_mask": utr3_mask,
            "cds_mask": cds_mask,
            "modality_mask": modality_mask
        }


class SpeciesSpecificJointSequenceDataModule(pl.LightningDataModule):
    def __init__(
            self, 
            lmdb_path: str, 
            metadata_path: str,
            training_key_list_file: Optional[str] = None,
            validation_key_list_file: Optional[str] = None,
            test_key_list_file: Optional[str] = None,
            # validation_split: float = 0.1,
            num_workers: int = 16,
            batch_size: Optional[int] = None,
            tokens_per_batch: Optional[int] = None,
            sequence_length_file: Optional[str] = None,
            work_dir: Union[str, Path] = Path().cwd(),
            overwrite: bool = False,
            protein_tokenizer=None,
            random_validation=None
        ):
        super().__init__()
        self.lmdb_path = lmdb_path
        self.training_key_list_file = training_key_list_file
        self.validation_key_list_file = validation_key_list_file
        self.test_key_list_file = test_key_list_file
        # self.validation_split = validation_split
        if batch_size is None and tokens_per_batch is None:
            raise ValueError("either batch_size or tokens_per_batch must be specified")
        elif tokens_per_batch is not None:
            if sequence_length_file is None:
                raise ValueError("sequence_length_file must be specified when tokens_per_batch is specified")
            if batch_size is not None:
                raise ValueError("batch_size and tokens_per_batch cannot be specified at the same time")
        self.batch_size = batch_size
        self.toks_per_batch = tokens_per_batch
        
        self.global_tokenizer = Alphabet.initialize_for_global(offset=0)
        self.utr_offset = len(self.global_tokenizer.all_toks)
        self.utr_alphabet = Alphabet.initialize_for_utr(offset=self.utr_offset)
        self.codon_offset = self.utr_offset + len(self.utr_alphabet.all_toks)
        self.codon_alphabet = Alphabet.initialize_for_codon(offset=self.codon_offset)
        self.protein_tokenizer = protein_tokenizer
        # get offsets
        self.rna_vocab_size = len(self.global_tokenizer) + len(self.utr_alphabet) + len(self.codon_alphabet) + len(self.utr_alphabet)
        self.protein_vocab_size = len(self.protein_tokenizer)
        self.num_workers = num_workers
        self.sequence_length_file = sequence_length_file
        self.work_dir = Path(work_dir)
        self.overwrite = overwrite
        self.training_dataset = None
        self.validation_dataset = None
        self.test_dataset = None
        self.metadata_path = metadata_path
        metadata = read_compressed_msgpack(self.metadata_path)
        self.species_list = list(set(metadata.values()))
        self.species_list.append("Unknown")
        self.species_list.sort()
        self.batch_converter = SpeciesSpecificJointSequenceBatchConverter(
            global_alphabet=self.global_tokenizer,
            utr_alphabet=self.utr_alphabet,
            codon_alphabet=self.codon_alphabet, 
            protein_tokenizer=self.protein_tokenizer,
            random_validation=random_validation,
            species_list=self.species_list
        )

    def setup(self, stage: Optional[str] = None):      
        if os.path.exists(self.training_key_list_file) and \
            os.path.exists(self.validation_key_list_file) and \
                not self.overwrite:
            print(f"training_key_list_file {self.training_key_list_file} already exists, skip generating")
            print(f"validation_key_list_file {self.validation_key_list_file} already exists, skip generating")
            training_key_list = read_compressed_msgpack(self.training_key_list_file)
            validation_key_list = read_compressed_msgpack(self.validation_key_list_file)
        else:
            raise ValueError("training_key_list_file and validation_key_list_file must be specified and exist")
        if self.sequence_length_file is not None:
            print("loading sequence length from file")
            with open(self.sequence_length_file, "rb") as f:
                sequence_length = msgpack.unpackb(
                    zstd.ZstdDecompressor().decompress(f.read()), 
                    raw=False)
            print("sequence length loaded")
        
        # Assign train/val datasets for use in dataloaders
        self.training_key_list = training_key_list
        self.validation_key_list = validation_key_list
        if stage == "fit" or stage == "validate" or stage is None:
            self.training_dataset = SpeciesSpecificJointSequenceLMDBSequenceDataset(self.lmdb_path, self.metadata_path, keys=training_key_list)
            self.validation_dataset = SpeciesSpecificJointSequenceLMDBSequenceDataset(self.lmdb_path, self.metadata_path, keys=validation_key_list)
            if self.sequence_length_file is not None:
                self.training_dataset.register_sequence_length(sequence_length)
                self.validation_dataset.register_sequence_length(sequence_length)
        
        if stage == "test":
            test_key_list = read_compressed_msgpack(self.test_key_list_file)
            self.test_dataset = SpeciesSpecificJointSequenceLMDBSequenceDataset(self.lmdb_path, self.metadata_path, keys=test_key_list)
            if self.sequence_length_file is not None:
                self.test_dataset.register_sequence_length(sequence_length)

    def train_dataloader(self):
        data_loader = self.get_dataloader(self.training_dataset)
        if self.trainer is not None and self.trainer.model.module.resumed_dataloader_state_from_ckpt is not None:
            print("Resuming dataloader state from checkpoint...")
            data_loader.batch_sampler.load_state_dict(self.trainer.model.module.resumed_dataloader_state_from_ckpt)
            self.trainer.model.module.resumed_dataloader_state_from_ckpt = None
        return data_loader

    def val_dataloader(self):
        return self.get_dataloader(self.validation_dataset)
    
    def get_dataloader(self, dataset):
        if self.toks_per_batch is None:
            distributed_bucket_sampler = DistributedSequenceBucketBatchSampler(
                dataset, batch_size=self.batch_size,
            )

            if self.trainer is not None and self.trainer.current_epoch is not None:
                distributed_bucket_sampler.set_epoch(self.trainer.current_epoch)
            return DataLoaderWrapper(dataset, 
                            shuffle=False, 
                            collate_fn=self.batch_converter,
                            num_workers=self.num_workers,
                            batch_sampler=distributed_bucket_sampler,
                            pin_memory=True,
                            prefetch_factor=2
                            # drop_last=True
                    )
        else:
            if self.trainer is not None and self.trainer.overfit_batches > 0:
                shuffle = False
            else:
                shuffle = True
            distributed_bucket_sampler = DistributedSequenceBucketBatchSampler(
                dataset, toks_per_batch=self.toks_per_batch, shuffle=shuffle
            )

            if self.trainer is not None and self.trainer.current_epoch is not None:
                distributed_bucket_sampler.set_epoch(self.trainer.current_epoch)
            return DataLoaderWrapper(dataset, 
                            shuffle=False, 
                            collate_fn=self.batch_converter,
                            num_workers=self.num_workers,
                            batch_sampler=distributed_bucket_sampler,
                            pin_memory=True,
                            prefetch_factor=2
                            # drop_last=True
                    )
           

def prepare_protein_inputs_for_model(
        protein_sequence,
        protein_tokenizer,
        protein_encoder,
        batch_size=1,
    ):
    """Prepare inputs for the model from a protein sequence."""
    with torch.no_grad():
        # Tokenize the protein sequence
        if isinstance(protein_sequence, str):
            protein_sequence = [protein_sequence]
        elif isinstance(protein_sequence, list):
            protein_sequence = [seq for seq in protein_sequence if isinstance(seq, str)]
        else:
            raise ValueError("protein_sequence must be a string or a list of strings.")
        device = protein_encoder.device
        # Tokenize the protein sequence
        protein_input_ids = esm_tokenize(
            protein_sequence, protein_tokenizer
        )
        protein_input_ids = protein_input_ids.to(device, dtype=torch.int64)
        protein_embeddings = protein_encoder(protein_input_ids).embeddings
    
    # repeat for batch size times
    protein_embeddings = protein_embeddings.repeat(batch_size, 1, 1)
    return protein_embeddings


def prepare_inputs_for_model(
        protein_sequence,
        protein_tokenizer,
        protein_encoder,
        global_tokenizer,
        batch_size=1,
        prompt=None,
        utr_5_tokenizer=None,
        codon_tokenizer=None,
    ):
    """Prepare inputs for the model from a protein sequence."""
    
    with torch.no_grad():
        if prompt is None:
            protein_embeddings = prepare_protein_inputs_for_model(
                protein_sequence,
                protein_tokenizer,
                protein_encoder,
                batch_size=batch_size,
            )
            prompt_token = global_tokenizer.cls_idx
            prompt_input_ids = torch.tensor([prompt_token], device=protein_embeddings.device, dtype=torch.int64).view(1, 1)
        elif isinstance(prompt, str):
            prompt_input_ids, protein_embeddings = tokenize_inputs(
                protein_sequence, 
                utr5_sequence=prompt,
                cds_sequence=None,
                utr3_sequence=None,
                protein_tokenizer=protein_tokenizer,
                protein_encoder=protein_encoder,
                global_tokenizer=global_tokenizer,
                codon_tokenizer=codon_tokenizer,
                utr_5_tokenizer=utr_5_tokenizer,
                utr_3_tokenizer=None,
                complete_sequence=False
            )
            protein_embeddings = protein_embeddings.repeat(batch_size, 1, 1)
        else:
            raise ValueError("prompt must be a string or None.")
    # repeat for batch size times
    prompt_input_ids = prompt_input_ids.repeat(batch_size, 1)
    return protein_embeddings, prompt_input_ids

import os
def tokenize_inputs(
        protein_sequence, 
        utr5_sequence=None,
        cds_sequence=None,
        utr3_sequence=None,
        protein_tokenizer=None,
        protein_encoder=None,
        global_tokenizer=None,
        codon_tokenizer=None,
        utr_5_tokenizer=None,
        utr_3_tokenizer=None,
        complete_sequence=True
    ):
    protein_embeddings = prepare_protein_inputs_for_model(
        protein_sequence,
        protein_tokenizer,
        protein_encoder,
        batch_size=1,
    )
    
    tokenized_length = 1  # for <cls> token
    if utr5_sequence is not None:
        tokenized_length += len(utr5_sequence) + 2
    if cds_sequence is not None:
        tokenized_length += len(cds_sequence) // 3 + 2
    if utr3_sequence is not None:
        tokenized_length += len(utr3_sequence) + 2
    if complete_sequence:
        tokenized_length += 1

    tokens = torch.zeros((1, tokenized_length), device=protein_embeddings.device)

    tokens[0, 0] = global_tokenizer.cls_idx
    
    if utr5_sequence is not None:
        utr5_sequence = _handle_special_nucleotides(utr5_sequence, max_length=-1, replace_T=True)
        # >>> 5' UTR Section <<<
        utr_5_tensor = torch.tensor(
            [utr_5_tokenizer.get_idx(utr5_sequence[i]) for i in range(0, len(utr5_sequence))], dtype=torch.int64
        )
        tokens[
            0,
            1,
        ] = global_tokenizer.get_idx("<utr_5_bos>")
        tokens[
            0,
            2 : 2 + len(utr_5_tensor),
        ] = utr_5_tensor
        tokens[
            0,
            2 + len(utr_5_tensor)
        ] = global_tokenizer.get_idx("<utr_5_eos>")
    
    if cds_sequence is not None:    
        cds_sequence = _handle_special_nucleotides(cds_sequence, max_length=-1, replace_T=False)
        # >>> Codon Section <<<
        codon_list = [cds_sequence[i:i+3] for i in range(0, len(cds_sequence), 3)]
        if len(codon_list) != len(protein_sequence) + 1:
            print(f"irregular cds sequences that don't match protein, {len(codon_list)}, {len(protein_sequence)}")
        codon_tensor   = torch.tensor(
            [codon_tokenizer.get_idx(codon_list[i]) for i in range(0, len(codon_list))], dtype=torch.int64
        ) 
        tokens[
            0,
            3 + len(utr_5_tensor),
        ] = global_tokenizer.get_idx("<cds_bos>")
        tokens[
            0,
            4 + len(utr_5_tensor) : 4 + len(utr_5_tensor) + len(codon_tensor),
        ] = codon_tensor
        tokens[
            0,
            4 + len(utr_5_tensor) + len(codon_tensor)
        ] = global_tokenizer.get_idx("<cds_eos>")
    
    if utr3_sequence is not None:
        utr3_sequence = _handle_special_nucleotides(utr3_sequence, max_length=-1, replace_T=True)
        # >>> 3' UTR Section <<<
        utr_3_tensor = torch.tensor(
            [utr_3_tokenizer.get_idx(utr3_sequence[i]) for i in range(0, len(utr3_sequence))], dtype=torch.int64
        ) 
        tokens[
            0,
            4 + len(utr_5_tensor) + len(codon_tensor) +1,
        ] = global_tokenizer.get_idx("<utr_3_bos>")
        tokens[
            0,
            4 + len(utr_5_tensor) + len(codon_tensor) +2 : 4 + len(utr_5_tensor) + len(codon_tensor) +2 + len(utr_3_tensor),
        ] = utr_3_tensor
        tokens[
            0,
            6 + len(utr_5_tensor) + len(codon_tensor) + len(utr_3_tensor)
        ] = global_tokenizer.get_idx("<utr_3_eos>")
    
    # global eos
    if complete_sequence:
        tokens[0, tokenized_length-1] = global_tokenizer.eos_idx
    tokens = tokens.long().to(protein_embeddings.device)
    return tokens, protein_embeddings


