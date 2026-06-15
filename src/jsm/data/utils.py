from typing import Sequence, Tuple
import torch
import random
import math
import os
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.distributed as dist
from esm.utils.misc import stack_variable_length_tensors
from esm.utils import encoding
from torch.distributions import Normal
import pandas as pd
from pathlib import Path


rnaseq_toks = {
    'toks': ['A', 'C', 'G', 'U', 'R', 'Y', 'K', 'M', 'S', 'W', 'B', 'D', 'H', 'V', 'N', '-']
}
simplified_rnaseq_toks = {
    'toks': ['A', 'C', 'G', 'U', "I"]
}

modality_map = {
            "utr_5":  0,
            "cds":    1,
            "utr_3":  2,
            "global_special_tokens": 3,
            "padding": 4,
        }

codon_table = {
    'ATG': 'M',    # Methionine
    'TAA': '*', 'TAG': '*', 'TGA': '*',    # Stop
    'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',    # Serine
    'TTC': 'F', 'TTT': 'F',    # Phenylalanine
    'TTA': 'L', 'TTG': 'L',    # Leucine
    'TAC': 'Y', 'TAT': 'Y',    # Tirosine
    'TGC': 'C', 'TGT': 'C',    # Cisteine
    'TGG': 'W',    # Tryptofan
    'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',    # Leucine
    'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',    # Proline
    'CAC': 'H', 'CAT': 'H',    # Histidine
    'CAA': 'Q', 'CAG': 'Q',    # Glutamine
    'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',    # Arginine
    'ATA': 'I', 'ATC': 'I', 'ATT': 'I',    # Isoleucine
    'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',    # Threonine
    'AAC': 'N', 'AAT': 'N',    # Asparagine
    'AAA': 'K', 'AAG': 'K',    # Lysine
    'AGC': 'S', 'AGT': 'S',    # Serine
    'AGA': 'R', 'AGG': 'R',    # Arginine
    'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',    # Valine
    'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',    # Alanine
    'GAC': 'D', 'GAT': 'D',    # Aspartic Acid
    'GAA': 'E', 'GAG': 'E',    # Glutamic Acid
    'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G'     # Glycine
}

rna_codon_table = {
    'AUG': 'M', 
    'UAA': '*', 'UAG': '*', 'UGA': '*', 
    'UCA': 'S', 'UCC': 'S', 'UCG': 'S', 'UCU': 'S', 
    'UUC': 'F', 'UUU': 'F', 
    'UUA': 'L', 'UUG': 'L', 
    'UAC': 'Y', 'UAU': 'Y', 
    'UGC': 'C', 'UGU': 'C', 
    'UGG': 'W', 
    'CUA': 'L', 'CUC': 'L', 'CUG': 'L', 'CUU': 'L', 
    'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCU': 'P', 
    'CAC': 'H', 'CAU': 'H', 
    'CAA': 'Q', 'CAG': 'Q', 
    'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGU': 'R', 
    'AUA': 'I', 'AUC': 'I', 'AUU': 'I', 
    'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACU': 'T', 
    'AAC': 'N', 'AAU': 'N', 
    'AAA': 'K', 'AAG': 'K', 
    'AGC': 'S', 'AGU': 'S', 
    'AGA': 'R', 'AGG': 'R', 
    'GUA': 'V', 'GUC': 'V', 'GUG': 'V', 'GUU': 'V', 
    'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCU': 'A', 
    'GAC': 'D', 'GAU': 'D', 
    'GAA': 'E', 'GAG': 'E', 
    'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGU': 'G'
}



aminoacid_to_codon = {}
for codon, aminoacid in codon_table.items():
    if aminoacid not in aminoacid_to_codon:
        aminoacid_to_codon[aminoacid] = []
    aminoacid_to_codon[aminoacid].append(codon)


aminoacid_list = list(aminoacid_to_codon.keys())
aminoacid_list.remove("*")  # remove stop codon
aminoacid_list.sort()

codon_usage_file = Path(__file__).parent / "codon_usage" / "h_sapiens_9606.csv"
df_codon_usage = pd.read_csv(codon_usage_file)
df_codon_usage = df_codon_usage[df_codon_usage["amino_acid"] != "*"]
aa_with_highly_used_codons = result = df_codon_usage.loc[df_codon_usage.groupby("amino_acid")["relative_frequency"].idxmax()]

aminoacid_to_highly_used_codons = dict(zip(result.amino_acid, result.codon))
if 'U' not in aminoacid_to_highly_used_codons:
    aminoacid_to_highly_used_codons['U'] = 'UGC'
if 'X' not in aminoacid_to_highly_used_codons:
    aminoacid_to_highly_used_codons['X'] = 'GCC'


def detect_slurm_environment():
    """Detect if the script is running in a SLURM environment."""
    return "SLURM_JOB_ID" in os.environ


class Alphabet(object):
    def __init__(
        self,
        standard_toks: Sequence[str],
        prepend_toks: Sequence[str] = ("<null_0>", "<pad>", "<eos>", "<unk>"),
        append_toks: Sequence[str] = ("<cls>", "<mask>", "<sep>"),
        prepend_bos: bool = True,
        append_eos: bool = False,
        add_null_tokens: bool = True,
        offset: int = 0,
    ):
        self.standard_toks = list(standard_toks)
        self.prepend_toks = list(prepend_toks)
        self.append_toks = list(append_toks)
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos

        self.all_toks = list(self.prepend_toks)
        self.all_toks.extend(self.standard_toks)
        if add_null_tokens:
            for i in range((8 - (len(self.all_toks) % 8)) % 8):
                self.all_toks.append(f"<null_{i  + 1}>")
        self.all_toks.extend(self.append_toks)
        self.offset = offset
        self.tok_to_idx = {tok: i+self.offset for i, tok in enumerate(self.all_toks)}
        self.idx_to_tok = {i+self.offset: tok for i, tok in enumerate(self.all_toks)}
        self.pad_token = "<pad>"
        if "<unk>" in self.all_toks:
            self.unk_idx = self.tok_to_idx["<unk>"]
        else:
            self.unk_idx = None
        self.padding_idx = self.get_idx("<pad>") if "<pad>" in self.all_toks else None
        self.cls_idx = self.get_idx("<cls>") if "<cls>" in self.all_toks else None
        self.mask_idx = self.get_idx("<mask>") if "<mask>" in self.all_toks else None
        self.eos_idx = self.get_idx("<eos>") if "<eos>" in self.all_toks else None
        self.vocab_size = len(self.all_toks)
    
    def set_offset(self, offset):
        self.offset = offset
        self.tok_to_idx = {tok: i+self.offset for i, tok in enumerate(self.all_toks)}
        self.idx_to_tok = {i+self.offset: tok for i, tok in enumerate(self.all_toks)}

    def __len__(self):
        return len(self.all_toks)
    
    def get_idx(self, tok):
        if tok == "_":
            tok = "<mask>"
        if not tok in self.all_toks:
            raise RuntimeError(f"Unknown tokens {tok}")
        idx = self.tok_to_idx.get(tok, self.unk_idx)
        return idx

    def get_tok(self, ind):
        return self.idx_to_tok.get(ind, "-")

    def to_dict(self):
        return {"toks": self.all_toks}

    def get_batch_converter(self, *args, **kwargs):
        return BatchConverter(self, *args, **kwargs)

    @classmethod
    def from_dict(cls, d, **kwargs):
        return cls(standard_toks=d["toks"], **kwargs)

    @classmethod
    def initialize(cls, simplified: bool = False, offset=0) -> "Alphabet":
        prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
        append_toks = ("<mask>",)
        prepend_bos = True
        append_eos = True
        if simplified:
            standard_toks = simplified_rnaseq_toks["toks"]
        else:
            standard_toks = rnaseq_toks["toks"]
        return cls(standard_toks, prepend_toks, append_toks, prepend_bos, append_eos, offset=offset)
    
    
    @classmethod
    def initialize_for_codon(cls, offset=0) -> "Alphabet":
        standard_toks=sorted(list(codon_table.keys()))
        prepend_toks=()
        append_toks=()
        prepend_bos=False
        append_eos=False
        add_null_tokens=False
        return cls(standard_toks, prepend_toks, append_toks, prepend_bos, append_eos, add_null_tokens, offset=offset)
    
    @classmethod
    def initialize_for_utr(cls, offset=0) -> "Alphabet":
        standard_toks = simplified_rnaseq_toks["toks"]
        prepend_toks=()
        append_toks=()
        prepend_bos=False
        append_eos=False
        add_null_tokens=False
        return cls(standard_toks, prepend_toks, append_toks, prepend_bos, append_eos, add_null_tokens, offset=offset)
    
    @classmethod
    def initialize_for_global(cls, offset=0):
        standard_toks=[]
        prepend_toks=("<cls>", "<pad>", "<eos>", "<unk>")
        append_toks=(
                    "<mask>","<utr_5_bos>", "<utr_5_eos>",
                    "<cds_bos>", "<cds_eos>",
                    "<utr_3_bos>", "<utr_3_eos>",
                    )
        prepend_bos=True
        append_eos=True
        add_null_tokens=False
        return cls(standard_toks, prepend_toks, append_toks, prepend_bos, append_eos, add_null_tokens, offset=offset)
    
    def batch_decode(self, samples):
        decoded_samples = []
        for sample in samples:
            decoded_sample = []
            for idx in sample:
                # if idx == self.padding_idx:
                #     continue
                # if idx == self.eos_idx:
                #     break
                token = self.get_tok(idx)
                if token is None:
                    print(f"Warning: idx {idx} not found in idx_to_tok mapping.")
                decoded_sample.append(self.get_tok(idx))
            decoded_samples.append("".join(decoded_sample))
        return decoded_samples


NoncanonicalNucleotideLabels = ['R', 'Y', 'K', 'M', 'S', 'W', 'B', 'D', 'H', 'V', '-', "X", ".", "F", "P", "O", "N"]
nucleotide_label_combination_rules = {
    'R': ['A', 'G'],
    'Y': ['C', 'U'],
    'K': ['G', 'U'],
    'M': ['A', 'C'],    
    'S': ['C', 'G'],
    'W': ['A', 'U'],
    'B': ['C', 'G', 'U'],
    'D': ['A', 'G', 'U'],
    'H': ['A', 'C', 'U'],
    'V': ['A', 'C', 'G'],
    '-': [""],
    "X": ["A", "C", "G", "U"],
    "N": ["A", "C", "G", "U"],
    ".": [""],
    "F": ["A", "C", "G", "U"],
    "P": ["A", "C", "G", "U"],
    "O": ["A", "C", "G", "U"]
}


class BatchConverter(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, alphabet):
        self.alphabet = alphabet

    def __call__(self, 
                 raw_batch: Sequence[Tuple],
                 ):
        # if self.nested_tensor:
            # return self.convert_nested_tensor_batch(raw_batch)
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)
        max_len = max(len(seq_str) for _, seq_str in raw_batch)
        tokens = torch.empty(
            (
                batch_size,
                max_len
                + int(self.alphabet.prepend_bos)
                + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)
        labels = []

        for i, (label, seq_str) in enumerate(raw_batch):
            labels.append(label)
            if self.alphabet.prepend_bos:
                tokens[i, 0] = self.alphabet.cls_idx
            seq = torch.tensor(
                [self.alphabet.get_idx(seq_str[i]) for i in range(0, len(seq_str))], dtype=torch.int64
            )
            tokens[
                i,
                int(self.alphabet.prepend_bos) : len(seq_str) 
                                                 + int(self.alphabet.prepend_bos),
            ] = seq
            if self.alphabet.append_eos:
                tokens[
                    i, len(seq_str) + int(self.alphabet.prepend_bos)
                ] = self.alphabet.eos_idx
        
        return {
            "labels": labels,
            "input_ids": tokens,
        }
    

class DistributedBucketBatchSampler(DistributedSampler):
    def __init__(self, batch_indices, 
                 num_replicas=None, 
                 rank=None, 
                 shuffle=True,
                 drop_last=False,
                 seed=0):
        """
        Args:
            batch_indices (list of lists): List where each sublist contains indices forming a batch.
            num_replicas (int): Number of processes (usually set by DDP).
            rank (int): The current process rank.
            shuffle (bool): Whether to shuffle batches each epoch.
        """
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        # super().__init__(batch_indices, num_replicas=num_replicas, rank=rank, drop_last=True)
        self.batch_indices = batch_indices
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.drop_last = drop_last
        # Ensure batches are evenly split
        self.total_batches = len(self.batch_indices)
        if self.drop_last and len(self.batch_indices) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples_per_rank = math.ceil(
                (len(self.batch_indices) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples_per_rank = math.ceil(len(self.batch_indices) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples_per_rank * self.num_replicas
    
    def state_dict(self):
        r"""
        Returns:
            dict: State dictionary containing the epoch and seed.
        """
        return {"epoch": self.epoch, "seed": self.seed}

    def __iter__(self):
        indices = list(range(self.total_batches))

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # Shuffle indices
            indices = torch.randperm(len(indices), generator=g).tolist()
        
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # Split batches across GPUs
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter([self.batch_indices[i] for i in indices])

    def __len__(self):
        return self.num_samples_per_rank  # Number of batches per GPU
    
    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


def truncate(seq, max_len, random_truncate=False, inverse_truncate=False):
    if len(seq) > max_len:
        if random_truncate:
            start = random.randint(0, len(seq) - max_len)
            return seq[start:start + max_len]
        else:
            if inverse_truncate:
                return seq[-max_len:]
            else:
                return seq[:max_len]
    else:
        return seq



class DistributedSequenceBucketBatchSampler(DistributedSampler):
    def __init__(
            self, 
            dataset, 
            toks_per_batch=None,
            batch_size=None,
            num_replicas=None, 
            rank=None, 
            shuffle=True,
            drop_last=False,
            epoch=0,
            seed=0,
            current_batch_idx=0
        ):
        """
        Args:
            datasets (list): List of datasets to sample from.
            num_replicas (int): Number of processes (usually set by DDP).
            rank (int): The current process rank.
            shuffle (bool): Whether to shuffle batches each epoch.
        """
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, drop_last=True)
        self.toks_per_batch = toks_per_batch
        self.batch_size = batch_size
        if self.toks_per_batch is not None:
            self.batch_indices = dataset.get_batch_indices(self.toks_per_batch, 
                                                    sort=True,
                                                    shuffle_batch=False,
                                                    )
        elif self.batch_size is not None:
            self.batch_indices = dataset.get_batch_indices_by_count(
                                                    batch_size=self.batch_size, 
                                                    sort=True,
                                                    shuffle_batch=False,
                                                    )
        else:
            raise ValueError("batch_size and toks_per_batch can't be none at the same time.")
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.epoch = epoch
        self.drop_last = drop_last
        self.seed = seed
        # Ensure batches are evenly split
        self.total_batches = len(self.batch_indices)
        if self.drop_last and len(self.batch_indices) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples_per_rank = math.ceil(
                (len(self.batch_indices) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples_per_rank = math.ceil(len(self.batch_indices) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples_per_rank * self.num_replicas
        self.current_batch_idx = current_batch_idx
    
    def reinitialize(self, epoch=None, current_batch_idx=0, seed=None):
        if seed is None:
            seed = self.seed
        if epoch is None:
            epoch = self.epoch
        self.__init__(
            self.dataset, 
            toks_per_batch=self.toks_per_batch,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            epoch=epoch,
            seed=seed,
            current_batch_idx=current_batch_idx
        )
    
    def state_dict(self):
        r"""
        Returns:
            dict: State dictionary containing the epoch and seed.
        """
        return {"epoch": self.epoch, "seed": self.seed, 
                "current_batch_idx": self.current_batch_idx,
                "toks_per_batch": self.toks_per_batch,
                "batch_size": self.batch_size,
                "shuffle": self.shuffle,
                "drop_last": self.drop_last}
    
    def load_state_dict(self, state_dict):
        self.reinitialize(
            epoch=state_dict["epoch"],
            current_batch_idx=state_dict["current_batch_idx"],
            seed=state_dict["seed"],
        )
        self.epoch = state_dict["epoch"]
        self.seed = state_dict["seed"]
        print(f"Sampler reinitialized to epoch {self.epoch}, "
              f"current_batch_idx {self.current_batch_idx}, "
              f"seed {self.seed}")

    def __iter__(self):
        indices = list(range(self.total_batches))

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # Shuffle indices
            indices = torch.randperm(len(indices), generator=g).tolist()
        
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # Split batches across GPUs
        indices = indices[self.rank:self.total_size:self.num_replicas]
        for i in indices[self.current_batch_idx:]:
            self.current_batch_idx += 1
            if self.current_batch_idx > len(indices):
                raise RuntimeError("Reached the end of the sampler for this epoch.")
            yield self.batch_indices[i]
        

    def __len__(self):
        return int(self.num_samples_per_rank)  # Number of batches per GPU
    
    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class DataLoaderWrapper(DataLoader):
    """A wrapper for DataLoader to handle distributed training and batch sampling."""

    def state_dict(self):
        return self.batch_sampler.state_dict()

    def load_state_dict(self, state_dict):
        self.batch_sampler.load_state_dict(state_dict)

def esm_tokenize(sequence, tokenizer):
    assert isinstance(sequence, list), "Input sequence must be a list."
    pad = tokenizer.pad_token_id
    assert pad is not None
    return stack_variable_length_tensors(
        [
            encoding.tokenize_sequence(x, tokenizer, add_special_tokens=True)
            for x in sequence
        ],
        constant_value=pad,
    )


class ConcatenatedAlphabet(Alphabet):
    """Concatenated alphabet for multiple alphabets."""

    def __init__(self, alphabets: Sequence[Alphabet]):
        self.alphabets = alphabets
        self.idx_to_tok = {}
        for alphabet in alphabets:
            for idx, tok in alphabet.idx_to_tok.items():
                if idx in self.idx_to_tok:
                    raise ValueError(f"Duplicate index {idx} found in concatenated alphabets.")
                self.idx_to_tok[idx] = tok
    
    def decode(self, idx: int) -> str:
        """Decode an index to a token."""
        return self.idx_to_tok.get(idx, None)


class DecodingController:

    def __init__(
            self, 
            batch_size, 
            max_length, 
            device, 
            protein_sequences,
            vocab_size = None,
            expected_utr_5_length=None,
            expected_utr_3_length=None
        ):
        self.global_tokenizer = Alphabet.initialize_for_global(offset=0)
        self.utr_offset = len(self.global_tokenizer.all_toks)
        self.utr_alphabet = Alphabet.initialize_for_utr(offset=self.utr_offset)
        self.codon_offset = self.utr_offset + len(self.utr_alphabet.all_toks)
        self.codon_alphabet = Alphabet.initialize_for_codon(offset=self.codon_offset)
        self.decoding_stage_table = {
            "UTR_5_start": 0,
            "UTR_5": 1,
            "CDS_start": 2,
            "CDS": 3,
            "CDS_end": 4,
            "UTR_3_start": 5,
            "UTR_3": 6,
            "WrapUp": 7,
            "STOP": 8
        }
        for idx, aminoacid in enumerate(list(aminoacid_to_codon.keys()), \
                                        start=len(self.decoding_stage_table)):
            
            self.decoding_stage_table["aa_"+aminoacid] = idx
        self.concat_tokenizer = ConcatenatedAlphabet(
            [
                self.global_tokenizer, self.codon_alphabet, self.utr_alphabet
            ]
        ) 
        self.max_length = max_length
        self.batch_size = batch_size
        self.status = torch.zeros(batch_size, dtype=torch.int64, device=device)
        self.sequences = []
        self.sequence_lengths = len(protein_sequences) + 2
        if vocab_size is None:
            vocab_size = sum(len(x) for x in [
                self.global_tokenizer,
                self.utr_alphabet,
                self.codon_alphabet,
                # self.utr_alphabet,
            ])
        self.vocab_size = vocab_size
        self.num_stages = len(self.decoding_stage_table)
        self.device = device
        self.stage_token_mask = self.create_reference_stage_token_mask()
        # protein conditioned decoding
        assert len(protein_sequences) == batch_size or len(protein_sequences) == 1, \
            "Invalid protein sequences length"
        if len(protein_sequences) == 1:
            self.protein_sequences = [protein_sequences[0]+"*"] * batch_size
        else:
            self.protein_sequences = [x+"*" for x in protein_sequences]
        self.decoding_aa_positions = torch.zeros(batch_size, dtype=torch.int64, device=device) - 1
        self.protein_length = torch.tensor([len(x) for x in self.protein_sequences], dtype=torch.int64, device=device)
        self.expected_utr_5_length = expected_utr_5_length
        if expected_utr_5_length is not None:
            self.utr_5_length_mean = torch.tensor(expected_utr_5_length, device=device)
            self.utr_5_length_std = torch.tensor(expected_utr_5_length, device=device) * 0.1
            self.utr_5_length_dist = Normal(self.utr_5_length_mean, self.utr_5_length_std)
            self.utr_5_length = torch.zeros(batch_size, dtype=torch.int64, device=device)
        self.expected_utr_3_length = expected_utr_3_length
        if expected_utr_3_length is not None:
            self.utr_3_length_mean = torch.tensor(expected_utr_3_length, device=device)
            self.utr_3_length_std = torch.tensor(expected_utr_3_length, device=device) * 0.1
            self.utr_3_length_dist = Normal(self.utr_3_length_mean, self.utr_3_length_std)
            self.utr_3_length = torch.zeros(batch_size, dtype=torch.int64, device=device)
        self._update_count = 0

    def create_reference_stage_token_mask(self):
        mask = torch.zeros((self.num_stages, self.vocab_size), dtype=torch.bool, device=self.device)
        for stage, idx in self.decoding_stage_table.items():
            valid_tokens = self.get_valid_token_idx(stage)
            if valid_tokens is None:
                mask[idx, :] = True
            else:
                mask[idx, valid_tokens] = True
        return mask
    
    def create_status_token_mask(self):
        mask = self.stage_token_mask[self.status].clone()
        protein_mask = torch.zeros_like(mask, dtype=torch.bool)
        is_within_cds = (self.status == self.decoding_stage_table["CDS"])
        aa_list = []
        for i, protein_sequence in enumerate(self.protein_sequences):
            if is_within_cds[i]:
                aa = protein_sequence[self.decoding_aa_positions[i]]
                aa_list.append(aa)
                if aa in aminoacid_to_codon:
                    valid_tokens = self.get_valid_token_idx("aa_" + aa)
                    if valid_tokens is not None:
                        protein_mask[i, valid_tokens] = True
                else:
                    if aa.upper() == 'U':
                        valid_tokens_C = self.get_valid_token_idx("aa_C")
                        valid_tokens_S = self.get_valid_token_idx("aa_*")
                        if valid_tokens_C is not None:
                            protein_mask[i, valid_tokens_C] = True
                        if valid_tokens_S is not None:
                            protein_mask[i, valid_tokens_S] = True
                    else:
                        raise RuntimeError(f"Unknown aminoacid {aa} in protein sequence.")
        mask = torch.where(
            is_within_cds.view(-1, 1),
            protein_mask,
            mask
        )
        return mask

    def pad_tokens(self, tokens):
        within_stop = (self.status == self.decoding_stage_table["STOP"]).view(-1, 1)
        not_eos = tokens != self.global_tokenizer.eos_idx
        tokens[within_stop & not_eos] = self.global_tokenizer.padding_idx
        return tokens

    def update(self, tokens):
        self.sequences.append(tokens)
        self.sequence_lengths += tokens.shape[1]
        self.update_status(tokens)
        if self.expected_utr_5_length is not None:
            self.utr_5_length += (self.status == self.decoding_stage_table["UTR_5"]).long()
        if self.expected_utr_3_length is not None:
            self.utr_3_length += (self.status == self.decoding_stage_table["UTR_3"]).long()

    def show_sequences(self):
        return torch.concatenate(self.sequences, dim=1)
    
    def update_status(self, tokens):
        if tokens.dim() > 1 and tokens.shape[1] > 1:
            # it means we are processing prompt
            # so we only take the last token
            tokens = tokens[:, -1]  
            # it also means we already in the middle of decoding
            # so we initialize the status based on previous status
            self.status = torch.zeros(tokens.shape[0], dtype=torch.int64, device=tokens.device) + \
                self.decoding_stage_table["UTR_5"]
        tokens = tokens.squeeze(1) if tokens.dim() == 2 else tokens
        self.status = torch.where(
            tokens == self.global_tokenizer.get_idx("<utr_5_bos>"),
            self.decoding_stage_table["UTR_5"],
            self.status
        )
        self.status = torch.where(
            tokens == self.global_tokenizer.get_idx("<utr_5_eos>"),
            self.decoding_stage_table["CDS_start"],
            self.status
        )
        self.status = torch.where(
            tokens == self.global_tokenizer.get_idx("<cds_bos>"),
            self.decoding_stage_table["CDS"],
            self.status
        )
        self.status = torch.where(
            (self.status == self.decoding_stage_table["CDS"]) & \
                (self.decoding_aa_positions >= self.protein_length - 1),
            self.decoding_stage_table["CDS_end"],
            self.status
        )
        self.status = torch.where(
            tokens == self.global_tokenizer.get_idx("<cds_eos>"),
            self.decoding_stage_table["UTR_3_start"],
            self.status
        )
        self.status = torch.where(
            tokens == self.global_tokenizer.get_idx("<utr_3_bos>"),
            self.decoding_stage_table["UTR_3"],
            self.status
        )
        self.status = torch.where(
            tokens == self.global_tokenizer.get_idx("<utr_3_eos>"),
            self.decoding_stage_table["WrapUp"],
            self.status
        )
        self.status = torch.where(
            tokens == self.global_tokenizer.get_idx("<eos>"),
            self.decoding_stage_table["STOP"],
            self.status
        )
        self.decoding_aa_positions[self.status == self.decoding_stage_table["CDS"]] += 1

    def should_stop(self):
        return torch.all(
            self.status == self.decoding_stage_table["STOP"]
        ) or self.sequence_lengths >= self.max_length

    def modify_logits(self, logits):
        mask = self.create_status_token_mask()
        logits = modify_logits_for_partial_generation(logits, mask)
        if (self.status == self.decoding_stage_table["UTR_5"]).any() and self.expected_utr_5_length is not None:
            modifying_prob = self.utr_5_length_dist.cdf(self.utr_5_length.float())
            logits = self.modify_logits_for_sequence_length_control(
                logits, 
                modifying_prob,
                eos_token_idx=self.global_tokenizer.get_idx("<utr_5_eos>"),
                valid_token_mask=mask,
                status_mask=(self.status == self.decoding_stage_table["UTR_5"])
            )
        if (self.status == self.decoding_stage_table["UTR_3"]).any() and self.expected_utr_3_length is not None:
            modifying_prob = self.utr_3_length_dist.cdf(self.utr_3_length.float())
            logits = self.modify_logits_for_sequence_length_control(
                logits, 
                modifying_prob,
                eos_token_idx=self.global_tokenizer.get_idx("<utr_3_eos>"),
                valid_token_mask=mask,
                status_mask=(self.status == self.decoding_stage_table["UTR_3"])
            )
        return logits

    def modify_logits_for_sequence_length_control(self, logits, modifying_prob, eos_token_idx, valid_token_mask, status_mask):
        # modifying_prob = gaussian_function(current_length/expected_length, 1.0, 0.05)
        modifying_mask = torch.rand(self.batch_size, device=self.device) < modifying_prob
        modifying_mask = (modifying_mask & status_mask)
        if not torch.any(modifying_mask):
            return logits
        # Set all valid token logits to -inf, except eos token
        logits[(modifying_mask.unsqueeze(-1) & valid_token_mask)] = -float("inf")
        logits[modifying_mask, eos_token_idx] = 0.0
        return logits

    def get_valid_token_idx(self, stage):
        """Get valid token indices for a given stage."""
        if stage == "UTR_5_start":
            return self._get_utr_5_start_token_ids()
        elif stage == "UTR_5":
            return self._get_utr_5_token_idx()
        elif stage == "CDS_start":
            return self._get_cds_start_token_idx()
        elif stage == "CDS":
            return None  # This should be handled in the sampling logic
        elif stage.startswith("aa_"):
            return self._get_cds_token_idx(stage[3:])
        elif stage == "CDS_end":
            return self._get_cds_end_token_idx()
        elif stage == "UTR_3_start":
            return self._get_utr_3_start_token_idx()
        elif stage == "UTR_3":
            return self._get_utr_3_token_idx()
        elif stage == "WrapUp":
            return [self.global_tokenizer.get_idx("<eos>")]
        elif stage == "STOP":
            return [self.global_tokenizer.get_idx("<pad>")]
        else:
            raise ValueError(f"Unknown stage: {stage}")
    
    def _get_utr_5_start_token_ids(self):
        return [self.global_tokenizer.get_idx("<utr_5_bos>"),]

    def _get_utr_5_token_idx(self):
        tokens = []
        tokens.append(self.global_tokenizer.get_idx("<utr_5_eos>"))
        tokens.append(self.utr_alphabet.get_idx("A"))
        tokens.append(self.utr_alphabet.get_idx("U"))
        tokens.append(self.utr_alphabet.get_idx("C"))
        tokens.append(self.utr_alphabet.get_idx("G"))
        # tokens.append(self.utr_alphabet.get_idx("I"))
        return tokens
    
    def _get_cds_start_token_idx(self):
        return [self.global_tokenizer.get_idx("<cds_bos>"),]
    
    def _get_cds_token_idx(self, aminoacid):
        """Get valid token indices for CDS stage."""
        if aminoacid not in aminoacid_to_codon:
            raise ValueError(f"Unknown amino acid: {aminoacid}")
        codons = aminoacid_to_codon[aminoacid]
        return [self.codon_alphabet.get_idx(codon) for codon in codons]
    
    def _get_cds_end_token_idx(self):
        return [self.global_tokenizer.get_idx("<cds_eos>"),]

    def _get_utr_3_start_token_idx(self):
        return [self.global_tokenizer.get_idx("<utr_3_bos>"),]
    
    def _get_utr_3_token_idx(self):
        tokens = []
        tokens.append(self.global_tokenizer.get_idx("<utr_3_eos>"))
        tokens.append(self.utr_alphabet.get_idx("A"))
        tokens.append(self.utr_alphabet.get_idx("U"))
        tokens.append(self.utr_alphabet.get_idx("C"))
        tokens.append(self.utr_alphabet.get_idx("G"))
        # tokens.append(self.utr_alphabet.get_idx("I"))
        return tokens
    


def modify_logits_for_partial_generation(logits, valid_token_mask):
    """Modify logits for partial generation.
    Arguments:
        logits: Tensor of shape (batch_size, vocab_size)
        valid_tokens: Tensor of shape (batch_size, seq_len) with valid tokens
    """
    # Set logits for invalid tokens to -inf
    logits[~valid_token_mask] = float("-inf")
    return logits

def gaussian_function(x, mu, sigma):
    return (1 / (sigma * math.sqrt(2 * math.pi))) * torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

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


def tokenize_utr_sequence(
        utr_sequence,
        utr_tokenizer,
        mask_token_id=-100,
    ):
    utr_sequence = _handle_special_nucleotides(utr_sequence, max_length=-1, replace_T=True)
    utr_tensor = torch.tensor(
        [utr_tokenizer.get_idx(utr_sequence[i]) if utr_sequence[i] != "_" else mask_token_id for i in range(0, len(utr_sequence))], 
        dtype=torch.int64
    )
    return utr_tensor

def tokenize_cds_sequence(
        cds_sequence,
        codon_tokenizer,
    ):
    cds_sequence = _handle_special_nucleotides(cds_sequence, max_length=-1, replace_T=False)
    codon_list = [cds_sequence[i:i+3] for i in range(0, len(cds_sequence), 3)]
    codon_tensor   = torch.tensor(
        [codon_tokenizer.get_idx(codon_list[i]) for i in range(0, len(codon_list))], dtype=torch.int64
    ) 
    return codon_tensor