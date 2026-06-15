from jsm.data.utils import Alphabet, aminoacid_to_codon
import torch


class DiffusionDecodingConstraint:
    def __init__(self, ):
        self.global_tokenizer = Alphabet.initialize_for_global(offset=0)
        self.utr_offset = len(self.global_tokenizer.all_toks)
        self.utr_alphabet = Alphabet.initialize_for_utr(offset=self.utr_offset)
        self.codon_offset = self.utr_offset + len(self.utr_alphabet.all_toks)
        self.codon_alphabet = Alphabet.initialize_for_codon(offset=self.codon_offset)
        self.aminoacid_to_codon = aminoacid_to_codon.copy()
        if 'U' not in self.aminoacid_to_codon:
            self.aminoacid_to_codon['U'] = self.aminoacid_to_codon['C']
        if 'X' not in self.aminoacid_to_codon:
            all_codons = []
            for aa in self.aminoacid_to_codon.keys():
                all_codons.extend(self.aminoacid_to_codon[aa])
            all_codons = list(set(all_codons))
            self.aminoacid_to_codon['X'] = all_codons
    
    def get_allowed_token_mask(self, protein_sequence, utr5_length, utr3_length):
        cds_length = len(protein_sequence) + 1  # +1 for the stop codon
        sequence_length = utr5_length + cds_length + utr3_length + 8  # +8 for special tokens
        
        vocab_size = sum([
            len(self.global_tokenizer),
            len(self.utr_alphabet),
            len(self.codon_alphabet),
            len(self.utr_alphabet),
        ])
        allowed_tokens = torch.zeros((sequence_length, vocab_size), dtype=torch.bool)
        
        # special tokens
        allowed_tokens[0, self.global_tokenizer.get_idx("<cls>")] = True
        allowed_tokens[1, self.global_tokenizer.get_idx("<utr_5_bos>")] = True
        allowed_tokens[2+utr5_length, self.global_tokenizer.get_idx("<utr_5_eos>")] = True
        allowed_tokens[3+utr5_length, self.global_tokenizer.get_idx("<cds_bos>")] = True
        allowed_tokens[4+utr5_length+cds_length, self.global_tokenizer.get_idx("<cds_eos>")] = True
        allowed_tokens[5+utr5_length+cds_length, self.global_tokenizer.get_idx("<utr_3_bos>")] = True
        allowed_tokens[6+utr5_length+cds_length+utr3_length, self.global_tokenizer.get_idx("<utr_3_eos>")] = True
        allowed_tokens[7+utr5_length+cds_length+utr3_length, self.global_tokenizer.get_idx("<eos>")] = True
        
        # Allow UTR tokens for the UTR regions
        nucleotide_token_indices = [self.utr_alphabet.get_idx(tok) for tok in ["A", "C", "G", "U"]]
        nucleotide_token_indices = torch.tensor(nucleotide_token_indices, dtype=torch.long)
        allowed_tokens[2:2+utr5_length, nucleotide_token_indices] = True
        allowed_tokens[6+utr5_length+cds_length:6+utr5_length+cds_length+utr3_length, nucleotide_token_indices] = True
        
        # Allow codon tokens for the coding region
        coding_region_offset = 4+utr5_length
        for i in range(cds_length-1):
            amino_acid = protein_sequence[i]
            codons = self.aminoacid_to_codon[amino_acid]
            codon_token_indices = [self.codon_alphabet.get_idx(codon) for codon in codons]
            codon_token_indices = torch.tensor(codon_token_indices, dtype=torch.long)
            allowed_tokens[coding_region_offset + i, codon_token_indices] = True
        # stop codon
        stop_codons = self.aminoacid_to_codon["*"]
        stop_codon_token_indices = [self.codon_alphabet.get_idx(codon) for codon in stop_codons]
        stop_codon_token_indices = torch.tensor(stop_codon_token_indices, dtype=torch.long)
        allowed_tokens[coding_region_offset + cds_length - 1, stop_codon_token_indices] = True
        
        return allowed_tokens
