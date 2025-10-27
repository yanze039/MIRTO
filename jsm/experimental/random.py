# This file helps generate the dummy sequences as the control. We can compare the 
# properties between random sequences and designed sequences to see if the
# designed sequences are significantly different from random sequences.

import random
from jsm.data.utils import codon_table, aminoacid_to_codon


def generate_random_sequence(
        length: int, 
        alphabet: list[str] = ['A', 'C', 'G', 'T']
    ):
    """Generate a random sequence of given length from the specified alphabet."""
    return ''.join(random.choices(alphabet, k=length))


def generate_random_codon(
        amino_acid: str,
    ) -> str:
    """Generate a random codon for a given amino acid."""
    codons = aminoacid_to_codon.get(amino_acid, [])
    if not codons:
        raise ValueError(f"Unknown amino acid: {amino_acid}")
    return random.choice(codons)


def generate_random_cds_for_protein_sequence(
        protein_sequence: str
    ) -> str:
        """Generate a random coding sequence (CDS) for a given protein sequence."""
        return ''.join(generate_random_codon(aa) for aa in protein_sequence)


def generate_random_rna_transcript(
        protein_sequence: str,
        utr_5_length: int,
        utr_3_length: int,
        rna_alphabet: list[str] = ['A', 'C', 'G', 'U'],
        append_stop_codon: bool = True,
    ) -> str:
    """Generate a random RNA transcript with specified UTR lengths and protein sequence."""
    utr_5 = generate_random_sequence(utr_5_length, rna_alphabet)
    if append_stop_codon:
        protein_sequence += codon_table['TAA']
    cds = generate_random_cds_for_protein_sequence(protein_sequence).replace('T', 'U')
    utr_3 = generate_random_sequence(utr_3_length, rna_alphabet)
    return utr_5, cds, utr_3


