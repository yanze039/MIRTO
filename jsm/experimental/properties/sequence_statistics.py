# calculate the properties of RNA sequences
from collections import Counter
import math
from .efficient.calculate_mfe import compute_mfe
from typing import Iterable, Iterator, List, Optional, Tuple, Union

def calculate_token_frequency(sequence: str, token: str) -> float:
    """Calculate the frequency of a specific token in the sequence."""
    token_count = sequence.count(token)
    total_tokens = len(sequence)
    return token_count / total_tokens if total_tokens > 0 else 0.0

def calculate_gc_content(sequence: str) -> float:
    """Calculate the GC content of the sequence."""
    g_freq = calculate_token_frequency(sequence, 'G')
    c_freq = calculate_token_frequency(sequence, 'C')
    return g_freq + c_freq

def calculate_u_content(sequence: str) -> float:
    """Calculate the U content of the sequence."""
    return calculate_token_frequency(sequence, 'U')

# def calculate_codon_adaptation_index(cds: str, codon_usage: dict[str, float]) -> float:
#     """Calculate the Codon Adaptation Index (CAI) for a given coding sequence (CDS)."""
#     codons = [cds[i:i+3] for i in range(0, len(cds), 3) if len(cds[i:i+3]) == 3]
#     if not codons:
#         return 0.0
#     weights = [codon_usage.get(codon, 0.01) for codon in codons]  # Avoid log(0)
#     log_weights = [math.log(w) for w in weights]
#     cai = math.exp(sum(log_weights) / len(log_weights))
#     return cai