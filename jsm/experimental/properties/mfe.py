# linearfold_mfe.py
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np

LINEARFOLD = None  # set once per worker


def _init_worker(linearfold_path: str):
    """Initializer runs once per process to avoid re-resolving the binary."""
    global LINEARFOLD
    LINEARFOLD = linearfold_path


def _mfe_linearfold(sequence: str) -> float:
    """Call LinearFold on a single RNA sequence and parse the MFE."""
    proc = subprocess.run(
        [LINEARFOLD],
        input=sequence + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    out = proc.stdout.strip()
    try:
        # Typical LinearFold line ends with '( -xx.xx )'
        mfe_str = out.rsplit("(", 1)[-1].split(")")[0]
        return float(mfe_str)
    except Exception as e:
        raise RuntimeError(f"Failed to parse LinearFold output: {out!r}") from e


def _compute_one(seq: str) -> Tuple[str, float]:
    """Worker: compute MFE for a single sequence string."""
    full_seq = seq.replace("T", "U")  # DNA -> RNA just in case
    mfe = _mfe_linearfold(full_seq)
    return seq, mfe


def _resolve_linearfold_path(linearfold_path: Optional[str]) -> str:
    """Ensure LinearFold exists and is runnable; return the path used."""
    lf = linearfold_path or "linearfold"
    try:
        subprocess.run([lf, "--help"], capture_output=True, check=True)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Could not find LinearFold executable at {lf!r}. "
            "Ensure it's installed and on your PATH, or pass an explicit path."
        ) from e
    return lf


def compute_mfe(
    sequences: Iterable[str],
    linearfold_path: Optional[str] = None,
    n_workers: Optional[int] = None,
    chunksize: int = 50,
    return_numpy: bool = True,
    continue_on_error: bool = False,
) -> Union[np.ndarray, List[Tuple[str, float]]]:
    """
    Compute MFEs for an iterable of sequences using LinearFold with multiprocessing.

    Parameters
    ----------
    sequences : iterable of str
        RNA/DNA sequences (DNA 'T' will be converted to RNA 'U').
    linearfold_path : str or None
        Path to the LinearFold binary. If None, assumes 'linearfold' in PATH.
    n_workers : int or None
        Number of worker processes. Defaults to min(os.cpu_count(), 48) if None.
    chunksize : int
        Chunk size for executor.map to reduce overhead (tune for your sequence length).
    return_numpy : bool
        If True, returns a NumPy array of MFEs in the same order as input.
        If False, returns a list of (sequence, mfe) tuples.
    continue_on_error : bool
        If True, errors on individual sequences yield NaN instead of raising.

    Returns
    -------
    np.ndarray or list of (str, float)
        MFEs aligned with the input order (no reordering).
    """
    lf = _resolve_linearfold_path(linearfold_path)

    # Normalize worker count
    max_workers = min(max(os.cpu_count() or 1, 1), 48) if n_workers is None else n_workers
    if max_workers < 1:
        max_workers = 1

    # If sequences is a generator, we’ll want to materialize it once to preserve order.
    seq_list = list(sequences)

    results: List[Tuple[str, float]] = []
    if not seq_list:
        return np.array([], dtype=float) if return_numpy else results

    def _safe_compute(seq: str) -> Tuple[str, float]:
        if not continue_on_error:
            return _compute_one(seq)
        try:
            return _compute_one(seq)
        except Exception:
            return (seq, float("nan"))

    with ProcessPoolExecutor(
        max_workers=max_workers, initializer=_init_worker, initargs=(lf,)
    ) as ex:
        # executor.map preserves input order
        for seq, mfe in ex.map(_safe_compute, seq_list, chunksize=chunksize):
            results.append((seq, mfe))

    if return_numpy:
        return np.array([mfe for _, mfe in results], dtype=float)
    return results


def compute_mfe_iter(
    sequences: Iterable[str],
    linearfold_path: Optional[str] = None,
    n_workers: Optional[int] = None,
    chunksize: int = 50,
    continue_on_error: bool = False,
) -> Iterator[Tuple[str, float]]:
    """
    Generator variant that yields (sequence, mfe) as results arrive (still in input order).

    Useful if you want to stream results without holding everything in memory.
    """
    lf = _resolve_linearfold_path(linearfold_path)
    seq_list = list(sequences)
    if not seq_list:
        return iter(())  # empty iterator

    def _safe_compute(seq: str) -> Tuple[str, float]:
        if not continue_on_error:
            return _compute_one(seq)
        try:
            return _compute_one(seq)
        except Exception:
            return (seq, float("nan"))

    max_workers = min(max(os.cpu_count() or 1, 1), 48) if n_workers is None else n_workers
    if max_workers < 1:
        max_workers = 1

    with ProcessPoolExecutor(
        max_workers=max_workers, initializer=_init_worker, initargs=(lf,)
    ) as ex:
        for pair in ex.map(_safe_compute, seq_list, chunksize=chunksize):
            yield pair


# --- Optional tiny CLI to keep parity with your original main() ---
def _main_cli():
    """
    Example CLI:
        python -m linearfold_mfe input.txt > mfes.txt

    Where input.txt contains one sequence per line.
    """
    import argparse
    import sys
    import yaml
    
    parser = argparse.ArgumentParser(description="Compute MFE of RNA sequences using LinearFold.")
    parser.add_argument("input", type=str, help="Input text file with one sequence per line.")
    parser.add_argument("output", type=str, help="Output yaml file to write MFE results.")
    args = parser.parse_args()

    infile = Path(args.input)
    if not infile.exists():
        print(f"Input file not found: {infile}", file=sys.stderr)
        sys.exit(1)

    seqs = [line.strip() for line in infile.read_text().splitlines() if line.strip()]
    mfes = compute_mfe(seqs, return_numpy=True)
    
    with open(args.output, 'w') as f:
        yaml.dump(list(mfes), f)
    

if __name__ == "__main__":
    _main_cli()
