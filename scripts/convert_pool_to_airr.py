"""Convert a VDJtools pool file (e.g. human.trb.aa.txt) to a minimal AIRR TSV.

Input columns (tab-separated, header present): count, freq, cdr3nt, cdr3aa, v, d, j,
VEnd, DStart, DEnd, JStart, incidence, convergence, occurrences.

Output columns: junction_aa, v_call, j_call, locus (only what read_airr_table /
prepare_wordpiece_corpus.py actually need). locus is written as the exact string
"beta" to match --locus beta used everywhere downstream (load_filtered_sequences
does an exact df["locus"] == locus comparison, no LOCUS_ALIASES normalization).

Processing order: dedup by cdr3aa (file has many cdr3nt variants collapsing to the
same cdr3aa) -> filter VALID_AA/max_len -> random subsample to --target-size.
Subsampling is done AFTER dedup, over unique sequences, since the pool file is
sorted by descending count - sampling raw rows would bias toward frequent clones.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from irrm_codec.tokenization import VALID_AA
from irrm_codec.utils import setup_logging

# must match --locus beta exactly (no LOCUS_ALIASES normalization downstream)
OUTPUT_LOCUS_VALUE = "beta"
INPUT_COLUMNS = ["cdr3aa", "v", "j"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert VDJtools pool file to minimal AIRR TSV.")
    parser.add_argument("--pool-path", required=True)
    parser.add_argument("--output-path", default="data/processed/trb_1M.tsv")
    parser.add_argument("--target-size", type=int, default=1_000_000)
    parser.add_argument("--max-len", type=int, default=40)  # matches gap_pad_cdr3's target_len default
    parser.add_argument("--seed", type=int, default=42)  # matches split_indices() default elsewhere
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    return parser.parse_args()


def is_valid_sequence(seq: str, max_len: int) -> bool:
    if not seq or len(seq) > max_len:
        return False
    return all(char in VALID_AA for char in seq)


def collect_unique_sequences(
    pool_path: str, chunksize: int, max_len: int, logger: logging.Logger
) -> dict[str, tuple[str, str]]:
    unique_seqs: dict[str, tuple[str, str]] = {}
    rows_scanned = 0
    rows_invalid = 0

    reader = pd.read_csv(pool_path, sep="\t", usecols=INPUT_COLUMNS, dtype=str, chunksize=chunksize)
    for chunk_idx, chunk in enumerate(reader):
        rows_scanned += len(chunk)
        for cdr3aa, v_gene, j_gene in zip(chunk["cdr3aa"], chunk["v"], chunk["j"]):
            if not is_valid_sequence(cdr3aa, max_len):
                rows_invalid += 1
                continue
            # v_call/j_call are never read downstream; first-seen is fine, just needs to be deterministic
            unique_seqs.setdefault(cdr3aa, (v_gene, j_gene))

        logger.info(
            "chunk=%d rows_scanned=%d unique_so_far=%d rows_invalid_so_far=%d",
            chunk_idx, rows_scanned, len(unique_seqs), rows_invalid,
        )

    logger.info(
        "collection done rows_scanned=%d rows_invalid=%d unique_sequences=%d",
        rows_scanned, rows_invalid, len(unique_seqs),
    )
    return unique_seqs


def subsample_sequences(
    unique_seqs: dict[str, tuple[str, str]], target_size: int, seed: int, logger: logging.Logger
) -> list[str]:
    all_seqs = list(unique_seqs.keys())
    if len(all_seqs) <= target_size:
        logger.warning(
            "unique_sequences=%d <= target_size=%d, using all unique sequences without subsampling",
            len(all_seqs), target_size,
        )
        return all_seqs

    # sampling over deduplicated set, not raw rows: file is sorted by descending count,
    # so sampling raw rows would over-represent frequent/public clonotypes
    rng = np.random.default_rng(seed)
    sampled_idx = rng.choice(len(all_seqs), size=target_size, replace=False)
    return [all_seqs[i] for i in sampled_idx]


def write_airr_tsv(
    sampled_seqs: list[str], unique_seqs: dict[str, tuple[str, str]], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"junction_aa": seq, "v_call": unique_seqs[seq][0], "j_call": unique_seqs[seq][1], "locus": OUTPUT_LOCUS_VALUE}
        for seq in sampled_seqs
    ]
    pd.DataFrame(rows).to_csv(output_path, sep="\t", index=False)


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path)
    logger = setup_logging(output_path.parent / "convert_pool_to_airr.log")

    unique_seqs = collect_unique_sequences(args.pool_path, args.chunksize, args.max_len, logger)
    if not unique_seqs:
        raise ValueError("No valid sequences found after VALID_AA/max_len filtering.")

    sampled_seqs = subsample_sequences(unique_seqs, args.target_size, args.seed, logger)
    write_airr_tsv(sampled_seqs, unique_seqs, output_path)

    logger.info(
        "wrote AIRR TSV path=%s rows=%d locus=%s",
        output_path, len(sampled_seqs), OUTPUT_LOCUS_VALUE,
    )


if __name__ == "__main__":
    main()