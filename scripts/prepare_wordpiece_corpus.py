
"""Prepare train-only CDR3 corpus for WordPiece tokenizer training.

Reuses read_airr_table + split_indices exactly as batch_cache.py, guaranteeing
identical train/val/test partitioning as the char-baseline run (no data leakage).
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np

from irrm_codec.dataio import read_airr_table
from irrm_codec.tokenization import VALID_AA
from irrm_codec.utils import setup_logging, split_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train-only CDR3 corpus for WordPiece.")
    parser.add_argument("--airr-path", required=True)
    parser.add_argument("--output-dir", default="artifacts/tokenizers")
    parser.add_argument("--locus", default="beta")
    parser.add_argument("--clone-id-col", default="")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_filtered_sequences(
    airr_path: str, locus: str, clone_id_col: str, logger: logging.Logger
) -> tuple[np.ndarray, "np.ndarray | None", int, int]:
    columns = ["junction_aa", "locus"]
    if clone_id_col:
        columns.append(clone_id_col)

    df = read_airr_table(
        airr_path, clone_id_col=clone_id_col, columns=list(dict.fromkeys(columns)), validate=False,
    )
    rows_before_filter = len(df)

    if locus is not None:
        df = df[df["locus"] == locus].reset_index(drop=True)

    rows_after_filter = len(df)
    if rows_after_filter == 0:
        raise ValueError(f"No rows left after filtering locus == {locus!r}.")

    logger.info(
        "loaded AIRR rows_before_filter=%d rows_after_filter=%d locus=%s",
        rows_before_filter, rows_after_filter, locus,
    )

    seqs = df["junction_aa"].astype(str).to_numpy(copy=True)
    clone_ids = df[clone_id_col].to_numpy(copy=True) if clone_id_col and clone_id_col in df.columns else None
    return seqs, clone_ids, rows_before_filter, rows_after_filter


def validate_sequences(seqs: np.ndarray) -> None:
    for seq in seqs:
        if not seq:
            raise ValueError("Found empty junction_aa sequence; cannot build corpus.")
        invalid_chars = sorted({char for char in seq if char not in VALID_AA})
        if invalid_chars:
            raise ValueError(
                f"Sequence {seq!r} contains characters outside VALID_AA: {invalid_chars}. "
                "Resolve this before training WordPiece to keep parity with the char tokenizer."
            )


def write_corpus(seqs: np.ndarray, indices: np.ndarray, corpus_path: Path) -> None:
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with corpus_path.open("w", encoding="utf-8") as handle:
        for idx in indices:
            handle.write(f"{seqs[idx]}\n")


def compute_length_stats(seqs: np.ndarray, indices: np.ndarray) -> dict:
    lengths = np.array([len(seqs[idx]) for idx in indices])
    return {
        "num_sequences": int(len(indices)),
        "min_length": int(lengths.min()),
        "max_length": int(lengths.max()),
        "mean_length": float(lengths.mean()),
        "median_length": float(np.median(lengths)),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    logger = setup_logging(output_dir / "prepare_corpus.log")

    seqs, clone_ids, rows_before_filter, rows_after_filter = load_filtered_sequences(
        args.airr_path, args.locus, args.clone_id_col, logger
    )
    validate_sequences(seqs)
    logger.info("all sequences passed VALID_AA sanity check")

    train_idx, val_idx, test_idx = split_indices(
        len(seqs), train_fraction=args.train_fraction, val_fraction=args.val_fraction, seed=args.seed,
    )
    logger.info("split ready train=%d val=%d test=%d", len(train_idx), len(val_idx), len(test_idx))

    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(seqs)

    corpus_path = output_dir / "train_cdr3.txt"
    write_corpus(seqs, train_idx, corpus_path)
    logger.info("wrote train corpus path=%s rows=%d", corpus_path, len(train_idx))

    split_payload = {
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "val_fraction": args.val_fraction,
        "airr_path": str(args.airr_path),
        "locus": args.locus,
        "rows_before_locus_filter": int(rows_before_filter),
        "rows_after_locus_filter": int(rows_after_filter),
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist(),
    }
    if clone_ids is not None:
        split_payload["train_clone_ids"] = clone_ids[train_idx].tolist()
        split_payload["val_clone_ids"] = clone_ids[val_idx].tolist()
        split_payload["test_clone_ids"] = clone_ids[test_idx].tolist()

    split_path = output_dir / "split_indices.json"
    split_path.write_text(json.dumps(split_payload, indent=2), encoding="utf-8")
    logger.info("wrote split indices path=%s", split_path)

    stats = compute_length_stats(seqs, train_idx)
    stats_path = output_dir / "corpus_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("corpus stats: %s", stats)


if __name__ == "__main__":
    main()
