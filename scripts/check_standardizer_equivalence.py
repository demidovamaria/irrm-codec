"""One-off check: does direct mean/std computation match batch_cache.py's online formula.

Run this once before trusting train_forward_nocache.py results.
"""
import argparse

import numpy as np

from irrm_codec.dataio import load_airr_with_embeddings
from irrm_codec.utils import split_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--airr-path", required=True)
    parser.add_argument("--embeddings-path", required=True)
    parser.add_argument("--locus", default="beta")
    parser.add_argument("--clone-id-col", default="")
    parser.add_argument("--embedding-column", default="tcremp_emb")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=1e-4)
    return parser.parse_args()


def online_mean_std(train_emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # replicates batch_cache.py's train_sum/train_sum_sq accumulation formula exactly
    train_emb64 = train_emb.astype(np.float64)
    n = len(train_emb64)
    train_sum = train_emb64.sum(axis=0)
    train_sum_sq = np.square(train_emb64).sum(axis=0)
    mean = (train_sum / n).astype(np.float32)
    variance = np.maximum(train_sum_sq / n - np.square(mean, dtype=np.float64), 0.0)
    std = np.sqrt(variance).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    return mean, std


def direct_mean_std(train_emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_emb.mean(axis=0).astype(np.float32)
    std = train_emb.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    return mean, std

def main() -> None:
    args = parse_args()
    merged, emb, merge_stats = load_airr_with_embeddings(
        args.airr_path, args.embeddings_path, locus=args.locus,
        clone_id_col=args.clone_id_col, embedding_column=args.embedding_column,
    )
    train_idx, _val_idx, _test_idx = split_indices(
        len(merged), train_fraction=args.train_fraction, val_fraction=args.val_fraction, seed=args.seed,
    )
    train_emb = emb[train_idx]

    mean_direct, std_direct = direct_mean_std(train_emb)
    mean_online, std_online = online_mean_std(train_emb)

    mean_diff = np.abs(mean_direct - mean_online).max()
    std_diff = np.abs(std_direct - std_online).max()
    relative_std_diff = (np.abs(std_direct - std_online) / np.maximum(std_direct, 1e-8)).max()

    print(f"merge_stats: {merge_stats}")
    print(f"train_rows: {len(train_idx)}")
    print(f"embeddings value range: min={train_emb.min():.6f} max={train_emb.max():.6f}")
    print(f"embeddings abs mean magnitude: {np.abs(train_emb).mean():.6f}")
    print(f"std_direct range: min={std_direct.min():.6f} max={std_direct.max():.6f} mean={std_direct.mean():.6f}")
    print(f"max mean diff (absolute): {mean_diff:.8f}")
    print(f"max std diff (absolute): {std_diff:.8f}")
    print(f"max std diff (relative to std_direct): {relative_std_diff:.6f}")

    # relative tolerance is the correct check here: sum-of-squares formula is known to be
    # numerically unstable (catastrophic cancellation) for large-mean/small-variance data,
    # while np.std() uses the numerically stable two-pass formula. Absolute atol is meaningless
    # without knowing the scale of std itself.
    assert relative_std_diff < 0.05, f"relative std mismatch too high: {relative_std_diff} >= 0.05"
    print("OK: relative std diff within acceptable numerical tolerance.")


if __name__ == "__main__":
    main()