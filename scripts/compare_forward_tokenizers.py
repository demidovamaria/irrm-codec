"""Compare char vs WordPiece tokenizers on the forward task (CDR3 -> TCRemP embedding).

Runs training in-process (reusing irrm_codec.train_forward.run_epoch and
irrm_codec.batch_nocache.prepare_cached_training_data) for each (tokenizer config, seed)
pair, times training and inference wall-clock, and writes per-run results plus a
mean +/- std summary across seeds.

Usage:
    python scripts/compare_forward_tokenizers.py \
        --airr-path data/processed/trb_1M.tsv \
        --embeddings-path data/embeddings/trb_1M.parquet \
        --locus beta --max-len 40 \
        --output-root artifacts/compare_forward_tokenizers \
        --seeds 1 42 777 \
        --configs char \
                  wordpiece:artifacts/tokenizers_1M/wordpiece_vocab_5000/tokenizer.json \
                  wordpiece:artifacts/tokenizers_1M/wordpiece_vocab_10000/tokenizer.json \
                  wordpiece:artifacts/tokenizers_1M/wordpiece_vocab_20000/tokenizer.json \
                  wordpiece:artifacts/tokenizers_1M/wordpiece_vocab_30000/tokenizer.json \
                  wordpiece:artifacts/tokenizers_1M/wordpiece_vocab_40000/tokenizer.json \
                  wordpiece:artifacts/tokenizers_1M/wordpiece_vocab_50000/tokenizer.json

Each --configs entry is "char" or "wordpiece:<path-to-tokenizer.json>". Output:
  <output-root>/raw_results.csv       one row per (config, seed) run
  <output-root>/summary_table.md      one row per config, mean +/- std over seeds
"""
import argparse
import csv
import statistics
import time
from pathlib import Path

import torch

from irrm_codec.batch_nocache import prepare_cached_training_data
from irrm_codec.datasets import collate_forward
from irrm_codec.forward_model import ForwardModel
from irrm_codec.tokenizer_cli import resolve_tokenizer
from irrm_codec.train_forward import run_epoch
from irrm_codec.utils import choose_device, save_json, set_seed, setup_logging

RESULT_FIELDS = [
    "tokenizer", "vocab_size", "max_len", "seed",
    "test_loss", "test_cosine", "best_val_loss", "epochs_to_best_val",
    "train_time_sec", "inference_time_sec", "num_parameters", "tokenizer_path",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark char vs WordPiece on the forward task.")
    parser.add_argument("--airr-path", required=True)
    parser.add_argument("--embeddings-path", required=True)
    parser.add_argument("--locus", default="alpha")
    parser.add_argument("--clone-id-col", default="clone_id")
    parser.add_argument("--embedding-column", default="tcremp_emb")
    parser.add_argument("--max-len", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--inference-repeats", type=int, default=3, help="Repeats to average inference wall-clock over.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 42, 777])
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="'char' or 'wordpiece:<path-to-tokenizer.json>', one per tokenizer to compare.",
    )
    parser.add_argument("--output-root", default="artifacts/compare_forward_tokenizers")
    return parser.parse_args()


def _tokenizer_args_from_spec(spec):
    """'char' / 'wordpiece:<path>' -> the (tokenizer_type, tokenizer_path) resolve_tokenizer needs."""
    if spec == "char":
        return argparse.Namespace(tokenizer_type="char", tokenizer_path=None, vocab_size=None)
    if spec.startswith("wordpiece:"):
        path = spec.split(":", 1)[1]
        return argparse.Namespace(tokenizer_type="wordpiece", tokenizer_path=path, vocab_size=None)
    raise ValueError(f"Unrecognized --configs entry {spec!r}; expected 'char' or 'wordpiece:<path>'.")


def _count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def _benchmark_inference(model, test_loader, device, repeats):
    """Average wall-clock seconds for one full forward pass over the test set."""
    model.eval()
    with torch.no_grad():
        for tokens, mask, _target, _lengths in test_loader:  # warm-up, excluded from timing
            model(tokens.to(device), mask.to(device))
            break
        durations = []
        for _ in range(repeats):
            start = time.perf_counter()
            for tokens, mask, _target, _lengths in test_loader:
                model(tokens.to(device), mask.to(device))
            durations.append(time.perf_counter() - start)
    return statistics.mean(durations)


def run_one(spec, seed, args, logger):
    """Train one (tokenizer config, seed) combination in-process and return its metrics row."""
    set_seed(seed)
    device = choose_device()

    tokenizer_args = _tokenizer_args_from_spec(spec)
    encode_fn, vocab_size, tokenizer_info = resolve_tokenizer(tokenizer_args, logger)

    data_args = argparse.Namespace(
        airr_path=args.airr_path,
        embeddings_path=args.embeddings_path,
        locus=args.locus,
        clone_id_col=args.clone_id_col,
        embedding_column=args.embedding_column,
        max_len=args.max_len,
        batch_size=args.batch_size,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        seed=seed,
        num_workers=args.num_workers,
    )
    prepared = prepare_cached_training_data(
        data_args, logger, task="forward", collate_fn=collate_forward, encode_fn=encode_fn
    )
    model = ForwardModel(
        vocab_size=vocab_size,
        output_dim=prepared["merge_stats"]["embedding_dim"],
        max_len=args.max_len,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    epochs_to_best_val = None
    train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        run_epoch(model, prepared["train_loader"], optimizer, device, "train", epoch, args.epochs, args.log_interval, False)
        val_metrics = run_epoch(model, prepared["val_loader"], None, device, "val", epoch, args.epochs, args.log_interval, False)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_to_best_val = epoch
    train_time_sec = time.perf_counter() - train_start

    test_metrics = run_epoch(model, prepared["test_loader"], None, device, "test", args.epochs, args.epochs, args.log_interval, False)
    inference_time_sec = _benchmark_inference(model, prepared["test_loader"], device, args.inference_repeats)

    return {
        "tokenizer": tokenizer_info["tokenizer_type"],
        "vocab_size": tokenizer_info["vocab_size"],
        "max_len": args.max_len,
        "seed": seed,
        "test_loss": test_metrics["loss"],
        "test_cosine": test_metrics["cosine"],
        "best_val_loss": best_val_loss,
        "epochs_to_best_val": epochs_to_best_val,
        "train_time_sec": train_time_sec,
        "inference_time_sec": inference_time_sec,
        "num_parameters": _count_parameters(model),
        "tokenizer_path": tokenizer_info["tokenizer_path"],
    }


def _mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def summarize(rows):
    """One aggregated row per (tokenizer, vocab_size), mean +/- std across seeds."""
    groups = {}
    for row in rows:
        key = (row["tokenizer"], row["vocab_size"])
        groups.setdefault(key, []).append(row)

    summary = []
    for (tokenizer, vocab_size), group_rows in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        summary.append({
            "tokenizer": tokenizer,
            "vocab_size": vocab_size,
            "max_len": group_rows[0]["max_len"],
            "n_seeds": len(group_rows),
            "test_loss": _mean_std([r["test_loss"] for r in group_rows]),
            "test_cosine": _mean_std([r["test_cosine"] for r in group_rows]),
            "best_val_loss": _mean_std([r["best_val_loss"] for r in group_rows]),
            "epochs_to_best_val": _mean_std([r["epochs_to_best_val"] for r in group_rows]),
            "train_time_sec": _mean_std([r["train_time_sec"] for r in group_rows]),
            "inference_time_sec": _mean_std([r["inference_time_sec"] for r in group_rows]),
            "num_parameters": group_rows[0]["num_parameters"],
        })
    return summary


def _fmt(mean_std, precision=4):
    mean, std = mean_std
    return f"{mean:.{precision}f} +/- {std:.{precision}f}"


def write_markdown_table(summary, path):
    lines = [
        "| tokenizer | vocab_size | max_len | test_loss | test_cosine | best_val_loss | "
        "epochs_to_best_val | train_time_sec | inference_time_sec | num_parameters | notes |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in summary:
        notes = f"n={row['n_seeds']} seeds"
        lines.append(
            f"| {row['tokenizer']} | {row['vocab_size']} | {row['max_len']} | "
            f"{_fmt(row['test_loss'])} | {_fmt(row['test_cosine'])} | {_fmt(row['best_val_loss'])} | "
            f"{_fmt(row['epochs_to_best_val'], precision=1)} | {_fmt(row['train_time_sec'], precision=1)} | "
            f"{_fmt(row['inference_time_sec'], precision=3)} | {row['num_parameters']} | {notes} |"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_root / "benchmark.log")

    rows = []
    for spec in args.configs:
        for seed in args.seeds:
            logger.info("running config=%s seed=%d", spec, seed)
            row = run_one(spec, seed, args, logger)
            rows.append(row)
            logger.info("done config=%s seed=%d test_loss=%.4f test_cosine=%.4f", spec, seed, row["test_loss"], row["test_cosine"])

    with open(output_root / "raw_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    save_json(output_root / "summary.json", summary)
    write_markdown_table(summary, output_root / "summary_table.md")
    logger.info("wrote raw_results.csv, summary.json, summary_table.md to %s", output_root)


if __name__ == "__main__":
    main()