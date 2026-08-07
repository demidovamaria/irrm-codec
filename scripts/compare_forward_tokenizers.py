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

Each --configs entry is "char" or "wordpiece:<path-to-tokenizer.json>". Optional
--max-token-len N filters every WordPiece vocab down to tokens of <= N amino acids
(char is unaffected) - runs with and without the filter are tracked as distinct configs,
never averaged together. Output:
  <output-root>/raw_results.csv       one row per (config, seed) run
  <output-root>/summary_table.md      one row per config, mean +/- std over seeds
  <output-root>/history/*.json        per-epoch train/val metrics, one file per run
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
from irrm_codec.wordpiece_tokenization import (
    WordpieceEncodeFn,
    filter_vocab_by_max_token_length,
    load_wordpiece_tokenizer,
    wordpiece_vocab_size,
)

DEFAULT_HIDDEN_DIM = 192
DEFAULT_NUM_CONV_BLOCKS = 4

RESULT_FIELDS = [
    "tokenizer", "vocab_size", "max_token_len", "pad_layout", "hidden_dim", "num_conv_blocks", "max_len", "seed",
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
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout used in ForwardModel's conv blocks and MLP head.")
    parser.add_argument("--hidden-dim", type=int, default=192, help="Width of ForwardModel's conv blocks.")
    parser.add_argument(
        "--num-conv-blocks", type=int, default=4,
        help="Depth: number of dilated conv blocks. Dilations are 2**0, 2**1, ..., 2**(n-1) (default 4 -> 1,2,4,8).",
    )
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--inference-repeats", type=int, default=3, help="Repeats to average inference wall-clock over.")
    parser.add_argument(
        "--max-token-len",
        type=int,
        default=None,
        help="Filter each WordPiece tokenizer's vocab down to tokens of at most this many "
        "amino acids (special tokens always kept) before training. Default: no filtering "
        "(the original vocab is used as-is). Does not touch tokenizer.json files or char "
        "configs - filtering happens per-run, in this script only, not in tokenizer training.",
    )
    parser.add_argument(
        "--pad-layout",
        choices=["end", "anchored"],
        default="end",
        help="'end' (default): pad after the real tokens, as before. 'anchored': keep the "
        "first --left-anchor and last --right-anchor tokens at fixed buffer positions and "
        "put all padding between them, so the (conserved) sequence ends land on the same "
        "position across every example. WordPiece configs only - char is unaffected.",
    )
    parser.add_argument("--left-anchor", type=int, default=1, help="Only used with --pad-layout anchored.")
    parser.add_argument("--right-anchor", type=int, default=1, help="Only used with --pad-layout anchored.")
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
    needs_custom_handling = tokenizer_args.tokenizer_type == "wordpiece" and (
        args.max_token_len is not None or args.pad_layout != "end"
    )
    if needs_custom_handling:
        tokenizer = load_wordpiece_tokenizer(tokenizer_args.tokenizer_path)
        original_vocab_size = wordpiece_vocab_size(tokenizer)
        if args.max_token_len is not None:
            tokenizer = filter_vocab_by_max_token_length(tokenizer, args.max_token_len)
        vocab_size = wordpiece_vocab_size(tokenizer)
        encode_fn = WordpieceEncodeFn(
            tokenizer, pad_layout=args.pad_layout, left_anchor=args.left_anchor, right_anchor=args.right_anchor
        )
        tokenizer_info = {
            "tokenizer_type": "wordpiece",
            "tokenizer_path": tokenizer_args.tokenizer_path,
            "vocab_size": vocab_size,
        }
        logger.info(
            "tokenizer_type=wordpiece vocab_size=%d (from %d, max_token_len=%s) pad_layout=%s tokenizer_path=%s",
            vocab_size, original_vocab_size, args.max_token_len, args.pad_layout, tokenizer_args.tokenizer_path,
        )
    else:
        # char, or wordpiece with every new option left at default: unchanged, shared with train_forward.py/train_inverse.py
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
        dropout=args.dropout,
        hidden_dim=args.hidden_dim,
        dilations=tuple(2**i for i in range(args.num_conv_blocks)),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    epochs_to_best_val = None
    history = []
    train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, prepared["train_loader"], optimizer, device, "train", epoch, args.epochs, args.log_interval, False)
        val_metrics = run_epoch(model, prepared["val_loader"], None, device, "val", epoch, args.epochs, args.log_interval, False)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_to_best_val = epoch
    train_time_sec = time.perf_counter() - train_start

    test_metrics = run_epoch(model, prepared["test_loader"], None, device, "test", args.epochs, args.epochs, args.log_interval, False)
    inference_time_sec = _benchmark_inference(model, prepared["test_loader"], device, args.inference_repeats)

    row_max_token_len = args.max_token_len if tokenizer_info["tokenizer_type"] == "wordpiece" else None
    row_pad_layout = args.pad_layout if tokenizer_info["tokenizer_type"] == "wordpiece" else "end"

    tokenizer_label = "char" if tokenizer_info["tokenizer_type"] == "char" else f"wordpiece_{vocab_size}"
    filter_suffix = f"_n{row_max_token_len}" if row_max_token_len is not None else ""
    layout_suffix = f"_{row_pad_layout}" if row_pad_layout != "end" else ""
    # Architecture applies to every tokenizer (char included) - only stamp the filename
    # when it differs from ForwardModel's own defaults, so default-architecture runs keep
    # producing the exact same filenames as before this option existed (resume stays intact).
    arch_suffix = (
        f"_h{args.hidden_dim}_d{args.num_conv_blocks}"
        if (args.hidden_dim != DEFAULT_HIDDEN_DIM or args.num_conv_blocks != DEFAULT_NUM_CONV_BLOCKS)
        else ""
    )
    history_dir = Path(args.output_root) / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / f"{tokenizer_label}{filter_suffix}{layout_suffix}{arch_suffix}_seed{seed}.json"
    save_json(history_path, history)
    logger.info("wrote per-epoch history path=%s", history_path)

    return {
        "tokenizer": tokenizer_info["tokenizer_type"],
        "vocab_size": tokenizer_info["vocab_size"],
        "max_token_len": row_max_token_len,
        "pad_layout": row_pad_layout,
        "hidden_dim": args.hidden_dim,
        "num_conv_blocks": args.num_conv_blocks,
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
    """One aggregated row per (tokenizer, vocab_size, max_token_len, pad_layout, hidden_dim,
    num_conv_blocks), mean +/- std across seeds."""
    groups = {}
    for row in rows:
        key = (
            row["tokenizer"], row["vocab_size"], row["max_token_len"], row.get("pad_layout", "end"),
            row.get("hidden_dim", DEFAULT_HIDDEN_DIM), row.get("num_conv_blocks", DEFAULT_NUM_CONV_BLOCKS),
        )
        groups.setdefault(key, []).append(row)

    summary = []
    for (tokenizer, vocab_size, max_token_len, pad_layout, hidden_dim, num_conv_blocks), group_rows in sorted(
        groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0, kv[0][2] or 0, kv[0][3], kv[0][4], kv[0][5])
    ):
        summary.append({
            "tokenizer": tokenizer,
            "vocab_size": vocab_size,
            "max_token_len": max_token_len,
            "pad_layout": pad_layout,
            "hidden_dim": hidden_dim,
            "num_conv_blocks": num_conv_blocks,
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
        "| tokenizer | vocab_size | max_token_len | pad_layout | hidden_dim | num_conv_blocks | max_len | "
        "test_loss | test_cosine | best_val_loss | "
        "epochs_to_best_val | train_time_sec | inference_time_sec | num_parameters | notes |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in summary:
        notes = f"n={row['n_seeds']} seeds"
        max_token_len = row["max_token_len"] if row["max_token_len"] is not None else "-"
        lines.append(
            f"| {row['tokenizer']} | {row['vocab_size']} | {max_token_len} | {row['pad_layout']} | "
            f"{row['hidden_dim']} | {row['num_conv_blocks']} | {row['max_len']} | "
            f"{_fmt(row['test_loss'])} | {_fmt(row['test_cosine'])} | {_fmt(row['best_val_loss'])} | "
            f"{_fmt(row['epochs_to_best_val'], precision=1)} | {_fmt(row['train_time_sec'], precision=1)} | "
            f"{_fmt(row['inference_time_sec'], precision=3)} | {row['num_parameters']} | {notes} |"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_completed_runs(csv_path):
    """(tokenizer_path or None, max_token_len, pad_layout, hidden_dim, num_conv_blocks, seed) tuples
    already present in raw_results.csv."""
    if not csv_path.exists():
        return set(), []
    with open(csv_path, newline="", encoding="utf-8") as f:
        existing_rows = list(csv.DictReader(f))
    # .get(..., default) covers resuming from a CSV written before these columns existed
    completed = {
        (
            row["tokenizer_path"] or None,
            int(row["max_token_len"]) if row.get("max_token_len") else None,
            row.get("pad_layout") or "end",
            int(row["hidden_dim"]) if row.get("hidden_dim") else DEFAULT_HIDDEN_DIM,
            int(row["num_conv_blocks"]) if row.get("num_conv_blocks") else DEFAULT_NUM_CONV_BLOCKS,
            int(row["seed"]),
        )
        for row in existing_rows
    }
    # csv.DictReader gives strings back; downstream code (summarize()) expects numeric fields
    for row in existing_rows:
        row["vocab_size"] = int(row["vocab_size"])
        row["max_token_len"] = int(row["max_token_len"]) if row.get("max_token_len") else None
        row["pad_layout"] = row.get("pad_layout") or "end"
        row["hidden_dim"] = int(row["hidden_dim"]) if row.get("hidden_dim") else DEFAULT_HIDDEN_DIM
        row["num_conv_blocks"] = int(row["num_conv_blocks"]) if row.get("num_conv_blocks") else DEFAULT_NUM_CONV_BLOCKS
        row["max_len"] = int(row["max_len"])
        row["seed"] = int(row["seed"])
        row["test_loss"] = float(row["test_loss"])
        row["test_cosine"] = float(row["test_cosine"])
        row["best_val_loss"] = float(row["best_val_loss"])
        row["epochs_to_best_val"] = int(row["epochs_to_best_val"]) if row["epochs_to_best_val"] else None
        row["train_time_sec"] = float(row["train_time_sec"])
        row["inference_time_sec"] = float(row["inference_time_sec"])
        row["num_parameters"] = int(row["num_parameters"])
        row["tokenizer_path"] = row["tokenizer_path"] or None
    return completed, existing_rows


def _spec_key(spec, args):
    """(tokenizer_path or None, max_token_len, pad_layout, hidden_dim, num_conv_blocks) identity
    matched against a CSV row."""
    if spec == "char":
        return None, None, "end", args.hidden_dim, args.num_conv_blocks
    path = spec.split(":", 1)[1]
    return path, args.max_token_len, args.pad_layout, args.hidden_dim, args.num_conv_blocks


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_root / "benchmark.log")

    csv_path = output_root / "raw_results.csv"
    completed, rows = _load_completed_runs(csv_path)
    if rows:
        logger.info("resuming: %d run(s) already completed in %s", len(rows), csv_path)

    # Open once in append mode (or fresh with header) and flush after every row, so a
    # killed/interrupted process only loses the run currently in flight, not prior ones.
    write_header = not csv_path.exists()
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=RESULT_FIELDS)
    if write_header:
        writer.writeheader()
        csv_file.flush()

    try:
        for spec in args.configs:
            key = _spec_key(spec, args)
            for seed in args.seeds:
                if (*key, seed) in completed:
                    logger.info("skipping already-completed config=%s seed=%d max_token_len=%s", spec, seed, args.max_token_len)
                    continue
                logger.info("running config=%s seed=%d max_token_len=%s", spec, seed, args.max_token_len)
                row = run_one(spec, seed, args, logger)
                rows.append(row)
                writer.writerow(row)
                csv_file.flush()
                logger.info(
                    "done config=%s seed=%d test_loss=%.4f test_cosine=%.4f",
                    spec, seed, row["test_loss"], row["test_cosine"],
                )
    finally:
        csv_file.close()

    summary = summarize(rows)
    save_json(output_root / "summary.json", summary)
    write_markdown_table(summary, output_root / "summary_table.md")
    logger.info("wrote raw_results.csv, summary.json, summary_table.md to %s", output_root)


if __name__ == "__main__":
    main()