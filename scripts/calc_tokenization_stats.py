"""Compute WordPiece tokenization statistics on train/val/test splits.

For each vocab_size tokenizer, computes required length/UNK/pad/compression
statistics per split, top-token reports on the train split, a cross-vocab-size
example table, and alarm checks (over-large vocab, misconfigured pre-tokenizer,
insufficient compression benefit).

Uses the same read_airr_table + split_indices(seed) call as
prepare_wordpiece_corpus.py / batch_cache.py, so split membership here matches
what models are trained/evaluated on.
"""
import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from irrm_codec.dataio import read_airr_table
from irrm_codec.utils import setup_logging, split_indices

try:
    import matplotlib
    matplotlib.use("Agg")  # headless-safe, no display required
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

SPLIT_NAMES = ("train", "val", "test")
MAIN_GRID_VOCAB_SIZES = (1000, 2000, 5000, 10000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute WordPiece tokenization statistics.")
    parser.add_argument("--airr-path", required=True)
    parser.add_argument("--locus", default="beta")
    parser.add_argument("--clone-id-col", default="")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizers-dir", default="artifacts/tokenizers")
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=[100, 1000, 2000, 5000, 10000])
    parser.add_argument("--example-vocab-sizes", type=int, nargs="+", default=list(MAIN_GRID_VOCAB_SIZES))
    parser.add_argument("--output-path", default="artifacts/tokenizers/tokenization_stats.json")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--num-examples", type=int, default=20)
    parser.add_argument("--max-len", type=int, default=40, help="Fixed length used for pad_token_fraction.")
    parser.add_argument("--high-1-token-threshold", type=float, default=0.2)
    parser.add_argument("--high-unk-threshold", type=float, default=0.001)
    parser.add_argument("--min-compression-benefit", type=float, default=0.2)
    parser.add_argument("--skip-plots", action="store_true", help="Do not generate the two diagnostic PNG plots.")
    return parser.parse_args()


def load_split_sequences(
    airr_path: str, locus: str, clone_id_col: str, train_fraction: float, val_fraction: float, seed: int,
    logger: logging.Logger,
) -> dict[str, list[str]]:
    columns = ["junction_aa", "locus"]
    if clone_id_col:
        columns.append(clone_id_col)

    df = read_airr_table(
        airr_path, clone_id_col=clone_id_col, columns=list(dict.fromkeys(columns)), validate=False,
    )
    if locus is not None:
        # exact match, no LOCUS_ALIASES normalization - must mirror batch_cache.py
        # so num_rows and split_indices() output match what models train on
        df = df[df["locus"] == locus].reset_index(drop=True)

    seqs = df["junction_aa"].astype(str).to_numpy(copy=True)
    train_idx, val_idx, test_idx = split_indices(
        len(seqs), train_fraction=train_fraction, val_fraction=val_fraction, seed=seed,
    )
    logger.info("split ready train=%d val=%d test=%d", len(train_idx), len(val_idx), len(test_idx))

    return {
        "train": seqs[train_idx].tolist(),
        "val": seqs[val_idx].tolist(),
        "test": seqs[test_idx].tolist(),
    }


def encode_raw(tokenizer: Tokenizer, seqs: list[str]) -> list:
    # tokenizer object is reused across splits/calls, a prior call may have left padding on
    tokenizer.no_padding()
    tokenizer.no_truncation()
    return tokenizer.encode_batch(seqs)


def count_tokens_per_sequence(encodings: list) -> list[int]:
    token_counts = []
    for encoding in encodings:
        token_counts.append(len(encoding.ids))
    return token_counts


def count_unk_tokens(encodings: list, unk_id: int) -> tuple[int, int, int]:
    total_tokens = 0
    total_unk_tokens = 0
    sequences_with_unk = 0

    for encoding in encodings:
        unk_count_in_this_sequence = 0
        for token_id in encoding.ids:
            total_tokens += 1
            if token_id == unk_id:
                total_unk_tokens += 1
                unk_count_in_this_sequence += 1
        if unk_count_in_this_sequence > 0:
            sequences_with_unk += 1

    return total_tokens, total_unk_tokens, sequences_with_unk


def count_short_sequences(token_counts: list[int]) -> tuple[int, int, int]:
    count_1_token = 0
    count_2_tokens_or_less = 0
    count_3_tokens_or_less = 0

    for num_tokens in token_counts:
        if num_tokens <= 1:
            count_1_token += 1
        if num_tokens <= 2:
            count_2_tokens_or_less += 1
        if num_tokens <= 3:
            count_3_tokens_or_less += 1

    return count_1_token, count_2_tokens_or_less, count_3_tokens_or_less


def summarize_token_counts(token_counts: list[int]) -> dict:
    token_counts_array = np.array(token_counts)
    return {
        "mean_num_tokens": float(token_counts_array.mean()),
        "median_num_tokens": float(np.median(token_counts_array)),
        "min_num_tokens": int(token_counts_array.min()),
        "max_num_tokens": int(token_counts_array.max()),
        "p95_num_tokens": float(np.percentile(token_counts_array, 95)),
    }


def count_pad_tokens(tokenizer: Tokenizer, seqs: list[str], pad_id: int, max_len: int) -> tuple[int, int]:
    # length=max_len mirrors what collate_forward/collate_inverse actually feed the model
    tokenizer.enable_padding(pad_id=pad_id, pad_token="[PAD]", length=max_len)
    tokenizer.enable_truncation(max_length=max_len)
    padded_encodings = tokenizer.encode_batch(seqs)

    total_positions = 0
    total_pad_positions = 0
    for encoding in padded_encodings:
        for token_id in encoding.ids:
            total_positions += 1
            if token_id == pad_id:
                total_pad_positions += 1

    # reset so later calls on this tokenizer (other splits, top tokens, examples) get raw output
    tokenizer.no_padding()
    tokenizer.no_truncation()
    return total_positions, total_pad_positions


def compute_split_stats(
    tokenizer: Tokenizer,
    seqs: list[str],
    unk_id: int,
    pad_id: int,
    max_len: int,
    mean_char_length: float,
) -> tuple[dict, list]:
    num_sequences = len(seqs)

    encodings = encode_raw(tokenizer, seqs)
    token_counts = count_tokens_per_sequence(encodings)
    total_tokens, total_unk_tokens, sequences_with_unk = count_unk_tokens(encodings, unk_id)
    count_1, count_2, count_3 = count_short_sequences(token_counts)
    total_positions, total_pad_positions = count_pad_tokens(tokenizer, seqs, pad_id, max_len)

    mean_num_tokens = float(np.mean(token_counts))
    compression_ratio = mean_num_tokens / mean_char_length if mean_char_length > 0 else float("nan")

    stats = {
        "number_of_sequences": num_sequences,
        **summarize_token_counts(token_counts),
        "unk_token_fraction": total_unk_tokens / total_tokens if total_tokens > 0 else 0.0,
        "unk_sequence_fraction": sequences_with_unk / num_sequences,
        "pad_token_fraction": total_pad_positions / total_positions if total_positions > 0 else 0.0,
        "fraction_encoded_as_1_token": count_1 / num_sequences,
        "fraction_encoded_as_2_tokens_or_less": count_2 / num_sequences,
        "fraction_encoded_as_3_tokens_or_less": count_3 / num_sequences,
        "mean_char_length": mean_char_length,
        "compression_ratio": compression_ratio,
    }
    return stats, encodings


def aa_length(token: str) -> int:
    # "##" is a continuation marker, not an amino acid - strip before measuring length
    if token.startswith("##"):
        return len(token[2:])
    return len(token)


def compute_top_token_reports(tokenizer: Tokenizer, train_encodings: list, top_n: int) -> dict:
    token_counter: Counter = Counter()
    for encoding in train_encodings:
        for token_id in encoding.ids:
            token_counter[token_id] += 1

    all_seen_tokens = []
    for token_id, count in token_counter.items():
        token_text = tokenizer.id_to_token(token_id)
        all_seen_tokens.append(
            {"token": token_text, "id": token_id, "count": count, "aa_length": aa_length(token_text)}
        )

    top_by_frequency = [
        {"token": tokenizer.id_to_token(tid), "id": tid, "count": count}
        for tid, count in token_counter.most_common(top_n)
    ]

    top_longest = sorted(all_seen_tokens, key=lambda item: (-item["aa_length"], -item["count"]))[:top_n]

    non_single_aa_tokens = [item for item in all_seen_tokens if item["aa_length"] > 1]
    top_non_single_aa = sorted(non_single_aa_tokens, key=lambda item: -item["count"])[:top_n]

    return {
        "top_tokens_by_frequency_train": top_by_frequency,
        "top_longest_tokens": top_longest,
        "top_non_single_amino_acid_tokens": top_non_single_aa,
    }


def compute_alarms(
    train_stats: dict,
    high_1_token_threshold: float,
    high_unk_threshold: float,
    min_compression_benefit: float,
) -> dict:
    # train-only: alarms describe tokenizer fit behavior, not generalization -
    # compare train vs val/test in `splits` manually (memorization looks fine on
    # train alarms but shows up as a train/val gap, e.g. vocab_size=10000 case)
    fraction_1_token = train_stats["fraction_encoded_as_1_token"]
    unk_fraction = train_stats["unk_token_fraction"]
    compression_ratio = train_stats["compression_ratio"]

    if compression_ratio == compression_ratio:  # not NaN
        compression_benefit = 1.0 - compression_ratio
    else:
        compression_benefit = float("nan")

    return {
        "vocab_size_too_large": {
            "triggered": fraction_1_token > high_1_token_threshold,
            "value": fraction_1_token,
            "threshold": high_1_token_threshold,
            "message": (
                f"{fraction_1_token:.1%} of train CDR3s encode as a single token "
                f"(threshold {high_1_token_threshold:.1%}); vocab_size may be too large."
            ),
        },
        "tokenizer_misconfigured": {
            "triggered": unk_fraction > high_unk_threshold,
            "value": unk_fraction,
            "threshold": high_unk_threshold,
            "message": (
                f"unk_token_fraction={unk_fraction:.4%} exceeds threshold {high_unk_threshold:.4%}; "
                "check pre_tokenizer / alphabet coverage."
            ),
        },
        "insufficient_compression_benefit": {
            "triggered": compression_benefit < min_compression_benefit,
            "value": compression_benefit,
            "threshold": min_compression_benefit,
            "message": (
                f"WordPiece reduces sequence length by only {compression_benefit:.1%} "
                f"vs raw amino acid count (threshold {min_compression_benefit:.1%}); "
                "benefit over char tokenization may be marginal."
            ),
        },
    }


def build_examples_table(tokenizers_by_vocab_size: dict, example_seqs: list[str]) -> list[dict]:
    examples = []
    for seq in example_seqs:
        row = {"original_cdr3": seq}
        for vocab_size, tokenizer in tokenizers_by_vocab_size.items():
            # reset in case this tokenizer object still has padding enabled from compute_split_stats
            tokenizer.no_padding()
            tokenizer.no_truncation()
            encoding = tokenizer.encode(seq)
            row[f"tokens_vocab_{vocab_size}"] = encoding.tokens
        examples.append(row)
    return examples


RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
BOLD = "\033[1m"
RESET = "\033[0m"

GENERALIZATION_GAP_THRESHOLD = 0.05

ROW_FORMAT = "{:<12}{:<7}{:>9}{:>9}{:>10}{:>10}{:>10}{:>10}"


def colorize(text: str, color: str) -> str:
    return f"{color}{text}{RESET}"


def build_summary_rows(results: dict) -> list[tuple]:
    # shared row classification (alarm / generalization-gap flags), consumed by
    # both the console renderer and the plain-text file renderer below
    groups = []
    for vocab_size, entry in results["per_vocab_size"].items():
        train_stats = entry["splits"]["train"]
        train_frac_1_token = train_stats["fraction_encoded_as_1_token"]
        any_alarm_this_vocab = any(alarm["triggered"] for alarm in entry["alarms"].values())

        rows = []
        for split_name in SPLIT_NAMES:
            stats = entry["splits"][split_name]
            is_alarm_row = split_name == "train" and any_alarm_this_vocab

            gap_warning = False
            if split_name != "train":
                # large gap vs train means the tokenizer memorized train sequences
                # rather than learning generalizable subwords (see vocab_size=10000 case)
                gap = abs(train_frac_1_token - stats["fraction_encoded_as_1_token"])
                gap_warning = gap > GENERALIZATION_GAP_THRESHOLD

            rows.append((split_name, stats, is_alarm_row, gap_warning))

        groups.append((vocab_size, any_alarm_this_vocab, rows))
    return groups


def format_row(vocab_size: str, split_name: str, stats: dict) -> str:
    return ROW_FORMAT.format(
        vocab_size,
        split_name,
        f"{stats['mean_num_tokens']:.2f}",
        f"{stats['compression_ratio']:.3f}",
        f"{stats['unk_token_fraction']:.4f}",
        f"{stats['pad_token_fraction']:.3f}",
        f"{stats['fraction_encoded_as_1_token']:.4f}",
        f"{stats['fraction_encoded_as_3_tokens_or_less']:.3f}",
    )


def print_colored_summary(results: dict) -> None:
    groups = build_summary_rows(results)
    header = ROW_FORMAT.format(
        "vocab_size", "split", "mean_tok", "compr", "unk_frac", "pad_frac", "frac=1tok", "frac<=3tok",
    )

    print()
    print(BOLD + "=== Tokenization statistics summary ===" + RESET)
    print(BOLD + header + RESET)
    print("-" * len(header))

    any_alarm_triggered = False
    for vocab_size, any_alarm_this_vocab, rows in groups:
        any_alarm_triggered = any_alarm_triggered or any_alarm_this_vocab
        for split_name, stats, is_alarm_row, gap_warning in rows:
            row_text = format_row(vocab_size, split_name, stats)
            if is_alarm_row:
                print(colorize(row_text, RED))
            elif gap_warning:
                print(colorize(row_text, YELLOW))
            else:
                print(row_text)
        print()

    print(BOLD + "=== Alarms ===" + RESET)
    if not any_alarm_triggered:
        print(colorize("No alarms triggered.", GREEN))
    else:
        for vocab_size, entry in results["per_vocab_size"].items():
            for alarm_name, alarm in entry["alarms"].items():
                if alarm["triggered"]:
                    print(colorize(f"  [{vocab_size}] {alarm_name}: {alarm['message']}", RED))
    print()
    print(
        colorize(
            f"Yellow rows: |train_frac_1_token - split_frac_1_token| > {GENERALIZATION_GAP_THRESHOLD:.0%} "
            "(possible tokenizer memorization of train sequences).",
            YELLOW,
        )
    )


def save_plain_summary(results: dict, path: Path) -> None:
    # same content as print_colored_summary, no ANSI codes - readable in any text viewer.
    # flagged rows get a bracket marker instead of color, since plain text has no color.
    groups = build_summary_rows(results)
    header = ROW_FORMAT.format(
        "vocab_size", "split", "mean_tok", "compr", "unk_frac", "pad_frac", "frac=1tok", "frac<=3tok",
    )

    lines = ["=== Tokenization statistics summary ===", header, "-" * len(header)]

    any_alarm_triggered = False
    for vocab_size, any_alarm_this_vocab, rows in groups:
        any_alarm_triggered = any_alarm_triggered or any_alarm_this_vocab
        for split_name, stats, is_alarm_row, gap_warning in rows:
            row_text = format_row(vocab_size, split_name, stats)
            if is_alarm_row:
                row_text += "  [ALARM]"
            elif gap_warning:
                row_text += "  [GAP]"
            lines.append(row_text)
        lines.append("")

    lines.append("=== Alarms ===")
    if not any_alarm_triggered:
        lines.append("No alarms triggered.")
    else:
        for vocab_size, entry in results["per_vocab_size"].items():
            for alarm_name, alarm in entry["alarms"].items():
                if alarm["triggered"]:
                    lines.append(f"  [{vocab_size}] {alarm_name}: {alarm['message']}")
    lines.append("")
    lines.append(
        f"[GAP] rows: |train_frac_1_token - split_frac_1_token| > {GENERALIZATION_GAP_THRESHOLD:.0%} "
        "(possible tokenizer memorization of train sequences)."
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def plot_train_memorization(vocab_sizes: list[int], per_vocab_size: dict, alarm_threshold: float, output_path: Path) -> None:
    train_frac_1_token = [per_vocab_size[str(vs)]["splits"]["train"]["fraction_encoded_as_1_token"] for vs in vocab_sizes]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(vocab_sizes, train_frac_1_token, marker="o", color="tab:red")
    ax.axhline(alarm_threshold, color="gray", linestyle="--", label=f"alarm threshold ({alarm_threshold:.0%})")
    ax.set_xlabel("vocab_size")
    ax.set_ylabel("fraction_encoded_as_1_token (train)")
    ax.set_title("Train-set tokenizer memorization vs vocab_size")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_val_test_compression(vocab_sizes: list[int], per_vocab_size: dict, output_path: Path) -> None:
    val_mean_tokens = [per_vocab_size[str(vs)]["splits"]["val"]["mean_num_tokens"] for vs in vocab_sizes]
    test_mean_tokens = [per_vocab_size[str(vs)]["splits"]["test"]["mean_num_tokens"] for vs in vocab_sizes]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(vocab_sizes, val_mean_tokens, marker="o", label="val", color="tab:blue")
    ax.plot(vocab_sizes, test_mean_tokens, marker="o", label="test", color="tab:orange")
    ax.set_xlabel("vocab_size")
    ax.set_ylabel("mean_num_tokens (held-out)")
    ax.set_title("Real generalization benefit vs vocab_size (val/test)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_diagnostic_plots(results: dict, output_dir: Path, alarm_threshold: float, logger: logging.Logger) -> None:
    # needs at least 2 vocab_size points to draw a meaningful line
    vocab_sizes = sorted(int(vs) for vs in results["per_vocab_size"].keys())
    if len(vocab_sizes) < 2:
        logger.warning("fewer than 2 vocab_sizes computed, skipping diagnostic plots")
        return

    per_vocab_size = results["per_vocab_size"]

    train_plot_path = output_dir / "train_memorization_vs_vocab_size.png"
    plot_train_memorization(vocab_sizes, per_vocab_size, alarm_threshold, train_plot_path)
    logger.info("saved plot path=%s", train_plot_path)

    valtest_plot_path = output_dir / "val_test_compression_vs_vocab_size.png"
    plot_val_test_compression(vocab_sizes, per_vocab_size, valtest_plot_path)
    logger.info("saved plot path=%s", valtest_plot_path)


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path)
    logger = setup_logging(output_path.parent / "compute_tokenization_stats.log")

    split_seqs = load_split_sequences(
        args.airr_path, args.locus, args.clone_id_col,
        args.train_fraction, args.val_fraction, args.seed, logger,
    )
    mean_char_length_by_split = {
        split_name: float(np.mean([len(seq) for seq in seqs])) for split_name, seqs in split_seqs.items()
    }

    tokenizers_dir = Path(args.tokenizers_dir)
    results = {"per_vocab_size": {}, "examples": [], "max_len_used_for_padding_stats": args.max_len}
    example_tokenizers = {}

    for vocab_size in args.vocab_sizes:
        tokenizer_path = tokenizers_dir / f"wordpiece_vocab_{vocab_size}" / "tokenizer.json"
        if not tokenizer_path.exists():
            logger.warning("tokenizer file not found, skipping: %s", tokenizer_path)
            continue

        logger.info("computing stats for vocab_size=%d", vocab_size)
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        vocab = tokenizer.get_vocab()
        unk_id = vocab["[UNK]"]
        pad_id = vocab["[PAD]"]
        actual_vocab_size = tokenizer.get_vocab_size()

        split_results = {}
        train_encodings = None
        for split_name in SPLIT_NAMES:
            seqs = split_seqs[split_name]
            stats, encodings = compute_split_stats(
                tokenizer, seqs, unk_id, pad_id, args.max_len, mean_char_length_by_split[split_name],
            )
            split_results[split_name] = stats
            if split_name == "train":
                train_encodings = encodings
            logger.info(
                "vocab_size=%d split=%s mean_num_tokens=%.2f unk_token_fraction=%.6f "
                "pad_token_fraction=%.4f fraction_1_token=%.4f",
                vocab_size, split_name, stats["mean_num_tokens"], stats["unk_token_fraction"],
                stats["pad_token_fraction"], stats["fraction_encoded_as_1_token"],
            )

        top_token_reports = compute_top_token_reports(tokenizer, train_encodings, args.top_n)
        alarms = compute_alarms(
            split_results["train"], args.high_1_token_threshold, args.high_unk_threshold,
            args.min_compression_benefit,
        )
        for alarm_name, alarm in alarms.items():
            if alarm["triggered"]:
                logger.warning("ALARM vocab_size=%d %s: %s", vocab_size, alarm_name, alarm["message"])

        results["per_vocab_size"][str(vocab_size)] = {
            "role": "sanity_check" if vocab_size == 100 else "main_grid",
            "vocab_size_requested": vocab_size,
            "vocab_size_actual": actual_vocab_size,
            "splits": split_results,
            **top_token_reports,
            "alarms": alarms,
        }

        if vocab_size in args.example_vocab_sizes:
            example_tokenizers[vocab_size] = tokenizer

    example_seqs = split_seqs["test"][: args.num_examples]
    if len(example_seqs) < args.num_examples:
        example_seqs = (split_seqs["test"] + split_seqs["val"])[: args.num_examples]
    results["examples"] = build_examples_table(example_tokenizers, example_seqs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("wrote tokenization stats path=%s", output_path)

    summary_path = output_path.parent / "tokenization_stats_summary.txt"
    save_plain_summary(results, summary_path)
    logger.info("wrote plain-text summary path=%s", summary_path)

    if args.skip_plots:
        logger.info("--skip-plots set, not generating diagnostic plots")
    elif not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not installed, skipping diagnostic plots (pip install matplotlib)")
    else:
        save_diagnostic_plots(results, output_path.parent, args.high_1_token_threshold, logger)

    print_colored_summary(results)


if __name__ == "__main__":
    main()